"""LLM-based document metadata classification (company_id, document_type, period)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from neo4j import Driver
from pydantic import BaseModel, Field

from src.config import Settings
from src.llm.gemini import build_fast_model

logger = logging.getLogger(__name__)


class DocumentMetadataOutput(BaseModel):
    """Structured extraction result for company, document type, and fiscal period."""

    company_name: str = Field(
        description="Detected company name (e.g., NVIDIA, Apple Inc)"
    )
    ticker: str = Field(
        description="Detected ticker symbol (e.g., NVDA, AAPL). Return empty string if not found."
    )
    period_label: str = Field(
        default="",
        description=(
            "Fiscal period as found in the document, e.g. 'Q3 FY2025', "
            "'Three Months Ended October 27, 2024'. Return empty string if not found."
        ),
    )
    document_type: Literal[
        "annual_report",
        "quarterly_report",
        "earnings_release",
        "earnings_report",
        "press_release",
        "investor_presentation",
        "research_report",
        "regulatory_filing",
        "news_article",
        "uploaded_document",
    ] = Field(
        description=(
            "Document type — choose the closest match from the allowed values. "
            "Use 'uploaded_document' when the type cannot be determined."
        )
    )
    confidence: float = Field(
        description="Confidence score 0-1. 0.5+ is reliable, <0.3 indicates low certainty."
    )


@dataclass(frozen=True)
class DetectedMetadata:
    """Immutable result of metadata classification."""

    company_id: str  # ticker if found, else company_name
    company_name: str
    document_type: str
    confidence: float
    detected_by: str  # "llm" or "failed"
    period_label: str | None = None  # e.g. "Q3 FY2025", "Three Months Ended Oct 27 2024"


@dataclass(frozen=True)
class ExistingCompany:
    """Canonical company entry read from the graph."""

    company_id: str
    name: str

_COMPANY_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "LTD",
    "LLC",
    "PLC",
    "HOLDING",
    "HOLDINGS",
}
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        ".": " ",
        ",": " ",
        "-": " ",
        "_": " ",
        "/": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "'": " ",
        '"': " ",
        "&": " ",
        ":": " ",
        ";": " ",
    }
)


def _normalize_company_label(value: str) -> str:
    cleaned = value.strip().upper().translate(_PUNCTUATION_TRANSLATION)
    tokens = [token for token in cleaned.split() if token.isalnum()]
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def list_existing_companies(driver: Driver, settings: Settings) -> list[ExistingCompany]:
    """Return existing canonical company IDs from Neo4j."""
    cypher = (
        "MATCH (co:Company) "
        "RETURN co.company_id AS company_id, co.name AS name "
        "ORDER BY co.company_id"
    )
    with driver.session(database=settings.neo4j_database) as session:
        rows = session.run(cypher).data()

    companies: list[ExistingCompany] = []
    for row in rows:
        company_id = str(row.get("company_id") or "").strip()
        if not company_id:
            continue
        companies.append(
            ExistingCompany(
                company_id=company_id,
                name=str(row.get("name") or company_id).strip() or company_id,
            )
        )
    return companies


def _resolve_to_existing_company_id(
    proposed_company_id: str,
    proposed_company_name: str,
    existing_companies: list[ExistingCompany],
) -> str:
    """Map a detected company to an existing canonical ID when possible."""
    if not existing_companies:
        return proposed_company_id

    candidates = {
        _normalize_company_label(proposed_company_id),
        _normalize_company_label(proposed_company_name),
    }
    candidates.discard("")
    if not candidates:
        return proposed_company_id

    for company in existing_companies:
        existing_labels = {
            _normalize_company_label(company.company_id),
            _normalize_company_label(company.name),
        }
        if candidates & existing_labels:
            return company.company_id

    return proposed_company_id


def _company_catalog_for_prompt(existing_companies: list[ExistingCompany]) -> str:
    if not existing_companies:
        return "No existing companies are currently stored."
    lines = [
        f"- company_id: {company.company_id}; company_name: {company.name}"
        for company in existing_companies[:120]
    ]
    return "\n".join(lines)


def classify_metadata(
    text: str,
    settings: Settings,
    parsed_doc = None,  # ParsedDocument for multi-page extraction
    existing_companies: list[ExistingCompany] | None = None,
) -> DetectedMetadata:
    """Classify document metadata using LLM and DB canonicalization.

    Args:
        text: Document excerpt or parsed full-text
        settings: Configuration with LLM settings
        parsed_doc: Optional ParsedDocument for multi-page context extraction
    Returns:
        DetectedMetadata with company_id, document_type, period_label, confidence
    """
    # Build multi-page context if available.
    context = _build_classifier_context(text, parsed_doc) if parsed_doc else text

    try:
        llm_result = _classify_with_llm(
            context,
            settings,
            existing_companies=existing_companies or [],
        )
        company_id = llm_result.company_id
        company_name = llm_result.company_name

        if existing_companies and (company_id or company_name):
            company_id = _resolve_to_existing_company_id(
                proposed_company_id=company_id,
                proposed_company_name=company_name,
                existing_companies=existing_companies,
            )

        result = DetectedMetadata(
            company_id=company_id,
            company_name=company_name,
            document_type=llm_result.document_type,
            confidence=llm_result.confidence,
            detected_by="llm",
            period_label=llm_result.period_label,
        )
        logger.debug(
            f"Classification: {result.company_id} / {result.document_type} "
            f"/ period={result.period_label!r} (confidence={result.confidence})"
        )
        return result
    except Exception as e:
        logger.warning(f"LLM classification failed ({e}), using empty metadata")
        return DetectedMetadata(
            company_id="",
            company_name="",
            document_type="uploaded_document",
            confidence=0.0,
            detected_by="failed",
            period_label=None,
        )


def _build_classifier_context(text: str, parsed_doc) -> str:
    """Build extended context for classification from multiple pages.

    Fiscal period end dates appear on:
    - Cover page (page 1) — "For the quarterly period ended October 27, 2024"
    - Inside cover / page 2-3 — key metadata
    - Final notes — sometimes repeated

    Args:
        text: Initial text excerpt (first page)
        parsed_doc: ParsedDocument with metadata.get("pages")

    Returns:
        Expanded context combining cover page + key metadata pages
    """
    pages = parsed_doc.metadata.get("pages", []) if parsed_doc else []

    if not pages:
        # Flat text — take first 4000 chars (doubled from current 2000)
        return text[:4000]

    # Cover page (page 1) + pages 2-3 + last page for period info
    candidate_pages = pages[:3]
    if len(pages) > 3:
        candidate_pages.append(pages[-1])  # last page sometimes has fiscal period in notes

    # Extract text from candidate pages, capping each at 800 chars
    combined_parts = []
    for page in candidate_pages:
        page_text = page.get("text", "")
        if page_text:
            combined_parts.append(page_text[:800])

    combined = "\n\n".join(combined_parts)
    return combined[:5000]  # cap at 5000 chars for LLM efficiency

def _classify_with_llm(
    text: str,
    settings: Settings,
    existing_companies: list[ExistingCompany],
) -> DetectedMetadata:
    """LLM-based structured classification using with_structured_output()."""
    llm = build_fast_model(settings)
    structured_llm = llm.with_structured_output(DocumentMetadataOutput)

    company_catalog = _company_catalog_for_prompt(existing_companies)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a financial document metadata extractor. "
                    "From the text excerpt, identify the company name, ticker symbol, "
                    "document type, and fiscal period.\n\n"
                    "RULES:\n"
                    "1. Extract only information explicitly stated in the text.\n"
                    "2. confidence: 0.9+ = explicitly stated, 0.5–0.8 = inferred, "
                    "<0.3 = cannot determine.\n"
                    "3. If the ticker is not in the text, return an empty string for ticker.\n"
                    "4. For period_label: extract the fiscal quarter/period as written, "
                    "e.g. 'Q3 FY2025', 'Three Months Ended October 27, 2024'. "
                    "Return empty string if not found.\n"
                    "5. Choose the document_type that best matches from the schema's "
                    "allowed values.\n"
                    "6. Reuse an existing canonical company_id when the document refers "
                    "to the same company with wording variants. Existing companies:\n"
                    "{company_catalog}"
                ),
            ),
            (
                "user",
                "Extract metadata from this document excerpt:\n\n{text}",
            ),
        ]
    )

    runnable = prompt | structured_llm
    raw_result: Any = runnable.invoke({"text": text, "company_catalog": company_catalog})
    if isinstance(raw_result, DocumentMetadataOutput):
        result = raw_result
    elif isinstance(raw_result, BaseModel):
        result = DocumentMetadataOutput.model_validate(raw_result.model_dump())
    else:
        result = DocumentMetadataOutput.model_validate(raw_result)

    company_name = result.company_name.strip()
    ticker = result.ticker.strip().upper()

    # LLM-only normalization: prefer ticker, else fall back to detected company name.
    company_id = ticker if ticker else company_name

    # Ensure document_type is snake_case
    doc_type = result.document_type.lower().replace(" ", "_").replace("-", "_")

    period_label = result.period_label.strip() if result.period_label else None

    return DetectedMetadata(
        company_id=company_id or "",
        company_name=company_name,
        document_type=doc_type or "uploaded_document",
        confidence=result.confidence,
        detected_by="llm",
        period_label=period_label if period_label else None,
    )
