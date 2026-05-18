from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

try:
    import pdfplumber  # type: ignore[import-untyped]
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

from src.config import Settings
from src.ingestion.models import DocumentInput, ParsedDocument


# ---------------------------------------------------------------------------
# Section boundary patterns for financial documents (Yang et al. 2024)
# Groups pages into logical sections so tables stay bonded to their headers.
# ---------------------------------------------------------------------------
_SECTION_BOUNDARY_RE = re.compile(
    r"|".join([
        # 10-Q / 10-K structure
        r"^part\s+[i]{1,3}v?\b",          # Part I, II, III, IV
        r"^item\s+\d+[a-z]?\.",            # Item 1., Item 1A.
        r"management.{0,10}discussion",
        r"^results of operations",
        r"liquidity and capital",
        r"risk factors",
        r"segment information",
        r"^note\s+\d+\s+[-\u2014]",        # Note 14 — Segment Information
        # 8-K / press release exhibit structure
        r"^for immediate release",
        r"condensed consolidated statements? of income",
        r"condensed consolidated balance sheets?",
        r"condensed consolidated statements? of cash flows?",
        r"reconciliation of gaap to non.gaap",
        r"^cfo commentary",
        r"revenue by (?:reportable )?segment",
        r"^outlook\b",
        r"^highlights\b",
        r"^signature\b",
        # Common financial note headers
        r"notes to condensed",
        r"^financial statements",
    ]),
    re.IGNORECASE | re.MULTILINE,
)


class UnsupportedDocumentError(ValueError):
    """Raised when the file type is unsupported for Iteration 2 ingestion."""


class DocumentParseError(ValueError):
    """Raised when a supported file cannot be parsed into usable text."""


def parse_document(document: DocumentInput, settings: Settings) -> ParsedDocument:
    file_path = Path(document.file_path)
    suffix = file_path.suffix.lower()

    if suffix not in settings.supported_upload_extensions:
        raise UnsupportedDocumentError(
            f"Unsupported file type '{suffix}'. Supported types: {settings.supported_upload_extensions}"
        )

    if suffix == ".txt":
        text, metadata = _parse_text_file(file_path)
    else:  # .pdf — already validated above
        text, metadata = _parse_pdf_file(file_path)

    normalized_text = _normalize_text(text)
    if not normalized_text:
        raise DocumentParseError(f"No usable text found in '{file_path.name}'")

    title = document.title or file_path.stem.replace("_", " ").strip() or file_path.name
    doc_id = _build_document_id(document, file_path)

    merged_metadata = {
        **document.metadata,
        **metadata,
        "period_start": document.period_start,
        "period_end": document.period_end,
        "source_type": document.source_type,
    }

    return ParsedDocument(
        doc_id=doc_id,
        company_id=document.company_id,
        source_type=document.source_type,
        file_path=str(file_path),
        text=normalized_text,
        title=title,
        metadata=merged_metadata,
    )


def _parse_text_file(file_path: Path) -> tuple[str, dict[str, int | str]]:
    try:
        docs = TextLoader(str(file_path), encoding="utf-8").load()
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"Could not decode text file '{file_path.name}' as UTF-8") from exc
    except Exception as exc:
        raise DocumentParseError(f"Could not parse text file '{file_path.name}'") from exc

    text = _join_documents(docs, include_empty_page_markers=False)

    return text, {"file_type": "text", "file_name": file_path.name}


def _parse_pdf_file(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Section-aware PDF parser using pdfplumber.

    When pdfplumber is available:
    - Extracts text + tables per page via pdfplumber
    - Groups pages into logical sections using _SECTION_BOUNDARY_RE
    - Stores structured `pages` list in metadata for the section-aware chunker
    - Tables are converted to markdown with header separator so they are never
      split mid-row (Zhu et al. hierarchical table alignment)

    Falls back to PyPDFLoader if pdfplumber is unavailable.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return _parse_pdf_fallback(file_path)

    try:
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            current_section = "Cover"

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""

                # Detect section change — only scan the top portion of the page
                # to avoid boilerplate disclosure paragraphs hijacking the section
                # title for the rest of the document.
                boundary_match = _SECTION_BOUNDARY_RE.search(page_text[:400])
                if boundary_match:
                    current_section = page_text[
                        boundary_match.start(): boundary_match.start() + 60
                    ].strip().split("\n")[0]

                # Convert tables to intact markdown (header bonded to data rows)
                tables_md: list[str] = []
                for raw_table in page.extract_tables():
                    md = _table_to_markdown(raw_table)
                    if md:
                        tables_md.append(md)

                page_text_stripped = page_text.strip()
                if page_text_stripped or tables_md:
                    pages.append({
                        "page": page_num,
                        "text": page_text_stripped,
                        "tables_md": tables_md,
                        "section_title": current_section,
                    })
                    full_text_parts.append(page_text_stripped)

        full_text = "\n\n".join(full_text_parts)
        metadata: dict[str, Any] = {
            "file_type": "pdf",
            "file_name": file_path.name,
            "page_count": page_count,
            "pages": pages,  # structured page data for section-aware chunker
        }
        return full_text, metadata

    except Exception as exc:
        raise DocumentParseError(f"Could not parse PDF '{file_path.name}': {exc}") from exc


def _parse_pdf_fallback(file_path: Path) -> tuple[str, dict[str, Any]]:
    """PyPDFLoader fallback when pdfplumber is not installed."""
    from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
    try:
        docs = PyPDFLoader(str(file_path)).load()
    except Exception as exc:
        raise DocumentParseError(f"Could not open PDF '{file_path.name}'") from exc
    text = _join_documents(docs, include_empty_page_markers=True)
    return text, {
        "file_type": "pdf",
        "file_name": file_path.name,
        "page_count": len(docs),
    }


def _table_to_markdown(table: list[list[Any]]) -> str | None:
    """Convert pdfplumber raw table to markdown with header separator.

    Keeps header bonded to data rows — prevents error cascades from
    split tables (Zhu et al. hierarchical table alignment principle).
    """
    if not table or len(table) < 2:
        return None
    rows: list[str] = []
    for i, row in enumerate(table):
        cells = [str(c or "").strip().replace("\n", " ") for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:  # header separator after first row
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(rows)


def _join_documents(docs: list[Document], include_empty_page_markers: bool) -> str:
    if not docs:
        return ""

    page_text: list[str] = []
    for index, doc in enumerate(docs):
        content = doc.page_content or ""
        if content.strip():
            page_text.append(content)
        elif include_empty_page_markers:
            page_number = doc.metadata.get("page", index)
            page_text.append(f"[page {int(page_number) + 1} contained no extractable text]")

    return "\n\n".join(page_text)


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def _build_document_id(document: DocumentInput, file_path: Path) -> str:
    stable_key = f"{document.company_id}:{document.source_type}:{file_path.resolve()}"
    return f"doc_{uuid5(NAMESPACE_URL, stable_key).hex}"

