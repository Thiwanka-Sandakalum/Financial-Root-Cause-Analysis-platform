import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from llama_cloud import LlamaCloud
from llama_cloud.types.parsing_get_response import (
    MarkdownPageMarkdownResultPage,
    ItemsPageStructuredResultPage,
)
from pydantic import BaseModel


_client = LlamaCloud()
T = TypeVar("T")


class SectionSpan(BaseModel):
    section_type: Literal["MD&A", "FINANCIALS", "RISK", "EARNINGS", "NOTES", "GENERAL"]
    title: str
    page_start: int
    page_end: int


class DocumentOutline(BaseModel):
    sections: List[SectionSpan]


_CLASSIFY_PROMPT = """\
You are a financial document analyst. You will receive a compact outline of a \
financial document (page numbers, headings, and a brief text snippet per page). \
Group consecutive pages into named sections.

Use ONLY these section_type values:
- MD&A       : Management's Discussion & Analysis
- FINANCIALS : Consolidated financial statements (income statement, balance sheet, cash flows)
- RISK       : Risk factors
- EARNINGS   : Results of operations, quarterly/annual performance summaries
- NOTES      : Notes to financial statements
- GENERAL    : Cover pages, signatures, exhibits, legal boilerplate, everything else

Rules:
- Every page must belong to exactly one section.
- Sections must be contiguous — no gaps, no overlaps.
- page_start and page_end are inclusive page numbers.
- title is the most prominent heading found in that section (verbatim from the outline).
- Return sections in ascending page order.

Document outline:
{outline}"""


def _classify_pages(
    page_outlines: List[dict],
    llm: ChatGoogleGenerativeAI,
) -> List[SectionSpan]:
    """Single LLM call that groups pages into SectionSpan objects."""
    outline_text = "\n".join(
        f"Page {p['page']}: headings={p['headings']} | {p['snippet']}"
        for p in page_outlines
    )
    structured_llm = llm.with_structured_output(DocumentOutline)
    result = structured_llm.invoke(_CLASSIFY_PROMPT.format(outline=outline_text))
    assert isinstance(result, DocumentOutline)
    return result.sections


def _retry_call(func: Callable[[], T], context: str, max_attempts: int = 3) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            print(f"  [parser] Retry {attempt}/{max_attempts - 1} for {context}: {exc}")
            time.sleep(delay)
    assert last_exc is not None
    raise RuntimeError(f"Failed {context} after {max_attempts} attempts: {last_exc}")


def _spans_cover_all_pages(spans: List[SectionSpan], page_numbers: List[int]) -> bool:
    expected = sorted(page_numbers)
    if not expected or not spans:
        return False

    covered: List[int] = []
    for span in spans:
        if span.page_start > span.page_end:
            return False
        covered.extend(range(span.page_start, span.page_end + 1))

    return covered == expected


def _fallback_single_section(
    page_data: List[dict], title: str = "Document"
) -> List["ParsedSection"]:
    if not page_data:
        return []
    sorted_pages = sorted(page_data, key=lambda p: p["page_num"])
    return [
        ParsedSection(
            section_type="GENERAL",
            title=title,
            content="\n".join(pd["markdown"] for pd in sorted_pages),
            tables=[tbl for pd in sorted_pages for tbl in pd["tables"]],
            headings=[h for pd in sorted_pages for h in pd["headings"]],
            page_start=sorted_pages[0]["page_num"],
            page_end=sorted_pages[-1]["page_num"],
        )
    ]


@dataclass
class ParsedSection:
    section_type: str
    title: str
    content: str
    tables: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1


def extract_page_headings(items) -> List[str]:
    headings = []
    for item in items:
        if getattr(item, "type", None) == "heading":
            text = getattr(item, "value", "").strip()
            if len(text) > 3:
                headings.append(text)
    return headings


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _save_markdown(pdf_path: str, markdown: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(_DATA_DIR, f"{stem}.md")
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    return out_path


def parse_financial_pdf(
    pdf_path: str,
    llm: ChatGoogleGenerativeAI,
) -> List[ParsedSection]:
    """
    Two-phase structure-aware parser for financial PDFs.

    Phase 1 — Extract: LlamaParse produces per-page markdown, headings, and tables.
    Phase 2 — Classify: A single LLM call (with_structured_output) groups pages into
                        named financial sections, replacing all regex detection logic.
    Phase 3 — Assemble: Page slices are joined per SectionSpan into ParsedSection objects.
    """
    # Upload and parse synchronously (SDK handles job polling internally)
    file_obj = _retry_call(
        lambda: _client.files.create(file=pdf_path, purpose="parse"),
        context=f"LlamaCloud file upload for {pdf_path}",
    )
    result = _retry_call(
        lambda: _client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown", "items"],
        ),
        context=f"LlamaCloud parse for {pdf_path}",
    )

    if result.markdown is None or result.items is None:
        raise RuntimeError(
            f"LlamaParse returned no content for {pdf_path}. Job status: {result.job.status}"
        )

    _save_markdown(
        pdf_path,
        result.markdown_full
        or "\n\n".join(
            p.markdown
            for p in result.markdown.pages
            if isinstance(p, MarkdownPageMarkdownResultPage)
        ),
    )

    markdown_pages = [
        p
        for p in result.markdown.pages
        if isinstance(p, MarkdownPageMarkdownResultPage)
    ]
    items_pages = [
        p for p in result.items.pages if isinstance(p, ItemsPageStructuredResultPage)
    ]

    items_by_page = {p.page_number: p.items for p in items_pages}

    # ── Phase 1: Extract per-page data ──────────────────────────────────────
    page_data: List[dict] = []
    for md_page in markdown_pages:
        page_num = md_page.page_number
        page_md = md_page.markdown or ""
        page_items = items_by_page.get(page_num, [])
        headings = extract_page_headings(page_items)
        tables = [
            item.md
            for item in page_items
            if getattr(item, "type", None) == "table" and getattr(item, "md", None)
        ]
        page_data.append(
            {
                "page_num": page_num,
                "headings": headings,
                "tables": tables,
                "markdown": page_md,
            }
        )

    # ── Phase 2: Classify pages via a single LLM call ───────────────────────
    page_outlines = [
        {
            "page": pd["page_num"],
            "headings": pd["headings"],
            "snippet": pd["markdown"][:200].replace("\n", " "),
        }
        for pd in page_data
    ]
    spans: List[SectionSpan]
    try:
        spans = _retry_call(
            lambda: _classify_pages(page_outlines, llm),
            context=f"section classification for {pdf_path}",
        )
    except Exception as exc:
        print(f"  [parser] Classification failed, using fallback sectioning: {exc}")
        return _fallback_single_section(page_data)

    if not _spans_cover_all_pages(spans, [pd["page_num"] for pd in page_data]):
        print(
            "  [parser] Invalid section spans returned by classifier, using fallback sectioning"
        )
        return _fallback_single_section(page_data)

    # ── Phase 3: Assemble ParsedSection objects ──────────────────────────────
    by_page = {pd["page_num"]: pd for pd in page_data}

    sections: List[ParsedSection] = []
    for span in spans:
        span_pages = [
            by_page[n]
            for n in range(span.page_start, span.page_end + 1)
            if n in by_page
        ]
        sections.append(
            ParsedSection(
                section_type=span.section_type,
                title=span.title,
                content="\n".join(pd["markdown"] for pd in span_pages),
                tables=[tbl for pd in span_pages for tbl in pd["tables"]],
                headings=[h for pd in span_pages for h in pd["headings"]],
                page_start=span.page_start,
                page_end=span.page_end,
            )
        )

    return sections


def parse_text_file(txt_path: str) -> List[ParsedSection]:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    return [
        ParsedSection(
            section_type="GENERAL",
            title="Article",
            content=content,
            tables=[],
            headings=[],
            page_start=1,
            page_end=1,
        )
    ]
