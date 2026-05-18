from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict


# Lazy import to avoid circular dependency
def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (loaded on first use)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


@dataclass(slots=True)
class DocumentInput:
    file_path: str
    company_id: str | None = None
    source_type: str | None = None
    title: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    company_id: str
    source_type: str
    file_path: str
    text: str
    title: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count == 0 and self.text:
            self.token_count = _count_tokens(self.text)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    company_id: str
    text: str
    position: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    section_id: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    chunk_type: str = "prose"
    parent_chunk_id: str | None = None
    embedding: list[float] | None = None


@dataclass(slots=True)
class ExtractionRecord:
    entity_id: str
    entity_type: str
    canonical_name: str
    raw_text: str
    confidence: float
    source_chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LangGraph state types — live here to break the circular import between
# nodes.py and orchestration.py (both import models, neither imports the other)
# ---------------------------------------------------------------------------

class _DocTask(TypedDict):
    """Per-document fan-out payload sent via Send()."""

    upload: dict[str, str]


class IngestionGraphState(TypedDict, total=False):
    """Shared state for the ingestion LangGraph pipeline.

    Keys with ``Annotated[list, add]`` use the ``add`` reducer so that
    parallel ``process_single_document`` nodes can fan-in their partial
    results without overwriting each other.
    """

    uploads: list[dict[str, str]]
    processed_documents: Annotated[list[dict[str, object]], add]
    chunk_records: Annotated[list[dict[str, object]], add]
    extraction_records: Annotated[list[dict[str, object]], add]
    bm25_index_paths: Annotated[list[str], add]
    warnings: Annotated[list[dict[str, str]], add]
    errors: Annotated[list[dict[str, str]], add]
    status: str
