from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from src.config import Settings
from src.ingestion.models import ChunkRecord, ParsedDocument
from src.llm.gemini import build_embedding_model, build_fast_model

logger = logging.getLogger(__name__)

# Real tokenizer — compatible with modern LLMs (GPT-3.5, GPT-4, Claude, Gemini)
_ENC = tiktoken.get_encoding("cl100k_base")

# Sentence boundary pattern for sentence-aware chunking
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

MAX_TOKENS = 512
MIN_CHUNK_TOKENS = 40  # below this, merge into previous chunk
_WORDS_PER_TOKEN_RATIO = 0.75


def _count_tokens(text: str) -> int:
    """Count actual tokens using tiktoken (not word count estimate)."""
    return len(_ENC.encode(text))


def _sentence_boundary_split(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split text at sentence boundaries using real token counts.

    Algorithm (Stäbler et al. 2025):
    1. Split text at sentence boundaries (. ! ?)
    2. Group sentences while total tokens < max_tokens
    3. When a sentence is too long, split at clause boundary (comma/semicolon)
    4. If no clause boundaries, fall back to word-based split
    5. Merge orphaned short chunks into previous chunk
    """
    if not text.strip():
        return []

    # Split at sentence boundaries
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    chunks: list[str] = []
    current_tokens = 0
    current_sents: list[str] = []

    for sent in sentences:
        if not sent.strip():
            continue

        sent_tokens = _count_tokens(sent)

        # Single sentence too long — try clause boundary split first
        if sent_tokens > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents = []
                current_tokens = 0

            # Try splitting at semicolon or comma first
            parts = re.split(r"(?<=[;,])\s+", sent)
            if len(parts) > 1:
                # Clause boundaries found
                for part in parts:
                    if part.strip():
                        chunks.append(part.strip())
            else:
                # No clause boundaries — fall back to word-based split
                words = sent.split()
                chunk_words: list[str] = []
                chunk_tokens = 0
                
                for word in words:
                    word_tokens = _count_tokens(word)
                    if chunk_tokens + word_tokens > max_tokens and chunk_words:
                        chunks.append(" ".join(chunk_words))
                        chunk_words = [word]
                        chunk_tokens = word_tokens
                    else:
                        chunk_words.append(word)
                        chunk_tokens += word_tokens
                
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
            continue

        # Add sentence to current chunk if it fits
        if current_tokens + sent_tokens > max_tokens and current_sents:
            chunks.append(" ".join(current_sents))
            current_sents = [sent]
            current_tokens = sent_tokens
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens

    # Flush remaining sentences
    if current_sents:
        chunks.append(" ".join(current_sents))

    # Merge orphaned short chunks into the previous chunk
    merged: list[str] = []
    for chunk in chunks:
        if merged and _count_tokens(chunk) < MIN_CHUNK_TOKENS:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)

    return [c.strip() for c in merged if c.strip()]


class ChunkingError(ValueError):
    """Raised when a parsed document cannot be chunked safely."""


def chunk_document(document: ParsedDocument, settings: Settings) -> list[ChunkRecord]:
    if not document.text.strip():
        raise ChunkingError(f"Document '{document.doc_id}' does not contain any tokenizable text")

    pages: list[dict[str, Any]] | None = document.metadata.get("pages")
    if pages:
        chunks = _chunk_structured(document, pages, settings)
    else:
        chunks = _chunk_flat(document, settings)

    if not chunks:
        raise ChunkingError(f"Document '{document.doc_id}' could not be chunked into valid records")

    return chunks


def _build_section_id(doc_id: str, section_title: str) -> str:
    """Stable, deterministic section id from doc + title (Gijre & Laddha [16])."""
    digest = sha1(f"{doc_id}:{section_title}".encode("utf-8")).hexdigest()
    return f"sec_{digest}"


# ---------------------------------------------------------------------------
# Section-aware path (PDF with structured pages from parser.py)
# ---------------------------------------------------------------------------

def _chunk_structured(
    document: ParsedDocument,
    pages: list[dict[str, Any]],
    settings: Settings,
) -> list[ChunkRecord]:
    """Element-based chunking for PDFs (Yang et al. [21], Stäbler et al. [23]).

    Algorithm:
    1. Group consecutive pages sharing the same section_title into sections.
    2. Within each section, split prose at sentence boundaries (512-token cap).
    3. Each table is emitted as a SEPARATE chunk_type='table' chunk with:
       - its own embedding (clean table signal, not diluted by prose)
       - parent_chunk_id pointing to the first prose chunk of the section
       - section_id linking it to the Section node in the graph
    4. Tiny prose shards below min_chunk_length are merged into the previous
       prose chunk (not into table chunks).
    """
    max_tokens: int = 512
    max_words = int(max_tokens * _WORDS_PER_TOKEN_RATIO)  # ≈ 384 words
    min_length: int = settings.ingestion_min_chunk_length

    position = 0
    chunks: list[ChunkRecord] = []

    # base_meta: document-level fields propagated to every chunk's metadata.
    # Excludes "pages" (PDF-only structural data not needed after chunking).
    # graph_writer reads period_start/period_end from chunk.metadata; all
    # structural fields (section_title, section_id, etc.) live on the dataclass.
    base_meta = {k: v for k, v in document.metadata.items() if k != "pages"}

    # Group pages by section (consecutive pages with the same section_title)
    sections: list[tuple[str, list[dict[str, Any]]]] = []
    for page in pages:
        section_title = page.get("section_title") or "Cover"
        if sections and sections[-1][0] == section_title:
            sections[-1][1].append(page)
        else:
            sections.append((section_title, [page]))

    for section_title, section_pages in sections:
        section_id = _build_section_id(document.doc_id, section_title)
        first_page_num: int | None = None
        section_prose_parts: list[str] = []

        # Collect all tables across the section (preserve page order)
        section_tables: list[tuple[str, int | None]] = []  # (markdown, page_num)

        for page in section_pages:
            pg_num = page.get("page")
            if first_page_num is None:
                first_page_num = pg_num
            section_prose_parts.append(page.get("text") or "")
            for tbl_md in (page.get("tables_md") or []):
                section_tables.append((tbl_md, pg_num))

        section_prose = "\n".join(section_prose_parts).strip()
        if not section_prose and not section_tables:
            continue

        # --- Prose sub-chunks using sentence boundaries + real token counts ---
        sub_texts = _sentence_boundary_split(section_prose, MAX_TOKENS)
        section_first_prose_chunk_id: str | None = None

        for sub_idx, sub_text in enumerate(sub_texts):
            chunk_text = sub_text.strip()
            if not chunk_text:
                continue

            # Merge tiny trailing prose shard into previous prose chunk
            if _count_tokens(chunk_text) < MIN_CHUNK_TOKENS and position > 0 and chunks:
                prev = chunks[-1]
                if prev.chunk_type == "prose":
                    prev.text = f"{prev.text} {chunk_text}".strip()
                    prev.token_count = _count_tokens(prev.text)
                    continue

            token_count = _count_tokens(chunk_text)
            chunk_id = _build_chunk_id(document.doc_id, position, sub_idx, chunk_text)

            if section_first_prose_chunk_id is None:
                section_first_prose_chunk_id = chunk_id

            metadata = {
                **base_meta,
                "title": document.title,
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    company_id=document.company_id,
                    text=chunk_text,
                    position=position,
                    token_count=token_count,
                    section_title=section_title,
                    section_id=section_id,
                    page_number=first_page_num,
                    chunk_type="prose",
                    metadata=metadata,
                )
            )
            position += 1

        # --- Table chunks (separate nodes — Zhu et al. [24], Yang et al. [21]) ---
        # Each table gets its own ChunkRecord with chunk_type='table' so:
        # - embeddings stay pure table signal (not mixed with prose)
        # - queries can filter WHERE c.chunk_type = 'table'
        # - parent_chunk_id links back to the prose context of this section
        for tbl_idx, (tbl_md, tbl_page) in enumerate(section_tables):
            if not tbl_md.strip():
                continue
            token_count = _count_tokens(tbl_md)
            chunk_id = _build_chunk_id(document.doc_id, position, tbl_idx, tbl_md)
            metadata = {
                **base_meta,
                "title": document.title,
                "table_index": tbl_idx,
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    company_id=document.company_id,
                    text=tbl_md,
                    position=position,
                    token_count=token_count,
                    section_title=section_title,
                    section_id=section_id,
                    page_number=tbl_page,
                    chunk_type="table",
                    parent_chunk_id=section_first_prose_chunk_id,
                    metadata=metadata,
                )
            )
            position += 1

    return chunks




def _chunk_flat(document: ParsedDocument, settings: Settings) -> list[ChunkRecord]:
    """Flat path for TXT files using sentence boundaries + real token counts.

    Replaces RecursiveCharacterTextSplitter with sentence-aware splitting
    (Stäbler et al. 2025), achieving higher accuracy on evaluations.
    """
    # Use a main section for all flat text
    section_id = _build_section_id(document.doc_id, "Document")
    section_title = "Document"

    # Split using sentence boundaries + real token counts
    text_chunks = _sentence_boundary_split(document.text, MAX_TOKENS)

    chunks: list[ChunkRecord] = []
    base_metadata = {**document.metadata, "title": document.title}

    for position, text in enumerate(text_chunks):
        chunk_text = text.strip()
        if not chunk_text:
            continue

        # Merge tiny orphaned chunks into previous (already done in _sentence_boundary_split)
        token_count = _count_tokens(chunk_text)

        chunk_id = _build_chunk_id(document.doc_id, position, 0, chunk_text)
        
        # Track token boundaries within this chunk (0 to token_count)
        metadata = {
            **base_metadata,
            "chunk_start_token": 0,
            "chunk_end_token": token_count,
        }
        
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                company_id=document.company_id,
                text=chunk_text,
                position=position,
                token_count=token_count,
                section_id=section_id,
                section_title=section_title,
                page_number=None,
                chunk_type="prose",
                metadata=metadata,
            )
        )

    return chunks


def _build_chunk_id(doc_id: str, position: int, start_index: int, text: str) -> str:
    """Stable, deterministic chunk ID from doc + position + content hash."""
    digest = sha1(f"{doc_id}:{position}:{start_index}:{text}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


# ---------------------------------------------------------------------------
# Embedding (merged from embedder.py)
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[ChunkRecord], settings: Settings) -> list[dict[str, Any]]:
    """Embed chunks in batches and return chunk_id->embedding mapping records."""
    if not chunks:
        return []

    model = build_embedding_model(settings)
    batch_size = settings.ingestion_embedding_batch_size
    records: list[dict[str, Any]] = []

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        texts = [chunk.text for chunk in batch]
        vectors = model.embed_documents(texts)

        for chunk, vector in zip(batch, vectors, strict=True):
            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "embedding": vector,
                }
            )

    return records


# ---------------------------------------------------------------------------
# RAPTOR section summaries (merged from raptor.py)
# ---------------------------------------------------------------------------

# Sarthi et al. (2024): tree-organized retrieval beats flat RAG for long docs.
# Groups ALL chunks (prose AND table) by section_id and generates LLM summaries
# + embeddings so Stage-1 retrieval can match at the section level.

_MAX_CHUNKS_PER_SECTION = 5
_MAX_CHARS_PER_CHUNK = 600

_PROSE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a financial analyst assistant. "
                "Summarize the following financial document section in 2-3 sentences. "
                "Focus on: key metrics mentioned, main themes, and what a financial "
                "analyst would find important. Be specific with numbers if present. "
                "Return plain text only — no bullet points, no headers."
            ),
        ),
        ("human", "{section_text}"),
    ]
)

_TABLE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a financial analyst. The following is a financial table "
                "from a financial document. Summarize what this table shows in 2-3 sentences. "
                "Be specific: include the key metric names, the most significant values, "
                "and any notable year-over-year or sequential changes visible in the data. "
                "State the findings directly — do NOT say 'this table shows'. "
                "Return plain text only — no bullet points."
            ),
        ),
        ("human", "{table_text}"),
    ]
)


def build_section_summaries(
    chunks: list[ChunkRecord],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Generate LLM summaries + embeddings for every unique section in *chunks*.

    Includes BOTH prose AND table chunks. Returns records that graph_writer
    stores as ``summary`` / ``summary_embedding`` on the Section node.
    """
    by_section: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        if chunk.section_id:
            by_section[chunk.section_id].append(chunk)

    if not by_section:
        return []

    llm = build_fast_model(settings)
    prose_chain = _PROSE_SUMMARY_PROMPT | llm
    table_chain = _TABLE_SUMMARY_PROMPT | llm

    summaries: list[tuple[str, str]] = []

    for section_id, section_chunks in by_section.items():
        # --- Prose summary ---
        prose_chunks = [c for c in section_chunks if c.chunk_type == "prose"]
        if prose_chunks:
            ordered = sorted(prose_chunks, key=lambda c: c.position)
            combined = "\n\n".join(
                c.text[:_MAX_CHARS_PER_CHUNK]
                for c in ordered[:_MAX_CHUNKS_PER_SECTION]
            ).strip()
            if combined:
                try:
                    response = prose_chain.invoke({"section_text": combined})
                    prose_summary: str = (
                        response.content.strip()
                        if hasattr(response, "content")
                        else str(response).strip()
                    )
                    if prose_summary:
                        summaries.append((section_id, prose_summary))
                except Exception as exc:
                    logger.warning("RAPTOR prose summary failed for %s: %s", section_id, exc)

        # --- Table summaries ---
        table_chunks = [c for c in section_chunks if c.chunk_type == "table"]
        for table_idx, table_chunk in enumerate(table_chunks):
            if _count_tokens(table_chunk.text) < 30:
                continue
            try:
                response = table_chain.invoke({"table_text": table_chunk.text[:1200]})
                table_summary: str = (
                    response.content.strip()
                    if hasattr(response, "content")
                    else str(response).strip()
                )
                if table_summary:
                    summaries.append((f"{section_id}_table_{table_idx}", table_summary))
            except Exception as exc:
                logger.warning(
                    "RAPTOR table summary failed for %s table %d: %s",
                    section_id, table_idx, exc,
                )

    if not summaries:
        return []

    embedding_model = build_embedding_model(settings)
    summary_texts = [s for _, s in summaries]
    try:
        vectors: list[list[float] | None] = embedding_model.embed_documents(summary_texts)
    except Exception as exc:
        logger.warning("RAPTOR embedding batch failed: %s — stored without embeddings", exc)
        vectors = [None] * len(summaries)

    return [
        {
            "section_id": sid,
            "summary": summary,
            "summary_embedding": vectors[i],
        }
        for i, (sid, summary) in enumerate(summaries)
    ]


# ---------------------------------------------------------------------------
# BM25 full-text indexing (merged from bm25_indexer.py)
# ---------------------------------------------------------------------------

# Lin & Jang (2025): BM25 beats vector search for exact financial figures (+33% precision).
# Builds and persists per-document BM25 indexes during ingestion.
# Indexes are loaded at query time for hybrid retrieval (vector + BM25).

BM25_INDEX_DIR = Path("./data/bm25_indexes")
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _financial_tokenize(text: str) -> list[str]:
    """Financial-aware tokenizer preserving exact numbers and tickers.

    Examples:
        "$35,082M" → "35082M"   (currency normalisation, preserve magnitude)
        "74.6%"   → "74_6PCT"  (percentage signal)
        "Q3FY25"  → "Q3FY25"   (period token preserved)
    """
    text = text.upper()
    text = re.sub(
        r"\$\s*([\d,]+\.?\d*)\s*(M|B|K)?",
        lambda m: m.group(1).replace(",", "") + (m.group(2) or ""),
        text,
    )
    text = re.sub(r"([\d]+\.?\d*)\s*%", r"\1PCT", text)
    text = re.sub(r"Q([1-4])\s*(?:FY|FISCAL\s*YEAR)?\s*(20\d{2})", r"Q\1FY\2", text)
    return [t for t in re.split(r"\s+", text) if t and len(t) > 1]


class BM25ChunkIndex:
    """Per-document BM25 index — built at ingestion, loaded at query time."""

    def __init__(self, company_id: str, period_key: str) -> None:
        self.company_id = company_id
        self.period_key = period_key
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def index_path(self) -> Path:
        return BM25_INDEX_DIR / f"{self.company_id}_{self.period_key}.pkl"

    def build(self, chunks: list[ChunkRecord]) -> None:
        """Build the BM25 index from chunks."""
        self._chunks = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "section_title": c.section_title,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type,
            }
            for c in chunks
        ]
        tokenized = [_financial_tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Retrieve top-k chunks by BM25 score."""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_financial_tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return [
            {**chunk, "bm25_score": float(score)}
            for chunk, score in ranked[:top_k]
            if score > 0.0
        ]

    def save(self) -> Path:
        """Persist the index to disk."""
        path = self.index_path
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        return path

    @classmethod
    def load(cls, company_id: str, period_key: str) -> "BM25ChunkIndex":
        """Load a persisted index from disk."""
        instance = cls(company_id, period_key)
        path = instance.index_path
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            instance._chunks = data["chunks"]
            instance._bm25 = data["bm25"]
        return instance

