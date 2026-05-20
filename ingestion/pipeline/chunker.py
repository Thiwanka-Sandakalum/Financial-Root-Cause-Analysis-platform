from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pipeline.parser import ParsedSection


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    text: str
    section_type: str
    section_id: str  # parent Section node id
    page: int
    sequence: int  # position within its document (global)
    source_doc_id: str


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk_section(
    section: ParsedSection,
    doc_id: str,
    section_id: str,
    chunk_size: int = 2048,
    overlap: int = 256,
    sequence_offset: int = 0,
) -> List[Chunk]:
    """
    Sentence-window chunking with a 512-character window.

    Research rationale (Stäbler et al. 2025):
    A sentence-splitting strategy with a 512-token window achieved the
    highest Intersection-over-Union (IoU) score across all chunking
    strategies tested on domain-specific financial RAG.

    Tables are intentionally excluded from this split (Zhu et al. 2021,
    MultiHiertt). Blind fixed-size splitting severs table rows from their
    headers, triggering irreversible error cascades. Tables are instead
    stored as first-class Table nodes in Neo4j with their own embeddings,
    linked to this section via (Section)-[:HAS_TABLE]->(Table).
    """
    # Narrative text only — tables are written as separate graph nodes
    combined = section.content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # Prefer splitting at paragraph → sentence → word boundaries
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(combined)

    chunks: List[Chunk] = []
    for i, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                id=f"{doc_id}_{section.section_type}_{sequence_offset + i}",
                text=text,
                section_type=section.section_type,
                section_id=section_id,
                page=section.page_start,
                sequence=sequence_offset + i,
                source_doc_id=doc_id,
            )
        )

    return chunks


def chunk_document(
    sections: List[ParsedSection],
    doc_id: str,
    chunk_size: int = 2048,
    overlap: int = 256,
) -> List[Chunk]:
    """
    Chunk all sections of a document, maintaining a global sequence counter
    so NEXT_CHUNK relationships form a continuous chain across section boundaries.
    """
    all_chunks: List[Chunk] = []
    offset = 0
    for idx, section in enumerate(sections):
        section_id = f"{doc_id}_sec_{idx}"
        section_chunks = chunk_section(
            section, doc_id, section_id, chunk_size, overlap, sequence_offset=offset
        )
        all_chunks.extend(section_chunks)
        offset += len(section_chunks)
    return all_chunks
