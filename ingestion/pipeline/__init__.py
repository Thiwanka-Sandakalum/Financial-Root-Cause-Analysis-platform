"""
ingestion/pipeline — Core document-ingestion logic.

Public surface:
  parse_financial_pdf, parse_text_file  — produce ParsedSection lists
  chunk_document                         — split sections into Chunk objects
  extract_entities_and_relations        — entity / relation extraction
  ingest_document                        — end-to-end orchestration
"""

from ingestion.pipeline.chunker import Chunk, chunk_document, chunk_section
from ingestion.pipeline.entity_extractor import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    extract_entities_and_relations,
)
from ingestion.pipeline.graph_writer import ingest_document
from ingestion.pipeline.parser import (
    DocumentOutline,
    ParsedSection,
    SectionSpan,
    parse_financial_pdf,
    parse_text_file,
)

__all__ = [
    # parser
    "parse_financial_pdf",
    "parse_text_file",
    "ParsedSection",
    "SectionSpan",
    "DocumentOutline",
    # chunker
    "chunk_section",
    "chunk_document",
    "Chunk",
    # entity extractor
    "extract_entities_and_relations",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelation",
    # orchestration
    "ingest_document",
]
