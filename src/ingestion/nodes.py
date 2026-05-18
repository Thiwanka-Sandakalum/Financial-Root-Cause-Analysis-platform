"""Ingestion graph node functions.

Follows the LangGraph canonical pattern: node functions live in their own
module so they can be tested, imported, and reasoned about independently of
graph construction (``orchestration.py``).

Nodes that require external dependencies (settings, Neo4j client) are wrapped
in factory functions that bind those dependencies via closure.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from src.config import Settings
from src.db.neo4j_client import Neo4jClient
from src.llm.gemini import build_embedding_model
from src.ingestion.chunker import (
    BM25ChunkIndex,
    build_section_summaries,
    chunk_document,
    embed_chunks,
)
from src.ingestion.classifier import (
    DetectedMetadata,
    ExistingCompany,
    classify_metadata,
    list_existing_companies,
)
from src.ingestion.extractor import (
    extract_causal_links_from_chunks,
    extract_entities_from_chunks,
    extract_event_causal_chains_from_chunks,
    extract_events_from_chunks,
    extract_metrics_from_chunks,
    to_state_dicts,
)
from src.ingestion.graph_writer import (
    build_similar_chunk_edges,
    resolve_extracted_node_duplicates,
    write_document_graph,
)
from src.ingestion.models import DocumentInput
from src.ingestion.parser import parse_document
from src.ingestion.models import IngestionGraphState, _DocTask


# ---------------------------------------------------------------------------
# Stateless nodes (no external dependencies)
# ---------------------------------------------------------------------------


def validate_uploads(state: IngestionGraphState) -> dict[str, Any]:
    """Validate that the uploads list is non-empty."""
    if not state.get("uploads"):
        return {
            "status": "failed",
            "errors": [{"code": "NO_UPLOADS", "message": "No uploads provided"}],
        }
    return {"status": "validated"}


def fan_out_documents(state: IngestionGraphState) -> list[Send]:
    """Conditional-edge function: fan out one Send per upload."""
    return [
        Send("process_single_document", {"upload": upload})
        for upload in state.get("uploads", [])
    ]


# ---------------------------------------------------------------------------
# Stateful node factories (bind settings + client via closure)
# ---------------------------------------------------------------------------


def make_process_single_document(settings: Settings, neo4j_client: Neo4jClient):
    """Return a node function that processes one document end-to-end."""

    def process_single_document(task_state: _DocTask) -> dict[str, Any]:
        upload = task_state["upload"]

        document_input = DocumentInput(
            file_path=upload["file_path"],
            company_id=upload.get("company_id"),
            source_type=upload.get("source_type"),
            title=upload.get("title"),
            period_start=upload.get("period_start"),
            period_end=upload.get("period_end"),
            metadata={"upload_id": upload.get("upload_id")},
        )

        # Fix 8: Stage-level error isolation
        # Fatal stages: parse (must have text), chunk (must split text)
        parsed = parse_document(document_input, settings)
        chunks = chunk_document(parsed, settings)

        # Non-fatal stages: collect warnings but continue
        warnings: list[dict[str, str]] = []

        # --- Classification stage (non-fatal) ---
        detected_metadata: DetectedMetadata | None = None
        existing_companies: list[ExistingCompany] = []
        try:
            existing_companies = list_existing_companies(neo4j_client.driver, settings)
        except Exception as exc:  # pragma: no cover - db lookup failure
            warnings.append(
                {"code": "COMPANY_LOOKUP_FAILED", "message": str(exc)}
            )

        try:
            detected_metadata = classify_metadata(
                parsed.text,
                settings,
                parsed,
                existing_companies=existing_companies,
            )
            if detected_metadata.company_id:
                parsed.company_id = detected_metadata.company_id
            if detected_metadata.document_type:
                parsed.source_type = detected_metadata.document_type

            # Wire detected fiscal period — user-supplied dates take precedence
            if detected_metadata.period_label:
                if not parsed.metadata.get("period_start"):
                    parsed.metadata["period_start"] = detected_metadata.period_label
                if not parsed.metadata.get("period_end"):
                    parsed.metadata["period_end"] = detected_metadata.period_label
        except Exception as exc:  # pragma: no cover - classification failure
            warnings.append(
                {"code": "CLASSIFICATION_FAILED", "message": str(exc)}
            )

        # Ensure write path has a usable company id and source type.
        if not parsed.company_id:
            parsed.company_id = upload.get("company_id") or "unknown"
        if not parsed.source_type:
            parsed.source_type = upload.get("source_type") or "uploaded_document"

        company_metadata_embedding: list[float] | None = None
        company_metadata_text = ""
        try:
            embedding_model = build_embedding_model(settings)
            company_metadata_text = (
                f"company_id: {parsed.company_id}\n"
                f"company_name: {(detected_metadata.company_name if detected_metadata else parsed.company_id)}\n"
                f"document_type: {parsed.source_type}"
            )
            company_metadata_embedding = embedding_model.embed_query(company_metadata_text)
        except Exception as exc:  # pragma: no cover - embedding failure
            warnings.append(
                {"code": "COMPANY_EMBEDDING_FAILED", "message": str(exc)}
            )

        # --- Embedding stage (non-fatal) ---
        embedding_records = []
        try:
            embedding_records = embed_chunks(chunks, settings)
        except Exception as exc:  # pragma: no cover - embedding failure
            warnings.append(
                {"code": "EMBEDDING_FAILED", "message": str(exc)}
            )

        # --- BM25 indexing stage (non-fatal) ---
        bm25_index_path = None
        try:
            # Build per-document BM25 index for hybrid retrieval
            company_id = parsed.company_id or "unknown"
            period_key = parsed.metadata.get("period_end", "unknown")
            bm25_index = BM25ChunkIndex(company_id=company_id, period_key=period_key)
            bm25_index.build(chunks)
            bm25_index.save()
            bm25_index_path = str(bm25_index.index_path)
        except Exception as exc:  # pragma: no cover - BM25 indexing failure
            warnings.append(
                {"code": "BM25_INDEXING_FAILED", "message": str(exc)}
            )

        # --- RAPTOR section summaries (non-fatal) ---
        section_summary_records = []
        try:
            section_summary_records = build_section_summaries(chunks, settings)
        except Exception as exc:  # pragma: no cover - RAPTOR failure
            warnings.append(
                {"code": "SECTION_SUMMARY_FAILED", "message": str(exc)}
            )

        # --- Entity/Event/Metric extraction (non-fatal) ---
        entity_records = []
        event_records = []
        metric_records = []
        try:
            entity_records = extract_entities_from_chunks(chunks, settings, parsed.source_type)
        except Exception as exc:  # pragma: no cover - entity extraction failure
            warnings.append(
                {"code": "ENTITY_EXTRACTION_FAILED", "message": str(exc)}
            )
        try:
            event_records = extract_events_from_chunks(chunks, settings, parsed.source_type)
        except Exception as exc:  # pragma: no cover - event extraction failure
            warnings.append(
                {"code": "EVENT_EXTRACTION_FAILED", "message": str(exc)}
            )
        try:
            metric_records = extract_metrics_from_chunks(chunks, settings, parsed.source_type)
        except Exception as exc:  # pragma: no cover - metric extraction failure
            warnings.append(
                {"code": "METRIC_EXTRACTION_FAILED", "message": str(exc)}
            )
        
        extraction_records = entity_records + event_records + metric_records

        # --- Causal link extraction (non-fatal) ---
        causal_links = []
        event_causal_links = []
        try:
            causal_links = extract_causal_links_from_chunks(
                chunks, event_records, metric_records, settings, parsed.source_type
            )
        except Exception as exc:  # pragma: no cover - causal link extraction failure
            warnings.append(
                {"code": "CAUSAL_LINK_EXTRACTION_FAILED", "message": str(exc)}
            )
        try:
            event_causal_links = extract_event_causal_chains_from_chunks(
                chunks, event_records, settings, parsed.source_type
            )
        except Exception as exc:  # pragma: no cover - event causal chain extraction failure
            warnings.append(
                {"code": "EVENT_CAUSAL_CHAIN_FAILED", "message": str(exc)}
            )

        # --- Neo4j write (non-fatal, attempt to persist) ---
        canonical_company_id = None
        try:
            canonical_company_id = write_document_graph(
                neo4j_client.driver,
                settings,
                parsed,
                chunks,
                embedding_records,
                extraction_records,
                detected_metadata=detected_metadata,
                causal_links=causal_links,
                event_causal_links=event_causal_links,
                section_summary_records=section_summary_records,
                company_metadata_embedding=company_metadata_embedding,
                company_metadata_text=company_metadata_text,
            )
        except Exception as exc:  # pragma: no cover - write failure
            warnings.append(
                {"code": "GRAPH_WRITE_FAILED", "message": str(exc)}
            )

        # --- Entity resolution (non-fatal enrichment) ---
        try:
            resolve_extracted_node_duplicates(
                neo4j_client.driver,
                settings,
                parsed.doc_id,
            )
        except Exception as exc:  # pragma: no cover - optional enrichment path
            warnings.append(
                {"code": "ENTITY_RESOLUTION_FAILED", "message": str(exc)}
            )

        return {
            "processed_documents": [
                {
                    "doc_id": parsed.doc_id,
                    "company_id": canonical_company_id or "",
                    "chunk_count": len(chunks),
                    "entity_count": len(entity_records),
                    "event_count": len(event_records),
                    "metric_count": len(metric_records),
                    "extraction_count": len(extraction_records),
                }
            ],
            "chunk_records": [
                {"doc_id": parsed.doc_id, "chunk_count": len(chunks)}
            ],
            "extraction_records": [
                {
                    "doc_id": parsed.doc_id,
                    "records": to_state_dicts(extraction_records),
                }
            ],
            "bm25_index_paths": [bm25_index_path] if bm25_index_path else [],
            "warnings": warnings,
        }

    return process_single_document


def make_finalize(settings: Settings, neo4j_client: Neo4jClient):
    """Return a node function that runs post-ingestion similarity edge building."""

    def finalize(state: IngestionGraphState) -> dict[str, Any]:
        try:
            build_similar_chunk_edges(neo4j_client.driver, settings)
        except Exception as exc:  # pragma: no cover - optional post-processing path
            return {
                "status": "completed_with_warnings",
                "warnings": [{"code": "SIMILARITY_BUILD_FAILED", "message": str(exc)}],
            }
        return {"status": "completed"}

    return finalize
