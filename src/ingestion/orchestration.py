"""Ingestion pipeline graph construction (wires the LangGraph StateGraph).

State types live in ``models.py`` to avoid circular imports between this module
and ``nodes.py``.  Node functions live in ``nodes.py``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.db.neo4j_client import Neo4jClient
from src.ingestion.models import IngestionGraphState
from src.ingestion.nodes import (
    fan_out_documents,
    make_finalize,
    make_process_single_document,
    validate_uploads,
)


def build_ingestion_graph(settings: Settings, neo4j_client: Neo4jClient):
    """Compile the ingestion StateGraph with all nodes and edges wired."""
    builder = StateGraph(IngestionGraphState)

    builder.add_node("validate_uploads", validate_uploads)
    builder.add_node(
        "process_single_document",
        make_process_single_document(settings, neo4j_client),
    )
    builder.add_node("finalize", make_finalize(settings, neo4j_client))

    builder.add_edge(START, "validate_uploads")
    builder.add_conditional_edges(
        "validate_uploads",
        fan_out_documents,
        ["process_single_document"],
    )
    builder.add_edge("process_single_document", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def run_ingestion(
    uploads: list[dict[str, str]],
    settings: Settings,
    neo4j_client: Neo4jClient,
) -> dict[str, object]:
    graph = build_ingestion_graph(settings, neo4j_client)
    result = graph.invoke({"uploads": uploads})

    processed_documents = result.get("processed_documents", [])
    chunk_records = result.get("chunk_records", [])
    extraction_records = result.get("extraction_records", [])

    chunk_count = sum(int(item.get("chunk_count", 0)) for item in chunk_records)
    extraction_count = sum(
        len(item.get("records", []))
        for item in extraction_records
        if isinstance(item.get("records", []), list)
    )

    entity_count = sum(
        int(item.get("entity_count", 0))
        for item in processed_documents
        if isinstance(item, dict)
    )
    event_count = sum(
        int(item.get("event_count", 0))
        for item in processed_documents
        if isinstance(item, dict)
    )
    metric_count = sum(
        int(item.get("metric_count", 0))
        for item in processed_documents
        if isinstance(item, dict)
    )

    return {
        "status": result.get("status", "unknown"),
        "processed_documents": len(processed_documents),
        "chunk_count": chunk_count,
        "extraction_count": extraction_count,
        "entity_count": entity_count,
        "event_count": event_count,
        "metric_count": metric_count,
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }
