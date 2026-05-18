"""Evidence ranking for retrieved chunks before answer synthesis."""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.documents import Document

from src.config import Settings


_CHUNK_TYPE_BOOST: dict[str, dict[str, float]] = {
    "metric_comparison": {"table": 0.15, "prose": 0.0},
    "fact_lookup": {"table": 0.10, "prose": 0.0},
    "root_cause_analysis": {"table": 0.05, "prose": 0.05, "causal": 0.20},
    "temporal_analysis": {"table": 0.08, "prose": 0.02},
}


def rank_evidence(
    query_embedding: list[float] | None,
    retrieved_chunks: list[Document],
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    settings: Settings,
    intent_type: str = "fact_lookup",
) -> list[Document]:
    """Rank evidence chunks with weighted scoring and return top-k.

    Formula:
      final_score = w_relevance * relevance + w_temporal * temporal + w_confidence * confidence
    """
    del query_embedding  # Relevance currently comes from retrieval scores in metadata.

    if not retrieved_chunks:
        return []

    weights = settings.query_ranking_weights
    ranked: list[Document] = []

    for chunk in retrieved_chunks:
        relevance_score = _extract_relevance_score(chunk)
        temporal_score = _compute_temporal_score(chunk.metadata, period_start, period_end)
        confidence_score = _compute_confidence_score(chunk.metadata, company_id)
        type_boost = _compute_chunk_type_boost(chunk.metadata, intent_type)

        final_score = (
            weights["relevance"] * relevance_score
            + weights["temporal"] * temporal_score
            + weights["confidence"] * confidence_score
            + type_boost
        )

        chunk.metadata["relevance_score"] = relevance_score
        chunk.metadata["temporal_score"] = temporal_score
        chunk.metadata["confidence_score"] = confidence_score
        chunk.metadata["chunk_type_boost"] = type_boost
        chunk.metadata["final_score"] = final_score

        ranked.append(chunk)

    # Primary sort: final_score descending.
    # Tie-breaking (secondary/tertiary) keeps substantive chunks ahead of
    # boilerplate that accidentally share the same composite score:
    #   - Chunks with an explicit position (real document order) beat those
    #     without one (expanded/disclaimer nodes).
    #   - Among those still tied, higher raw combined_score wins.
    ranked.sort(
        key=lambda doc: (
            float(doc.metadata.get("final_score", 0.0)),
            # 1 if chunk has a recorded document position, else 0
            1 if doc.metadata.get("position") is not None else 0,
            # raw combined_score as tertiary key
            float(doc.metadata.get("combined_score", doc.metadata.get("vector_score", 0.0))),
        ),
        reverse=True,
    )
    return ranked[: settings.query_evidence_top_k]


def _extract_relevance_score(chunk: Document) -> float:
    metadata = chunk.metadata
    if "combined_score" in metadata:
        return _clamp01(float(metadata["combined_score"]))

    vector_score = float(metadata.get("vector_score", 0.0))
    fulltext_score = float(metadata.get("fulltext_score", 0.0))
    combined = 0.6 * vector_score + 0.4 * fulltext_score
    return _clamp01(combined)


def _compute_temporal_score(
    metadata: dict[str, Any],
    period_start: str | None,
    period_end: str | None,
) -> float:
    if not period_start and not period_end:
        return 1.0

    chunk_start = _parse_date(metadata.get("period_start"))
    chunk_end = _parse_date(metadata.get("period_end"))
    req_start = _parse_date(period_start)
    req_end = _parse_date(period_end)

    if req_start and req_end and chunk_start and chunk_end:
        overlap_start = max(req_start, chunk_start)
        overlap_end = min(req_end, chunk_end)
        if overlap_start <= overlap_end:
            requested_window = max((req_end - req_start).days + 1, 1)
            overlap_days = (overlap_end - overlap_start).days + 1
            return _clamp01(overlap_days / requested_window)

    if req_start and chunk_end and chunk_end >= req_start:
        return 0.7
    if req_end and chunk_start and chunk_start <= req_end:
        return 0.7

    return 0.2


def _compute_confidence_score(metadata: dict[str, Any], company_id: str | None) -> float:
    base = float(metadata.get("confidence", metadata.get("extraction_confidence", 0.5)))
    base = _clamp01(base)

    if company_id and metadata.get("company_id") == company_id:
        base = min(1.0, base + 0.1)

    return base


def _compute_chunk_type_boost(metadata: dict[str, Any], intent_type: str) -> float:
    boosts = _CHUNK_TYPE_BOOST.get(intent_type, {})
    expansion_reason = str(metadata.get("expansion_reason", "")).lower()
    if "causal" in expansion_reason:
        return boosts.get("causal", 0.0)
    chunk_type = str(metadata.get("chunk_type", "prose")).lower()
    return boosts.get(chunk_type, 0.0)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
