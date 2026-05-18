"""Coverage gate: detect missing data and guide users to upload required documents."""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from src.config import Settings


COVERAGE_THRESHOLD_STANDARD = 0.80
COVERAGE_THRESHOLD_RCA = 0.85


@dataclass
class CoverageReport:
    """Report on data availability for answering a query."""

    is_complete: bool
    missing_sources: list[str]
    missing_message: str
    coverage_score: float = 0.0  # 0.0–1.0 aggregate across categories


# ---------------------------------------------------------------------------
# Source-type category buckets used for per-category coverage scoring
# ---------------------------------------------------------------------------

_EARNINGS_SOURCE_TYPES = [
    "annual_report", "quarterly_report", "earnings_release", "earnings_report",
    "financial_report", "regulatory_filing", "sec_filing", "10k", "10q", "8k",
    "research_report", "analyst_report",
]
_NEWS_SOURCE_TYPES = ["news_article", "news"]
_COMMENTARY_SOURCE_TYPES = [
    "earnings_release", "earnings_report", "investor_presentation",
    "press_release", "company_announcement", "investor_update",
]

_CATEGORY_LABELS: dict[str, str] = {
    "earnings_reports": "earnings reports / financial filings",
    "news_coverage": "news articles",
    "management_commentary": "management commentary / earnings calls",
    "causal_event_data": "causal event data (earnings calls or analyst commentary)",
}


def check_coverage(
    driver: Driver,
    settings: Settings,
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    intent_type: str = "fact_lookup",
) -> CoverageReport:
    """Check if Neo4j has sufficient data to answer the query.

    Validates baseline doc + chunk existence, then runs per-category sub-queries
    for earnings reports, news coverage, and management commentary.
    Returns a CoverageReport with a 0–1 coverage_score and a targeted
    missing_message listing which categories are absent.

    Args:
        driver: Neo4j driver instance
        settings: Application settings
        company_id: Optional company ticker or ID to filter
        period_start: Optional start date (YYYY-MM-DD)
        period_end: Optional end date (YYYY-MM-DD)

    Returns:
        CoverageReport with completeness status, per-category scores, and
        targeted guidance for missing document types.
    """
    try:
        with driver.session(database=settings.neo4j_database) as session:
            # --- Baseline check: any docs + chunks at all ---
            baseline = _query_category(session, company_id, period_start, period_end, None)

            if not baseline or baseline.get("doc_count", 0) == 0 or baseline.get("chunk_count", 0) == 0:
                return CoverageReport(
                    is_complete=False,
                    missing_sources=list(_CATEGORY_LABELS.keys()),
                    missing_message=_format_missing_data_message(
                        company_id, period_start, period_end,
                        list(_CATEGORY_LABELS.values()),
                    ),
                    coverage_score=0.0,
                )

            # --- Open-ended query guard ---
            is_open_ended = not company_id and not period_start and not period_end
            _MIN_CHUNKS_OPEN_ENDED = 5
            if is_open_ended and baseline["chunk_count"] < _MIN_CHUNKS_OPEN_ENDED:
                return CoverageReport(
                    is_complete=False,
                    missing_sources=[
                        f"Corpus too small for open-ended query "
                        f"(chunks={baseline['chunk_count']}, minimum={_MIN_CHUNKS_OPEN_ENDED})"
                    ],
                    missing_message=_format_missing_data_message(
                        company_id, period_start, period_end, []
                    ),
                    coverage_score=baseline["chunk_count"] / _MIN_CHUNKS_OPEN_ENDED,
                )

            # --- Per-category scoring ---
            earnings = _query_category(
                session, company_id, period_start, period_end, _EARNINGS_SOURCE_TYPES
            )
            news = _query_category(
                session, company_id, period_start, period_end, _NEWS_SOURCE_TYPES
            )
            commentary = _query_category(
                session, company_id, period_start, period_end, _COMMENTARY_SOURCE_TYPES
            )

            scores: dict[str, float] = {
                "earnings_reports": _score_category(earnings),
                "news_coverage": _score_category(news),
                "management_commentary": _score_category(commentary),
            }
            category_score = sum(scores.values()) / len(scores)
            missing_categories = [k for k, v in scores.items() if v < 0.3]

            causal_score = 1.0
            if intent_type == "root_cause_analysis":
                causal_score, _ = _check_causal_coverage(
                    session,
                    company_id=company_id,
                    period_start=period_start,
                    period_end=period_end,
                )
                if causal_score < 0.5:
                    missing_categories.append("causal_event_data")

            aggregate_score = (category_score * 0.6) + (causal_score * 0.4)
            threshold = (
                COVERAGE_THRESHOLD_RCA
                if intent_type == "root_cause_analysis"
                else COVERAGE_THRESHOLD_STANDARD
            )
            missing_labels = [_CATEGORY_LABELS[c] for c in missing_categories]

            return CoverageReport(
                is_complete=aggregate_score >= threshold,
                missing_sources=missing_categories,
                missing_message=_format_supplemental_message(
                    company_id, period_start, period_end, missing_labels
                ),
                coverage_score=aggregate_score,
            )
    except Exception as exc:
        raise RuntimeError(f"Coverage check failed: {exc}") from exc


def _query_category(
    session: object,
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    source_types: list[str] | None,
) -> dict[str, int]:
    """Run a Neo4j query to count docs + chunks for a source-type category."""
    where_clauses: list[str] = [
        "($company_id = '' OR toLower(d.company_id) CONTAINS toLower($company_id))"
    ]
    if source_types is not None:
        where_clauses.append("d.source_type IN $source_types")
    if period_start:
        where_clauses.append("(d.period_end IS NULL OR d.period_end >= $period_start)")
    if period_end:
        where_clauses.append("(d.period_start IS NULL OR d.period_start <= $period_end)")

    where_clause = " AND ".join(where_clauses)
    cypher = (
        "MATCH (d:Document) "
        f"WHERE {where_clause} "
        "OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d) "
        "RETURN COUNT(DISTINCT d) AS doc_count, COUNT(DISTINCT c) AS chunk_count"
    )

    row = session.run(  # type: ignore[union-attr]
        cypher,
        company_id=company_id or "",
        source_types=source_types or [],
        period_start=period_start,
        period_end=period_end,
    ).single()

    if not row:
        return {"doc_count": 0, "chunk_count": 0}
    return {"doc_count": int(row["doc_count"]), "chunk_count": int(row["chunk_count"])}


def _score_category(result: dict[str, int]) -> float:
    """Return a 0–1 coverage score for a single category query result."""
    doc_count = result.get("doc_count", 0)
    chunk_count = result.get("chunk_count", 0)
    if doc_count == 0 or chunk_count == 0:
        return 0.0
    # 0.5 base for having any doc, up to 1.0 as chunk count approaches 10
    return min(1.0, 0.5 + 0.5 * min(chunk_count, 10) / 10)


def _check_causal_coverage(
    session: object,
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
) -> tuple[float, str]:
    """Check whether causal graph edges are present for RCA questions."""
    where_clauses: list[str] = [
        "($company_id = '' OR toLower(d.company_id) CONTAINS toLower($company_id))"
    ]
    if period_start:
        where_clauses.append("(d.period_end IS NULL OR d.period_end >= $period_start)")
    if period_end:
        where_clauses.append("(d.period_start IS NULL OR d.period_start <= $period_end)")

    where_clause = " AND ".join(where_clauses)
    cypher = (
        "MATCH (d:Document) "
        f"WHERE {where_clause} "
        "MATCH (d)-[:CONTAINS]->(:Section)-[:HAS_CHUNK]->(ch:Chunk) "
        "OPTIONAL MATCH (ch)-[:HAS_EVENT]->(:Event)-[r1:CAUSED]->(:Metric) "
        "OPTIONAL MATCH (ch)-[:HAS_EVENT]->(:Event)-[r2:LED_TO]->(:Event) "
        "RETURN count(DISTINCT r1) AS caused_edges, count(DISTINCT r2) AS led_to_edges"
    )

    row = session.run(  # type: ignore[union-attr]
        cypher,
        company_id=company_id or "",
        period_start=period_start,
        period_end=period_end,
    ).single()
    if not row:
        return 0.0, "No causal edges found"

    total_edges = int(row["caused_edges"] or 0) + int(row["led_to_edges"] or 0)
    if total_edges == 0:
        return 0.0, "No causal event data"
    if total_edges < 5:
        return 0.5, f"Limited causal data ({total_edges} edges)"
    return 1.0, f"Good causal coverage ({total_edges} edges)"


def _format_supplemental_message(
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    missing_labels: list[str],
) -> str:
    """Return an advisory message when some categories are absent but docs exist."""
    if not missing_labels:
        return ""
    parts: list[str] = []
    if company_id:
        parts.append(f"company {company_id}")
    if period_start and period_end:
        parts.append(f"period {period_start} to {period_end}")
    elif period_start:
        parts.append(f"from {period_start}")
    elif period_end:
        parts.append(f"until {period_end}")
    query_context = " for ".join(parts) if parts else "for this query"
    labels_text = ", ".join(missing_labels)
    return (
        f"Analysis {query_context} may be incomplete. "
        f"For better root cause coverage, consider uploading: {labels_text}."
    )


def _format_missing_data_message(
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    missing_labels: list[str],
) -> str:
    """Format a user-facing message when no documents are found at all."""
    parts: list[str] = []

    if company_id:
        parts.append(f"company {company_id}")

    if period_start and period_end:
        parts.append(f"period {period_start} to {period_end}")
    elif period_start:
        parts.append(f"from {period_start}")
    elif period_end:
        parts.append(f"until {period_end}")

    query_context = " for ".join(parts) if parts else "for this query"

    if missing_labels:
        needed = ", ".join(missing_labels)
        return (
            f"I don't have enough data {query_context}. "
            f"Please upload: {needed}. "
            "Use the ingest-documents command with the document path and company ID."
        )
    return (
        f"I don't have enough data {query_context}. "
        "Please upload relevant documents (earnings reports, annual reports, press releases) "
        "and try again. "
        "Use the ingest-documents command with the document path and company ID."
    )
