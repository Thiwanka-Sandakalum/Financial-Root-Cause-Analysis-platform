"""Query pipeline graph construction.

Follows the LangGraph canonical pattern: this module only wires the graph.
State types live in ``state.py``; node functions live in ``nodes.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on
from neo4j import Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from src.config import Settings, get_settings
from src.db.neo4j_client import Neo4jClient

# Re-export state types for backward compatibility
from src.query.state import (  # noqa: F401
    InputState,
    InternalState,
    OutputState,
    QueryPipeline,
    QueryState,
)
from src.query.nodes import (
    make_causal_expand_node,
    make_classify_node,
    make_coverage_node,
    make_grounding_node,
    make_ranking_node,
    make_retrieval_node,
    make_synthesis_node,
    response_node,
    route_after_classify,
    route_after_causal,
    route_after_coverage,
)

# Re-export symbols that tests monkeypatch on this module's namespace
from src.query.answer_synthesizer import synthesize_answer  # noqa: F401
from src.query.coverage_gate import check_coverage  # noqa: F401
from src.query.groundedness_validator import validate_groundedness  # noqa: F401
from src.query.query_classifier import classify_query  # noqa: F401
from src.query.graph_expander import expand_chunk_graph  # noqa: F401
from src.query.hybrid_retriever import HybridRetriever  # noqa: F401


def _retry_on_transient(exc: BaseException) -> bool:
    """Retry only transient infra/model failures for IO-heavy query nodes."""
    if isinstance(exc, (ServiceUnavailable, SessionExpired, ConnectionError, TimeoutError)):
        return True
    if default_retry_on(exc):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        transient_signals = (
            "timeout",
            "temporar",
            "rate limit",
            "defunct connection",
            "service unavailable",
            "failed to read from defunct connection",
        )
        return any(signal in text for signal in transient_signals)
    return False


def build_query_graph(settings: Settings, driver: Driver | Neo4jClient | object):
    """Compile the query StateGraph with all nodes and edges wired.

    Args:
        settings: Pipeline configuration.
        driver: Neo4j Driver or Neo4jClient for database access.

    Returns:
        Compiled StateGraph with InternalState, InputState, and OutputState schemas.
    """
    builder = StateGraph(InternalState, input_schema=InputState, output_schema=OutputState)

    io_retry = RetryPolicy(max_attempts=2, retry_on=_retry_on_transient)

    builder.add_node("classify_query", make_classify_node(settings), retry_policy=io_retry)
    builder.add_node("check_coverage", make_coverage_node(settings, driver), retry_policy=io_retry)
    builder.add_node("retrieve_evidence", make_retrieval_node(settings, driver), retry_policy=io_retry)
    builder.add_node("causal_expand", make_causal_expand_node(settings, driver), retry_policy=io_retry)
    builder.add_node("rank_and_filter", make_ranking_node(settings))
    builder.add_node("synthesize_with_citations", make_synthesis_node(settings), retry_policy=io_retry)
    builder.add_node("validate_grounding", make_grounding_node(settings), retry_policy=io_retry)
    builder.add_node("format_response", response_node)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "check_coverage": "check_coverage",
            "format_response": "format_response",
        },
    )
    builder.add_conditional_edges(
        "check_coverage",
        route_after_coverage,
        {
            "retrieve_evidence": "retrieve_evidence",
            "format_response": "format_response",
        },
    )
    builder.add_edge("retrieve_evidence", "causal_expand")
    builder.add_conditional_edges(
        "causal_expand",
        route_after_causal,
        {
            "rank_and_filter": "rank_and_filter",
        },
    )
    builder.add_edge("rank_and_filter", "synthesize_with_citations")
    builder.add_edge("synthesize_with_citations", "validate_grounding")
    builder.add_edge("validate_grounding", "format_response")
    builder.add_edge("format_response", END)

    return builder.compile()


def run_query(
    query: str,
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    settings: Settings,
    driver: Driver,
    pipeline: QueryPipeline | None = None,
) -> dict[str, Any]:
    """Backward-compatible shim: convert legacy string input into a chat message."""
    return run_query_messages(
        messages=[HumanMessage(content=query)],
        company_id=company_id,
        period_start=period_start,
        period_end=period_end,
        settings=settings,
        driver=driver,
        pipeline=pipeline,
    )


def run_query_messages(
    messages: Sequence[BaseMessage],
    company_id: str | None,
    period_start: str | None,
    period_end: str | None,
    settings: Settings,
    driver: Driver,
    pipeline: QueryPipeline | None = None,
) -> dict[str, Any]:
    """Run query through messages-based pipeline."""
    if pipeline is not None:
        return pipeline.invoke(
            messages=messages,
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

    transient_pipeline = QueryPipeline(settings, driver)
    try:
        return transient_pipeline.invoke(
            messages=messages,
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )
    finally:
        transient_pipeline.close()


# ============================================================================
# LangGraph Studio / Dev Server Export
# ============================================================================
# The 'agent' export is specifically for LangGraph Studio and the dev server.
# Production code uses the QueryPipeline singleton in service.py instead.
# The try/except lets tests import this module without a live Neo4j instance.
# ============================================================================

try:
    _dev_settings = get_settings()
    _dev_neo4j_client = Neo4jClient(_dev_settings)
    agent = build_query_graph(_dev_settings, _dev_neo4j_client)
except Exception:  # pragma: no cover
    agent = None  # type: ignore[assignment]
