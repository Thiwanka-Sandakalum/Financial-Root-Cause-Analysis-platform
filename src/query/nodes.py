"""Query pipeline node functions.

Follows the LangGraph canonical pattern: node functions live in their own
module so they can be tested, imported, and reasoned about independently of
graph construction (``orchestration.py``).

Nodes that require external dependencies (settings, driver) are wrapped in
factory functions that bind those dependencies via closure.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from neo4j import Driver

from src.config import Settings
from src.db.neo4j_client import Neo4jClient
from src.llm.gemini import build_embedding_model
from src.query.answer_synthesizer import Citation, SynthesisOutput, synthesize_answer
from src.query.coverage_gate import check_coverage
from src.query.evidence_ranker import rank_evidence
from src.query.groundedness_validator import GroundednessCheck, validate_groundedness
from src.query.query_classifier import QueryIntent, classify_query
from src.query.state import QueryState, QueryWarning
from src.query.graph_expander import causal_chain_expansion, expand_chunk_graph
from src.query.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_driver(driver_or_client: Driver | Neo4jClient | object) -> Driver | object:
    if isinstance(driver_or_client, Neo4jClient):
        return driver_or_client.driver
    return driver_or_client


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts).strip()
    return ""


def _latest_user_query(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _extract_text_from_content(msg.content)
            if text:
                return text
    return ""


def _assistant_text_for_response(final_response: dict[str, Any]) -> str:
    status = final_response.get("status")
    if status == "missing_data":
        return str(
            final_response.get(
                "missing_data_message",
                "I do not have enough data for this query yet.",
            )
        )
    if status == "failed":
        return str(final_response.get("answer") or "I could not process that query.")
    answer = str(final_response.get("answer", "")).strip()
    return answer or "I could not produce an answer for this query."


def _dedupe_chunks(chunks: list[Document]) -> list[Document]:
    by_chunk_id: dict[str, Document] = {}
    for doc in chunks:
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue
        current_score = float(
            doc.metadata.get("final_score", doc.metadata.get("combined_score", 0.0))
        )
        existing = by_chunk_id.get(chunk_id)
        if existing is None:
            by_chunk_id[chunk_id] = doc
            continue
        existing_score = float(
            existing.metadata.get("final_score", existing.metadata.get("combined_score", 0.0))
        )
        if current_score > existing_score:
            by_chunk_id[chunk_id] = doc

    deduped = list(by_chunk_id.values())
    deduped.sort(
        key=lambda item: float(
            item.metadata.get("final_score", item.metadata.get("combined_score", 0.0))
        ),
        reverse=True,
    )
    return deduped


def _build_period_key(period_start: str | None, period_end: str | None) -> str:
    if period_end:
        return period_end
    if period_start:
        return period_start
    return "unknown"


def _raptor_section_search(
    driver: Driver,
    settings: Settings,
    query_embedding: list[float],
    company_id: str | None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Stage-1 RAPTOR retrieval over Section.summary_embedding."""
    if not company_id:
        return []

    cypher = (
        "MATCH (co:Company {company_id: $company_id})-[:PUBLISHED]->(d:Document) "
        "MATCH (d)-[:CONTAINS]->(s:Section) "
        "WHERE s.summary_embedding IS NOT NULL "
        "WITH s, vector.similarity.cosine(s.summary_embedding, $qe) AS sim "
        "WHERE sim > 0.72 "
        "RETURN s.section_id AS section_id, "
        "       s.title AS title, "
        "       s.summary AS summary, "
        "       sim AS score "
        "ORDER BY sim DESC "
        "LIMIT $top_k"
    )
    with driver.session(database=settings.neo4j_database) as session:
        return session.run(
            cypher,
            company_id=company_id,
            qe=query_embedding,
            top_k=top_k,
        ).data()


def response_node(state: QueryState) -> dict[str, Any]:
    """Assemble the final_response dict and append an AIMessage to messages."""
    intent = state.get("intent")
    intent_type = getattr(intent, "intent_type", "other")
    query_text = state.get("query_text", "")

    if not query_text:
        final_response: dict[str, Any] = {
            "status": "failed",
            "answer": "Please send a question so I can run the query pipeline.",
            "citations": [],
            "is_grounded": True,
            "warnings": state.get("warnings", []),
        }
        return {
            "final_response": final_response,
            "messages": [AIMessage(content=_assistant_text_for_response(final_response))],
        }

    if not state.get("coverage_ok", False):
        final_response = {
            "status": "missing_data",
            "answer": "",
            "citations": [],
            "is_grounded": True,
            "missing_data_message": state.get("missing_data_msg", "Missing required data."),
            "warnings": state.get("warnings", []),
        }
        return {
            "final_response": final_response,
            "messages": [AIMessage(content=_assistant_text_for_response(final_response))],
        }

    final_response = {
        "status": "ok" if state.get("is_grounded", True) else "needs_review",
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "is_grounded": state.get("is_grounded", False),
        "unsupported_claims": state.get("unsupported_claims", []),
        "confidence": state.get("synthesis_confidence", 0.0),
        "grounding_confidence": state.get("grounding_confidence", 0.0),
        "intent_type": intent_type,
        "company_id": state.get("company_id"),
        "period_start": state.get("period_start"),
        "period_end": state.get("period_end"),
        "warnings": state.get("warnings", []),
    }
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=_assistant_text_for_response(final_response))],
    }


# ---------------------------------------------------------------------------
# Routing functions (pure state reads — no external dependencies)
# ---------------------------------------------------------------------------


def route_after_classify(state: QueryState) -> str:
    """Short-circuit non-financial queries before any DB or retrieval work."""
    if not state.get("query_text", ""):
        return "format_response"
    intent = state.get("intent")
    if intent is not None and intent.intent_type == "other":
        return "format_response"
    return "check_coverage"


def route_after_coverage(state: QueryState) -> str:
    if state.get("coverage_ok", False):
        return "retrieve_evidence"
    return "format_response"


def route_after_causal(state: QueryState) -> str:
    del state
    return "rank_and_filter"


# ---------------------------------------------------------------------------
# Node factories (bind settings + driver via closure)
# ---------------------------------------------------------------------------


def make_classify_node(settings: Settings):
    """Return classify_node bound to settings."""

    def classify_node(state: QueryState) -> dict[str, Any]:
        query_text = _latest_user_query(state.get("messages", []))
        if not query_text:
            return {
                "query_text": "",
                "coverage_ok": False,
                "intent": QueryIntent(
                    intent_type="other",
                    companies=[],
                    metrics=[],
                    time_range=None,
                    question_text="",
                ),
            }

        intent = classify_query(query_text, settings)

        if intent.intent_type == "other":
            return {
                "intent": intent,
                "coverage_ok": False,
                "missing_data_msg": (
                    "I can only answer questions about company financials, "
                    "earnings, metrics, and investor data. "
                    "Please ask a financial question."
                ),
            }

        company_id = state.get("company_id")
        if not company_id and intent.companies:
            company_id = intent.companies[0]

        period_start = state.get("period_start")
        period_end = state.get("period_end")
        if intent.time_range is not None:
            period_start = period_start or (
                intent.time_range.start.isoformat()
                if intent.time_range.start is not None
                else None
            )
            period_end = period_end or (
                intent.time_range.end.isoformat()
                if intent.time_range.end is not None
                else None
            )

        return {
            "query_text": query_text,
            "intent": intent,
            "company_id": company_id,
            "period_start": period_start,
            "period_end": period_end,
        }

    return classify_node


def make_coverage_node(settings: Settings, driver: Driver | Neo4jClient | object):
    """Return coverage_node bound to settings + driver."""

    def coverage_node(state: QueryState) -> dict[str, Any]:
        intent = state.get("intent")
        active_driver = cast(Driver, _resolve_driver(driver))
        report = check_coverage(
            driver=active_driver,
            settings=settings,
            company_id=state.get("company_id"),
            period_start=state.get("period_start"),
            period_end=state.get("period_end"),
            intent_type=getattr(intent, "intent_type", "fact_lookup"),
        )
        return {
            "coverage_ok": report.is_complete,
            "missing_data_msg": report.missing_message,
            "warnings": state.get("warnings", []),
        }

    return coverage_node


def make_retrieval_node(settings: Settings, driver: Driver | Neo4jClient | object):
    """Return retrieval_node bound to settings + driver."""

    def retrieval_node(state: QueryState) -> dict[str, Any]:
        typed_driver = cast(Driver, _resolve_driver(driver))
        retriever = HybridRetriever(driver=typed_driver, settings=settings)
        query_text = state.get("query_text", "")
        period_key = _build_period_key(state.get("period_start"), state.get("period_end"))
        company_id = state.get("company_id")
        warnings: list[QueryWarning] = list(state.get("warnings", []))

        raptor_sections: list[dict[str, Any]] = []
        try:
            embedding_model = build_embedding_model(settings)
            query_embedding = embedding_model.embed_query(query_text)
            raptor_sections = _raptor_section_search(
                typed_driver,
                settings,
                query_embedding=query_embedding,
                company_id=company_id,
                top_k=4,
            )
        except Exception as exc:
            warnings.append({"stage": "raptor_section_search", "message": str(exc)})
            raptor_sections = []
        raptor_section_ids = {
            str(row.get("section_id"))
            for row in raptor_sections
            if row.get("section_id")
        }

        try:
            retrieved = retriever.invoke(
                query_text,
                top_k=settings.query_retrieval_top_k,
                company_id=company_id,
                period_key=period_key,
                period_end=state.get("period_end"),
            )
        except TypeError:
            retrieved = retriever.invoke(query_text)

        for doc in retrieved:
            section_id = doc.metadata.get("section_id")
            if section_id and str(section_id) in raptor_section_ids:
                boosted = min(1.0, float(doc.metadata.get("combined_score", 0.0)) + 0.10)
                doc.metadata["combined_score"] = boosted
                doc.metadata["raptor_boosted"] = True

        seed_chunk_ids = [
            str(doc.metadata.get("chunk_id"))
            for doc in retrieved[: settings.query_retrieval_top_k]
            if isinstance(doc.metadata.get("chunk_id"), str)
            and doc.metadata.get("chunk_id")
        ]

        if seed_chunk_ids:
            try:
                expanded = expand_chunk_graph(
                    typed_driver,
                    seed_chunk_ids,
                    settings,
                    intent_type="fact_lookup",
                )
            except TypeError:
                expanded = expand_chunk_graph(typed_driver, seed_chunk_ids, settings)
        else:
            expanded = []
        candidate_chunks = _dedupe_chunks(retrieved + expanded)

        return {
            "retrieved_chunks": retrieved,
            "expanded_chunks": expanded,
            "candidate_chunks": candidate_chunks,
            "raptor_sections": raptor_sections,
            "warnings": warnings,
        }

    return retrieval_node


def make_causal_expand_node(settings: Settings, driver: Driver | Neo4jClient | object):
    """Return causal_expand node bound to settings + driver."""

    def causal_expand_node(state: QueryState) -> dict[str, Any]:
        if getattr(state.get("intent"), "intent_type", "") != "root_cause_analysis":
            return {"causal_evidence": []}
        typed_driver = cast(Driver, _resolve_driver(driver))
        seeds = state.get("candidate_chunks", state.get("retrieved_chunks", []))
        seed_ids = [
            str(doc.metadata.get("chunk_id"))
            for doc in seeds
            if isinstance(doc.metadata.get("chunk_id"), str)
            and doc.metadata.get("chunk_id")
        ]
        causal_docs = causal_chain_expansion(typed_driver, seed_ids, settings=settings)
        merged = _dedupe_chunks(seeds + causal_docs)
        return {
            "causal_evidence": causal_docs,
            "candidate_chunks": merged,
            "retrieved_chunks": _dedupe_chunks(state.get("retrieved_chunks", []) + causal_docs),
        }

    return causal_expand_node


def make_ranking_node(settings: Settings):
    """Return ranking_node bound to settings."""

    def ranking_node(state: QueryState) -> dict[str, Any]:
        intent = state.get("intent")
        ranked = rank_evidence(
            query_embedding=None,
            retrieved_chunks=state.get("candidate_chunks", []),
            company_id=state.get("company_id"),
            period_start=state.get("period_start"),
            period_end=state.get("period_end"),
            intent_type=getattr(intent, "intent_type", "fact_lookup"),
            settings=settings,
        )
        return {"ranked_evidence": ranked}

    return ranking_node


def make_synthesis_node(settings: Settings):
    """Return synthesis_node bound to settings."""

    def synthesis_node(state: QueryState) -> dict[str, Any]:
        causal_chain = state.get("causal_evidence", [])
        try:
            synthesis: SynthesisOutput = synthesize_answer(
                query=state.get("query_text", ""),
                ranked_evidence=state.get("ranked_evidence", []),
                settings=settings,
                raptor_sections=state.get("raptor_sections", []),
                causal_chain=causal_chain,
                intent_type=getattr(state.get("intent"), "intent_type", "fact_lookup"),
            )
        except TypeError:
            synthesis = synthesize_answer(
                query=state.get("query_text", ""),
                ranked_evidence=state.get("ranked_evidence", []),
                settings=settings,
            )
        return {
            "answer": synthesis.answer,
            "citations": [c.model_dump() for c in synthesis.citations],
            "synthesis_confidence": synthesis.confidence,
            "synthesis_reasoning": synthesis.reasoning,
        }

    return synthesis_node


def make_grounding_node(settings: Settings):
    """Return grounding_node bound to settings."""

    def grounding_node(state: QueryState) -> dict[str, Any]:
        check: GroundednessCheck = validate_groundedness(
            answer=state.get("answer", ""),
            citations=[Citation(**c) for c in state.get("citations", [])],
            settings=settings,
        )
        return {
            "is_grounded": check.is_grounded,
            "unsupported_claims": check.unsupported_claims,
            "grounding_confidence": check.confidence,
        }

    return grounding_node
