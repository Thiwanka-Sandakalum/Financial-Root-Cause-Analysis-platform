from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired, TransientError

from query.agent_query.evidence_utils import (
    dedupe_documents,
    evidence_item,
    serialize_docs,
    serialize_graph_paths,
    state_dict,
)
from query.agent_query.message_utils import messages_from_input_payload, question_from_state
from query.agent_query.prompts import (
    ANALYZE_PROMPT,
    ANSWER_PROMPT,
    DOMAIN_ADJUDICATE_PROMPT,
    DOMAIN_CLASSIFY_PROMPT,
    READINESS_PROMPT,
    TOOL_PLAN_PROMPT,
)
from query.agent_query.config import GUARDRAIL_THRESHOLDS
from query.agent_query.tool_planner import (
    GRAPH_MODEL_CONTEXT,
    build_default_tool_plan,
    normalize_tool_plan,
)
from query.core.neo4j_query_retriever import (
    retrieve_chunk_documents,
    retrieve_table_documents,
)
from query.core.query_models import (
    DomainDecision,
    FinalAnswer,
    QueryAnalysis,
    QueryState,
    ReadinessDecision,
    RetrievalToolPlan,
    VisualizationSpec,
)


logger = logging.getLogger(__name__)

NUMERIC_PATTERN = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?(?!\w)")
PERIOD_PATTERN = re.compile(r"\b(?:Q[1-4]\s*FY\d{4}|FY\d{4}|20\d{2})\b", re.IGNORECASE)
VALUE_WITH_UNIT_PATTERN = re.compile(
    r"(?P<value>\$?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>billion|million|%)?",
    re.IGNORECASE,
)


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError, TimeoutError, ConnectionError)):
        return "transient"
    if isinstance(exc, (TypeError, ValueError)):
        return "validation"
    if isinstance(exc, Neo4jError):
        return "database"
    return "runtime"


def _is_retryable_error(exc: Exception) -> bool:
    return _classify_error(exc) == "transient"


def _invoke_with_retry(
    operation_name: str,
    fn: Callable[[], Any],
    max_attempts: int = 3,
    base_backoff_seconds: float = 0.2,
) -> tuple[Any, int, str | None]:
    if max_attempts < 1:
        max_attempts = 1

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return result, attempt - 1, None
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_error(exc):
                break
            sleep_for = base_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "%s failed on attempt %s/%s (%s). Retrying in %.2fs",
                operation_name,
                attempt,
                max_attempts,
                _classify_error(exc),
                sleep_for,
            )
            time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


def _normalize_numeric_token(token: str) -> str:
    cleaned = token.strip().replace(",", "")
    if not cleaned:
        return ""
    if cleaned.endswith("%"):
        base = cleaned[:-1]
        try:
            value = float(base)
        except ValueError:
            return ""
        # Keep percentage suffix but normalize numeric formatting.
        return f"{value:g}%"

    try:
        value = float(cleaned)
    except ValueError:
        return ""
    return f"{value:g}"


def _extract_numeric_claims(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    claims: list[str] = []
    seen: set[str] = set()
    for raw in NUMERIC_PATTERN.findall(text):
        normalized = _normalize_numeric_token(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        claims.append(normalized)
    return claims


def _claim_support_diagnostics(answer_text: str, evidence: list[dict]) -> dict[str, Any]:
    claims = _extract_numeric_claims(answer_text)
    if not claims:
        return {
            "claim_total_count": 0,
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
            "claim_support_ratio": 1.0,
            "unsupported_claims": [],
        }

    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)
    normalized_evidence = {
        token
        for token in _extract_numeric_claims(evidence_text)
    }

    supported = [claim for claim in claims if claim in normalized_evidence]
    unsupported = [claim for claim in claims if claim not in normalized_evidence]
    ratio = len(supported) / max(1, len(claims))
    return {
        "claim_total_count": len(claims),
        "supported_claim_count": len(supported),
        "unsupported_claim_count": len(unsupported),
        "claim_support_ratio": ratio,
        "unsupported_claims": unsupported,
    }


def infer_ticker_from_evidence(
    evidence: list[dict],
    min_votes: int = 2,
    min_share: float = 0.6,
) -> str | None:
    tickers = [
        str(item.get("ticker") or "").strip().upper()
        for item in evidence
        if str(item.get("ticker") or "").strip()
    ]
    if not tickers:
        return None

    counts = Counter(tickers)
    ticker, votes = counts.most_common(1)[0]
    share = votes / len(tickers)
    if votes < min_votes:
        return None
    if share < min_share:
        return None
    return ticker


def deterministic_citations_from_evidence(
    evidence: list[dict],
    max_citations: int = 6,
) -> list[dict]:
    citations: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in evidence:
        source_type = str(item.get("source_type") or "").strip().lower()
        source_id = str(item.get("source_id") or "").strip()
        if source_type not in {"chunk", "table", "graph"} or not source_id:
            continue

        dedupe_key = (source_type, source_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        page_value = item.get("page")
        page = page_value if isinstance(page_value, int) else None
        score_value = item.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None

        citations.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "doc_id": item.get("doc_id"),
                "section_id": item.get("section_id"),
                "page": page,
                "score": score,
                "title": item.get("section_title")
                or item.get("title")
                or item.get("doc_type")
                or item.get("doc_id"),
            }
        )
        if len(citations) >= max_citations:
            break

    return citations


def disabled_visualization(reason: str) -> dict[str, Any]:
    return VisualizationSpec(enabled=False, reason=reason).model_dump()


def parse_numeric_value(value_text: str) -> float | None:
    cleaned = value_text.strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_period_value_points(answer_text: str) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return [], None

    points: list[dict[str, Any]] = []
    seen_periods: set[str] = set()
    unit: str | None = None
    for segment in re.split(r"[\n\.;]", answer_text):
        sentence = segment.strip()
        if not sentence:
            continue

        period_match = PERIOD_PATTERN.search(sentence)
        value_match = VALUE_WITH_UNIT_PATTERN.search(sentence)
        if not period_match or not value_match:
            continue

        period = re.sub(r"\s+", " ", period_match.group(0).strip()).upper()
        value = parse_numeric_value(value_match.group("value"))
        if value is None or period in seen_periods:
            continue

        seen_periods.add(period)
        points.append({"period": period, "value": value})

        raw_unit = value_match.group("unit")
        if isinstance(raw_unit, str):
            normalized_unit = raw_unit.strip().lower()
            if normalized_unit:
                unit = normalized_unit

    return points, unit


def build_visualization_payload(
    state: QueryState,
    final_answer: dict[str, Any],
    blocked_reason: str | None,
    citation_coverage: float,
) -> dict[str, Any]:
    if blocked_reason:
        return disabled_visualization("blocked_response")

    confidence = str(final_answer.get("confidence") or "").lower()
    if confidence == "low":
        return disabled_visualization("low_confidence_answer")

    if citation_coverage < GUARDRAIL_THRESHOLDS.citation_coverage_high_conf_floor:
        return disabled_visualization("low_citation_coverage")

    intent = state_dict(state.get("intent"))
    intent_name = str(intent.get("intent") or "").strip().lower()
    needs_multi_period = bool(intent.get("needs_multi_period", False))
    if intent_name not in {"comparison", "trend_analysis", "risk_assessment"} and not needs_multi_period:
        return disabled_visualization("intent_not_visualization_oriented")

    answer_text = final_answer.get("answer")
    if not isinstance(answer_text, str):
        answer_text = ""
    points, unit = extract_period_value_points(answer_text)
    if len(points) < 2:
        return disabled_visualization("insufficient_structured_points")

    chart_type = "line" if intent_name == "trend_analysis" else "grouped_bar"
    question = question_from_state(state)
    citations = final_answer.get("citations")
    if not isinstance(citations, list):
        citations = []

    bullets = final_answer.get("bullets")
    insight = ""
    if isinstance(bullets, list):
        bullet_candidates = [str(item).strip() for item in bullets if str(item).strip()]
        insight = bullet_candidates[0] if bullet_candidates else ""

    return VisualizationSpec(
        enabled=True,
        reason="comparison_or_trend_with_supported_period_values",
        chart_type=chart_type,
        title=(question[:120] if isinstance(question, str) and question.strip() else "Trend and Comparison View"),
        x_field="period",
        y_field="value",
        series=["value"],
        unit=unit,
        data=points,
        insight=insight or None,
        citations=citations[:3],
    ).model_dump()


def make_nodes(
    driver: Driver,
    llm: ChatGoogleGenerativeAI,
    embedder: GoogleGenerativeAIEmbeddings,
) -> dict[str, Callable[[QueryState], dict]]:
    domain_classify_chain = DOMAIN_CLASSIFY_PROMPT | llm.with_structured_output(DomainDecision)
    domain_adjudicate_chain = DOMAIN_ADJUDICATE_PROMPT | llm.with_structured_output(DomainDecision)
    analysis_chain = ANALYZE_PROMPT | llm.with_structured_output(QueryAnalysis)
    readiness_chain = READINESS_PROMPT | llm.with_structured_output(ReadinessDecision)
    tool_plan_chain = TOOL_PLAN_PROMPT | llm.with_structured_output(RetrievalToolPlan)
    answer_chain = ANSWER_PROMPT | llm.with_structured_output(FinalAnswer)
    company_catalog_cache: list[tuple[str, str | None]] | None = None

    def load_company_catalog() -> list[tuple[str, str | None]]:
        nonlocal company_catalog_cache
        if company_catalog_cache is not None:
            return company_catalog_cache

        with driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Company)
                WHERE c.ticker IS NOT NULL
                RETURN DISTINCT toUpper(c.ticker) AS ticker, c.name AS name
                ORDER BY ticker
                """
            ).data()

        company_catalog_cache = [
            (str(row.get("ticker") or "").strip(), row.get("name"))
            for row in rows
            if str(row.get("ticker") or "").strip()
        ]
        return company_catalog_cache

    def format_company_catalog(catalog: list[tuple[str, str | None]]) -> str:
        if not catalog:
            return "No companies available in the graph."

        return "\n".join(
            f"- {ticker}: {name}" if name else f"- {ticker}" for ticker, name in catalog
        )

    def normalize_ticker(raw_ticker: object, allowed_tickers: set[str]) -> str | None:
        if not isinstance(raw_ticker, str):
            return None
        candidate = raw_ticker.strip().upper()
        if not candidate:
            return None
        if not allowed_tickers:
            return candidate
        return candidate if candidate in allowed_tickers else None

    def require_question(state: QueryState) -> str:
        question = question_from_state(state)
        if not question:
            raise ValueError("No question text found in the graph input or messages.")
        return question

    def normalize_domain_decision(payload: dict[str, Any]) -> tuple[str, str, str, list[str], str | None]:
        raw_domain_status = str(payload.get("domain_status") or "uncertain").strip().lower()
        if raw_domain_status not in {"financial_supported", "non_financial", "uncertain"}:
            raw_domain_status = "uncertain"

        domain_reason = payload.get("domain_reason")
        if not isinstance(domain_reason, str):
            domain_reason = ""

        domain_confidence = str(payload.get("domain_confidence") or "medium").strip().lower()
        if domain_confidence not in {"low", "medium", "high"}:
            domain_confidence = "medium"

        raw_signals = payload.get("domain_signals")
        if not isinstance(raw_signals, list):
            raw_signals = []
        domain_signals = [str(item).strip() for item in raw_signals if str(item).strip()]

        clarification_question = payload.get("clarification_question")
        if isinstance(clarification_question, str):
            clarification_question = clarification_question.strip() or None
        else:
            clarification_question = None

        return raw_domain_status, domain_reason, domain_confidence, domain_signals, clarification_question

    def domain_policy_flags(
        gate_passed: bool,
        domain_reason: str,
        domain_confidence: str,
        domain_signals: list[str],
        adjudication_used: bool,
    ) -> dict[str, Any]:
        return {
            "domain_gate_passed": gate_passed,
            "domain_reason": domain_reason,
            "domain_confidence": domain_confidence,
            "domain_signals": domain_signals,
            "domain_adjudication_used": adjudication_used,
        }

    def classify_and_adjudicate_domain(
        question: str,
        explicit_ticker: str | None,
        time_filter: str | None,
        company_catalog: list[tuple[str, str | None]],
    ) -> tuple[dict[str, Any], int, str | None, bool]:
        retry_count = 0
        domain_error_class: str | None = None
        adjudication_used = False

        try:
            domain_result, used_retries, domain_error_class = _invoke_with_retry(
                "classify_domain",
                lambda: domain_classify_chain.invoke(
                    {
                        "question": question,
                        "company_ticker": explicit_ticker,
                        "time_filter": time_filter,
                        "company_catalog": format_company_catalog(company_catalog),
                    }
                ),
            )
            retry_count += used_retries
            domain_payload = state_dict(domain_result)
        except Exception as exc:
            logger.warning("Domain classification failed, defaulting to in-domain planning: %s", exc)
            domain_error_class = _classify_error(exc)
            domain_payload = {
                "domain_status": "financial_supported",
                "domain_reason": "Domain classifier unavailable; defaulting to in-domain planning.",
                "domain_confidence": "low",
                "domain_signals": ["domain_classifier_error"],
                "clarification_question": None,
            }

        raw_domain_status, domain_reason, domain_confidence, domain_signals, clarification_question = (
            normalize_domain_decision(domain_payload)
        )

        if raw_domain_status != "financial_supported":
            adjudication_used = True
            try:
                adjudicated_result, used_retries, domain_error_class = _invoke_with_retry(
                    "adjudicate_domain",
                    lambda: domain_adjudicate_chain.invoke(
                        {
                            "question": question,
                            "initial_domain_status": raw_domain_status,
                            "initial_domain_reason": domain_reason,
                            "initial_domain_confidence": domain_confidence,
                            "company_ticker": explicit_ticker,
                            "time_filter": time_filter,
                            "company_catalog": format_company_catalog(company_catalog),
                        }
                    ),
                )
                retry_count += used_retries
                raw_domain_status, domain_reason, domain_confidence, domain_signals, clarification_question = (
                    normalize_domain_decision(state_dict(adjudicated_result))
                )
            except Exception as exc:
                logger.warning("Domain adjudication failed, defaulting to in-domain planning: %s", exc)
                domain_error_class = _classify_error(exc)
                raw_domain_status = "financial_supported"
                domain_reason = "Domain adjudicator unavailable; defaulting to in-domain planning."
                domain_confidence = "low"
                domain_signals = list(dict.fromkeys([*domain_signals, "domain_adjudicator_error"]))
                clarification_question = None

        decision = {
            "domain_status": raw_domain_status,
            "domain_reason": domain_reason,
            "domain_confidence": domain_confidence,
            "domain_signals": domain_signals,
            "clarification_question": clarification_question,
        }
        return decision, retry_count, domain_error_class, adjudication_used

    def blocked_domain_response(
        domain_status: str,
        clarification_question: str | None,
    ) -> tuple[str, FinalAnswer]:
        blocked_reason = "non_financial" if domain_status == "non_financial" else "domain_uncertain"
        if domain_status == "non_financial":
            fallback = FinalAnswer(
                answer=(
                    "This assistant is configured for financial and filing-related questions only. "
                    "Ask about company filings, metrics, periods, risks, or financial performance."
                ),
                bullets=[],
                confidence="low",
                open_questions=[
                    "Ask a financial question tied to a company or reporting period available in the dataset."
                ],
            )
        else:
            clarification = clarification_question or (
                "Which company, filing period, or financial metric should be analyzed?"
            )
            fallback = FinalAnswer(
                answer=(
                    "The request appears ambiguous for this financial assistant. "
                    "Please clarify the financial metric, filing, company, or period you want."
                ),
                bullets=[],
                confidence="low",
                open_questions=[clarification],
            )
        return blocked_reason, fallback

    def normalize_readiness_decision(payload: dict[str, Any]) -> dict[str, Any]:
        answer_mode = str(payload.get("answer_mode") or "answer").strip().lower()
        if answer_mode not in {"answer", "clarify", "request_ingestion"}:
            answer_mode = "answer"

        reason = payload.get("reason")
        if not isinstance(reason, str):
            reason = ""

        raw_missing = payload.get("missing_slots")
        missing_slots = [str(item).strip() for item in raw_missing] if isinstance(raw_missing, list) else []
        missing_slots = [item for item in missing_slots if item]

        raw_clarify = payload.get("clarification_questions")
        clarification_questions = [str(item).strip() for item in raw_clarify] if isinstance(raw_clarify, list) else []
        clarification_questions = [item for item in clarification_questions if item]

        raw_ingest = payload.get("ingest_recommendations")
        ingest_recommendations = [str(item).strip() for item in raw_ingest] if isinstance(raw_ingest, list) else []
        ingest_recommendations = [item for item in ingest_recommendations if item]

        return {
            "answer_mode": answer_mode,
            "reason": reason,
            "missing_slots": missing_slots,
            "clarification_questions": clarification_questions,
            "ingest_recommendations": ingest_recommendations,
        }

    def should_override_ingestion_for_catalog_mismatch(
        decision: dict[str, Any],
        state: QueryState,
    ) -> bool:
        if str(decision.get("answer_mode") or "") != "request_ingestion":
            return False

        intent = state_dict(state.get("intent"))
        if bool(intent.get("needs_company_filter", False)):
            return False

        if state.get("company_ticker"):
            return False

        reason = str(decision.get("reason") or "").strip().lower()
        missing_slots_raw = decision.get("missing_slots")
        missing_slots_items = missing_slots_raw if isinstance(missing_slots_raw, list) else []
        missing_slots = {
            str(item).strip().lower()
            for item in missing_slots_items
            if str(item).strip()
        }

        catalog_reason_tokens = {
            "company catalog",
            "not in the company catalog",
            "company_ticker",
            "company ticker",
            "ticker",
        }
        has_catalog_signal = any(token in reason for token in catalog_reason_tokens)
        catalog_slots = {"entity_data", "company_ticker", "ticker", "entity"}
        only_catalog_slots = bool(missing_slots) and missing_slots.issubset(catalog_slots)

        return has_catalog_signal or only_catalog_slots

    def readiness_blocked_response(decision: dict[str, Any], question: str) -> tuple[str, FinalAnswer]:
        answer_mode = str(decision.get("answer_mode") or "answer")
        reason = str(decision.get("reason") or "").strip()

        if answer_mode == "clarify":
            questions = decision.get("clarification_questions") or []
            if not isinstance(questions, list):
                questions = []
            filtered_questions = [str(item).strip() for item in questions if str(item).strip()][:3]
            if not filtered_questions:
                filtered_questions = [
                    "Which fiscal year or reporting period should be used?",
                    "Which exact metric or scope should be prioritized?",
                ]
            fallback = FinalAnswer(
                answer=(
                    "I need a bit more detail before I can answer this reliably from the available filings context."
                ),
                bullets=[reason] if reason else [],
                confidence="low",
                open_questions=filtered_questions,
            )
            return "needs_clarification", fallback

        ingest_recs = decision.get("ingest_recommendations") or []
        if not isinstance(ingest_recs, list):
            ingest_recs = []
        filtered_recs = [str(item).strip() for item in ingest_recs if str(item).strip()][:4]
        if not filtered_recs:
            filtered_recs = [
                "Ingest a newer filing or report that explicitly contains the requested metric and period.",
                "Include a source covering the exact timeframe needed for this question.",
                "Re-run ingestion and ask the same question again.",
            ]

        fallback = FinalAnswer(
            answer=(
                "The current knowledge base does not contain enough evidence to answer this request reliably. "
                "Please ingest related source documents, then re-ask the same question."
            ),
            bullets=[reason] if reason else [],
            confidence="low",
            open_questions=filtered_recs,
        )
        return "needs_additional_data_ingestion", fallback

    def to_int(value: int | float | str | None, default: int, minimum: int = 0) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else default

    def retrieval_context(state: QueryState) -> tuple[dict, dict, dict, set[str], str | None, str | None]:
        plan = state_dict(state.get("retrieval_plan"))
        intent = state_dict(state.get("intent"))

        raw_tool_plan = state_dict(state.get("tool_plan"))
        tool_plan = raw_tool_plan or build_default_tool_plan(intent, plan).model_dump()

        selected_tools = {
            str(tool).strip()
            for tool in (tool_plan.get("selected_tools") or [])
            if str(tool).strip()
        }
        use_company_filter = bool(
            tool_plan.get("use_company_filter", plan.get("use_company_filter", True))
        )
        ticker = state.get("company_ticker") if use_company_filter else None
        time_filter = state.get("time_filter")
        return plan, intent, tool_plan, selected_tools, ticker, time_filter

    def analyze_request(state: QueryState) -> dict:
        question = require_question(state)
        messages = state.get("messages") or messages_from_input_payload(state.get("input"))
        retry_count = int(state.get("retry_count") or 0)

        if not messages and question:
            messages = [HumanMessage(content=question)]

        catalog = load_company_catalog()
        allowed_tickers = {ticker for ticker, _ in catalog}
        explicit_ticker = normalize_ticker(state.get("company_ticker"), allowed_tickers)
        normalized_question = question
        resolved_ticker = explicit_ticker

        domain_decision, domain_retries, domain_error_class, adjudication_used = classify_and_adjudicate_domain(
            question=question,
            explicit_ticker=explicit_ticker,
            time_filter=state.get("time_filter"),
            company_catalog=catalog,
        )
        retry_count += domain_retries

        raw_domain_status = domain_decision["domain_status"]
        domain_reason = domain_decision["domain_reason"]
        domain_confidence = domain_decision["domain_confidence"]
        domain_signals = domain_decision["domain_signals"]
        clarification_question = domain_decision["clarification_question"]

        if raw_domain_status in {"non_financial", "uncertain"}:
            blocked_reason, fallback = blocked_domain_response(
                domain_status=raw_domain_status,
                clarification_question=clarification_question,
            )

            return {
                "question": normalized_question,
                "company_ticker": resolved_ticker,
                "time_filter": state.get("time_filter"),
                "domain_status": raw_domain_status,
                "policy_flags": domain_policy_flags(
                    gate_passed=False,
                    domain_reason=domain_reason,
                    domain_confidence=domain_confidence,
                    domain_signals=domain_signals,
                    adjudication_used=adjudication_used,
                ),
                "blocked_reason": blocked_reason,
                "final_answer": fallback.model_dump(),
                "messages": [AIMessage(content=fallback.answer)],
                "retry_count": retry_count,
                "error_class": domain_error_class,
                "evaluation_signals": {
                    "citation_coverage": 0.0,
                    "has_open_questions": True,
                    "confidence": "low",
                },
            }

        result, used_retries, analysis_error_class = _invoke_with_retry(
            "analyze_request",
            lambda: analysis_chain.invoke(
                {
                    "question": question,
                    "domain_status": raw_domain_status,
                    "domain_reason": domain_reason,
                    "domain_confidence": domain_confidence,
                    "domain_signals": ", ".join(domain_signals),
                    "company_ticker": explicit_ticker,
                    "time_filter": state.get("time_filter"),
                    "company_catalog": format_company_catalog(catalog),
                }
            ),
        )
        retry_count += used_retries
        payload = state_dict(result)

        resolved_ticker = explicit_ticker or normalize_ticker(
            payload.get("company_ticker"),
            allowed_tickers,
        )
        normalized_question = payload.get("question")
        if not isinstance(normalized_question, str) or not normalized_question.strip():
            normalized_question = question

        return {
            "question": normalized_question,
            "company_ticker": resolved_ticker,
            "time_filter": state.get("time_filter"),
            "domain_status": raw_domain_status,
            "blocked_reason": None,
            "final_answer": {},
            "policy_flags": domain_policy_flags(
                gate_passed=True,
                domain_reason=domain_reason,
                domain_confidence=domain_confidence,
                domain_signals=domain_signals,
                adjudication_used=adjudication_used,
            ),
            "intent": payload.get("intent") or {},
            "retrieval_plan": payload.get("retrieval_plan") or {},
            "messages": messages,
            "retry_count": retry_count,
            "error_class": analysis_error_class or domain_error_class,
        }

    def assess_readiness(state: QueryState) -> dict:
        question = require_question(state)
        retry_count = int(state.get("retry_count") or 0)

        try:
            readiness_result, used_retries, readiness_error = _invoke_with_retry(
                "assess_readiness",
                lambda: readiness_chain.invoke(
                    {
                        "question": question,
                        "domain_status": state.get("domain_status") or "financial_supported",
                        "domain_reason": state_dict(state.get("policy_flags")).get("domain_reason") or "",
                        "company_ticker": state.get("company_ticker"),
                        "time_filter": state.get("time_filter"),
                        "intent_json": state_dict(state.get("intent")),
                    }
                ),
            )
            retry_count += used_retries
            decision = normalize_readiness_decision(state_dict(readiness_result))
        except Exception as exc:
            logger.warning("Readiness assessment failed, defaulting to answer mode: %s", exc)
            readiness_error = _classify_error(exc)
            decision = {
                "answer_mode": "answer",
                "reason": "Readiness controller unavailable; proceeding with retrieval.",
                "missing_slots": [],
                "clarification_questions": [],
                "ingest_recommendations": [],
            }

        if should_override_ingestion_for_catalog_mismatch(decision, state):
            decision = {
                **decision,
                "answer_mode": "answer",
                "reason": (
                    "Proceeding with retrieval without a strict company filter because the request does "
                    "not require company-catalog membership."
                ),
                "missing_slots": [],
                "clarification_questions": [],
                "ingest_recommendations": [],
            }

        if decision.get("answer_mode") in {"clarify", "request_ingestion"}:
            blocked_reason, fallback = readiness_blocked_response(decision, question)
            return {
                "readiness_decision": decision,
                "blocked_reason": blocked_reason,
                "final_answer": fallback.model_dump(),
                "messages": [AIMessage(content=fallback.answer)],
                "retry_count": retry_count,
                "error_class": readiness_error,
                "evaluation_signals": {
                    "citation_coverage": 0.0,
                    "has_open_questions": True,
                    "confidence": "low",
                },
                "policy_flags": {
                    **state_dict(state.get("policy_flags")),
                    "readiness_passed": False,
                    "answer_mode": decision.get("answer_mode"),
                    "readiness_reason": decision.get("reason") or "",
                },
            }

        return {
            "readiness_decision": decision,
            "blocked_reason": None,
            "final_answer": {},
            "retry_count": retry_count,
            "error_class": readiness_error,
            "policy_flags": {
                **state_dict(state.get("policy_flags")),
                "readiness_passed": True,
                "answer_mode": "answer",
                "readiness_reason": decision.get("reason") or "",
            },
        }

    def plan_retrieval_tools(state: QueryState) -> dict:
        question = require_question(state)
        plan, intent, _, _, _, _ = retrieval_context(state)
        retry_count = int(state.get("retry_count") or 0)
        tool_plan_payload = {}
        error_class: str | None = None
        used_retries = 0

        try:
            tool_plan_result, used_retries, error_class = _invoke_with_retry(
                "plan_retrieval_tools",
                lambda: tool_plan_chain.invoke(
                    {
                        "question": question,
                        "intent_json": state_dict(intent),
                        "plan_json": state_dict(plan),
                        "graph_model": GRAPH_MODEL_CONTEXT,
                    }
                ),
            )
            tool_plan_payload = state_dict(tool_plan_result)
        except (TypeError, ValueError) as exc:
            logger.warning("Tool plan generation failed, using heuristic fallback: %s", exc)
            tool_plan_payload = {}
            used_retries = 0
            error_class = _classify_error(exc)
        except RuntimeError as exc:
            logger.warning("Tool plan generation runtime error, using heuristic fallback: %s", exc)
            tool_plan_payload = {}
            used_retries = 0
            error_class = _classify_error(exc)

        tool_plan = normalize_tool_plan(tool_plan_payload, intent, plan)
        return {
            "tool_plan": tool_plan.model_dump(),
            "retry_count": retry_count + used_retries,
            "error_class": error_class,
        }

    def retrieve_with_tools(state: QueryState) -> dict:
        question = require_question(state)
        plan, _, tool_plan, selected_tools, ticker, time_filter = retrieval_context(state)
        retry_count = int(state.get("retry_count") or 0)
        error_class: str | None = None

        chunk_top_k = to_int(tool_plan.get("chunk_top_k", plan.get("top_k_chunks", 8)), 8, minimum=1)
        table_top_k = to_int(tool_plan.get("table_top_k", plan.get("top_k_tables", 6)), 6, minimum=0)
        use_causal_edges = bool(tool_plan.get("use_causal_edges", plan.get("use_causal_edges", True)))

        embedding = embedder.embed_query(question)

        chunk_hits: list[dict] = []
        table_hits: list[dict] = []
        graph_paths: list[dict] = []

        if "chunk_search" in selected_tools:
            try:
                chunk_hits, used_retries, error_class = _invoke_with_retry(
                    "retrieve_chunk_documents",
                    lambda: retrieve_chunk_documents(
                        driver=driver,
                        embedding=embedding,
                        top_k=chunk_top_k,
                        company_ticker=ticker,
                        time_filter=time_filter,
                        use_causal_edges=use_causal_edges,
                    ),
                )
                retry_count += used_retries
            except Exception as exc:
                logger.warning("Chunk retrieval failed after retries: %s", exc)
                error_class = _classify_error(exc)
                chunk_hits = []

        if "table_search" in selected_tools:
            if table_top_k > 0:
                try:
                    table_hits, used_retries, error_class = _invoke_with_retry(
                        "retrieve_table_documents",
                        lambda: retrieve_table_documents(
                            driver=driver,
                            embedding=embedding,
                            top_k=table_top_k,
                            company_ticker=ticker,
                            time_filter=time_filter,
                        ),
                    )
                    retry_count += used_retries
                except Exception as exc:
                    logger.warning("Table retrieval failed after retries: %s", exc)
                    error_class = _classify_error(exc)
                    table_hits = []

        seed_docs = dedupe_documents(chunk_hits + table_hits)
        if "graph_traversal" in selected_tools:
            if not seed_docs:
                try:
                    chunk_hits, used_retries, error_class = _invoke_with_retry(
                        "retrieve_chunk_documents_seed",
                        lambda: retrieve_chunk_documents(
                            driver=driver,
                            embedding=embedding,
                            top_k=chunk_top_k,
                            company_ticker=ticker,
                            time_filter=time_filter,
                            use_causal_edges=use_causal_edges,
                        ),
                    )
                    retry_count += used_retries
                except Exception as exc:
                    logger.warning("Seed chunk retrieval failed after retries: %s", exc)
                    error_class = _classify_error(exc)
                    chunk_hits = []
                seed_docs = dedupe_documents(chunk_hits + table_hits)

            graph_paths = [dict(doc.get("metadata", {})) for doc in seed_docs]

        avg_score = 0.0
        if seed_docs:
            scores = [
                float(doc.get("metadata", {}).get("score", 0.0))
                for doc in seed_docs
                if isinstance(doc.get("metadata", {}).get("score"), (int, float))
            ]
            avg_score = (sum(scores) / len(scores)) if scores else 0.0

        return {
            "chunk_hits": chunk_hits,
            "table_hits": table_hits,
            "graph_paths": graph_paths,
            "tool_plan": tool_plan,
            "retry_count": retry_count,
            "error_class": error_class,
            "retrieval_quality": {
                "evidence_count": len(seed_docs),
                "avg_score": avg_score,
                "used_tools": sorted(selected_tools),
            },
        }

    def merge_and_rank_evidence(state: QueryState) -> dict:
        chunk_hits = state.get("chunk_hits") or []
        table_hits = state.get("table_hits") or []
        all_docs = dedupe_documents(chunk_hits + table_hits)
        evidence = [evidence_item(doc) for doc in all_docs]
        current_ticker = state.get("company_ticker")
        inferred_ticker = current_ticker or infer_ticker_from_evidence(evidence)
        scores: list[float] = []
        for item in evidence:
            score_value = item.get("score")
            if isinstance(score_value, (int, float)):
                scores.append(float(score_value))
        return {
            "evidence": evidence,
            "company_ticker": inferred_ticker,
            "retrieval_quality": {
                "evidence_count": len(evidence),
                "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
                "chunk_count": len([e for e in evidence if e.get("source_type") == "chunk"]),
                "table_count": len([e for e in evidence if e.get("source_type") == "table"]),
            },
        }

    def quality_gate(state: QueryState) -> dict:
        evidence = state.get("evidence") or []
        intent = state_dict(state.get("intent"))
        rq = state_dict(state.get("retrieval_quality"))

        evidence_count = int(rq.get("evidence_count") or len(evidence))
        avg_score = float(rq.get("avg_score") or 0.0)
        table_count = int(rq.get("table_count") or 0)

        min_evidence = GUARDRAIL_THRESHOLDS.quality_min_evidence_count
        min_avg_score = GUARDRAIL_THRESHOLDS.quality_min_avg_score

        failures: list[str] = []
        if evidence_count < min_evidence:
            failures.append("insufficient_evidence_count")
        if avg_score < min_avg_score:
            failures.append("low_retrieval_score")
        if bool(intent.get("needs_tables")) and table_count < 1:
            failures.append("missing_table_evidence")

        if failures:
            guidance: list[str] = []
            if "insufficient_evidence_count" in failures:
                guidance.append("Ingest additional filings or disclosures covering the requested metric and period.")
            if "low_retrieval_score" in failures:
                guidance.append("Ingest a source that directly states the requested value or comparison.")
            if "missing_table_evidence" in failures:
                guidance.append("Ingest a tabular source for the requested metric (for example, earnings table or outlook table).")
            if not guidance:
                guidance.append("Ingest related documents and re-ask this question.")

            fallback = FinalAnswer(
                answer=(
                    "I do not have enough high-quality evidence to answer this reliably. "
                    "Please ingest related documents and re-ask the same question."
                ),
                bullets=[],
                confidence="low",
                open_questions=guidance,
            )
            return {
                "evidence": [],
                "blocked_reason": "insufficient_evidence_quality",
                "final_answer": fallback.model_dump(),
                "messages": [AIMessage(content=fallback.answer)],
                "policy_flags": {
                    "quality_gate_passed": False,
                    "quality_gate_failures": failures,
                },
                "evaluation_signals": {
                    "citation_coverage": 0.0,
                    "has_open_questions": True,
                    "confidence": "low",
                },
            }

        return {
            "policy_flags": {
                "quality_gate_passed": True,
                "quality_gate_failures": [],
            }
        }

    def synthesize_answer(state: QueryState) -> dict:
        if state.get("blocked_reason") and state.get("final_answer"):
            final_answer = state_dict(state.get("final_answer"))
            answer_text = final_answer.get("answer") if isinstance(final_answer, dict) else ""
            if not isinstance(answer_text, str):
                answer_text = ""
            return {
                "final_answer": final_answer,
                "visualization": disabled_visualization("blocked_response"),
                "messages": [AIMessage(content=answer_text)],
            }

        evidence = state.get("evidence") or []
        retry_count = int(state.get("retry_count") or 0)
        error_class = state.get("error_class")

        if not evidence:
            fallback = FinalAnswer(
                answer="The current evidence is insufficient to give a reliable answer.",
                bullets=[],
                confidence="low",
            )
            return {
                "final_answer": fallback.model_dump(),
                "visualization": disabled_visualization("insufficient_evidence"),
                "messages": [AIMessage(content=fallback.answer)],
                "blocked_reason": state.get("blocked_reason") or "insufficient_evidence",
            }

        try:
            result, used_retries, answer_error_class = _invoke_with_retry(
                "synthesize_answer",
                lambda: answer_chain.invoke(
                    {
                        "question": question_from_state(state),
                        "company_ticker": state.get("company_ticker"),
                        "time_filter": state.get("time_filter"),
                        "intent_json": state_dict(state.get("intent")),
                        "plan_json": state_dict(state.get("retrieval_plan")),
                        "evidence_text": serialize_docs(
                            [{"page_content": item.get("text", ""), "metadata": item} for item in evidence]
                        ),
                        "graph_text": serialize_graph_paths(state.get("graph_paths") or []),
                    }
                ),
            )
        except Exception as exc:
            classified = _classify_error(exc)
            logger.warning("Answer synthesis failed after retries: %s", exc)
            fallback = FinalAnswer(
                answer="A generation error occurred while composing the answer. Please retry.",
                bullets=[],
                confidence="low",
            )
            return {
                "final_answer": fallback.model_dump(),
                "visualization": disabled_visualization("generation_error"),
                "messages": [AIMessage(content=fallback.answer)],
                "retry_count": retry_count,
                "error_class": classified,
                "blocked_reason": "generation_error",
                "evaluation_signals": {
                    "citation_coverage": 0.0,
                    "has_open_questions": False,
                    "confidence": "low",
                },
            }
        final_answer = state_dict(result)
        claim_diag = {
            "claim_total_count": 0,
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
            "claim_support_ratio": 1.0,
            "unsupported_claims": [],
        }
        confidence_overridden = False
        blocked_reason = state.get("blocked_reason")

        if isinstance(final_answer, dict):
            citations = deterministic_citations_from_evidence(evidence)
            final_answer["citations"] = citations
            citation_coverage = len(citations) / max(1, len(evidence))

            answer_body = final_answer.get("answer")
            answer_text_for_check = answer_body if isinstance(answer_body, str) else ""
            claim_diag = _claim_support_diagnostics(answer_text_for_check, evidence)

            support_ratio = float(claim_diag.get("claim_support_ratio", 1.0))
            claim_total = int(claim_diag.get("claim_total_count", 0))
            current_conf = str(final_answer.get("confidence") or "medium").lower()

            if claim_total > 0 and support_ratio < GUARDRAIL_THRESHOLDS.claim_support_block_threshold:
                final_answer = FinalAnswer(
                    answer=(
                        "The available evidence is not sufficient to confidently support the numeric claims for this request. "
                        "Please refine the question, scope to a specific period, or ingest additional filings."
                    ),
                    bullets=[],
                    confidence="low",
                    open_questions=[
                        "Which exact filing period or metric should be prioritized for retrieval?"
                    ],
                ).model_dump()
                final_answer["citations"] = citations
                blocked_reason = "low_claim_support"
                confidence_overridden = True
            elif claim_total > 0 and support_ratio < GUARDRAIL_THRESHOLDS.claim_support_downgrade_threshold:
                # Partial support: downgrade confidence and attach open question.
                final_answer["confidence"] = "low"
                open_questions = final_answer.get("open_questions")
                if not isinstance(open_questions, list):
                    open_questions = []
                open_questions.append(
                    "Some numeric claims were only partially supported by retrieved evidence."
                )
                final_answer["open_questions"] = list(dict.fromkeys(str(q) for q in open_questions if q))
                confidence_overridden = True
            elif current_conf == "high" and citation_coverage < GUARDRAIL_THRESHOLDS.citation_coverage_high_conf_floor:
                final_answer["confidence"] = "medium"
                confidence_overridden = True
        else:
            citation_coverage = 0.0
        answer_text = final_answer.get("answer") if isinstance(final_answer, dict) else ""
        if not isinstance(answer_text, str):
            answer_text = ""
        visualization = build_visualization_payload(
            state=state,
            final_answer=final_answer if isinstance(final_answer, dict) else {},
            blocked_reason=blocked_reason,
            citation_coverage=citation_coverage,
        )
        return {
            "final_answer": final_answer,
            "visualization": visualization,
            "messages": [AIMessage(content=answer_text)],
            "retry_count": retry_count + used_retries,
            "error_class": answer_error_class or error_class,
            "blocked_reason": blocked_reason,
            "evaluation_signals": {
                "citation_coverage": citation_coverage,
                "has_open_questions": bool(final_answer.get("open_questions")) if isinstance(final_answer, dict) else False,
                "confidence": final_answer.get("confidence") if isinstance(final_answer, dict) else "low",
                "claim_support_ratio": float(claim_diag.get("claim_support_ratio", 1.0)),
                "claim_total_count": int(claim_diag.get("claim_total_count", 0)),
                "unsupported_claim_count": int(claim_diag.get("unsupported_claim_count", 0)),
                "confidence_overridden": confidence_overridden,
            },
        }

    return {
        "analyze_request": analyze_request,
        "assess_readiness": assess_readiness,
        "plan_retrieval_tools": plan_retrieval_tools,
        "retrieve_with_tools": retrieve_with_tools,
        "merge_and_rank_evidence": merge_and_rank_evidence,
        "quality_gate": quality_gate,
        "synthesize_answer": synthesize_answer,
    }
