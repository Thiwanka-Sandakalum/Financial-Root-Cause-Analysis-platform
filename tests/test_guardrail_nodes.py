from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from query.agent_query.nodes import _extract_numeric_claims, make_nodes


class _FakeLLM:
    def with_structured_output(self, _schema):
        # These tests only exercise guardrail nodes, so this runnable is never invoked.
        return RunnableLambda(lambda _input: {})


class _SchemaAwareLLM:
    def __init__(self, payload_by_schema: dict[str, dict]):
        self._payload_by_schema = payload_by_schema

    def with_structured_output(self, schema):
        payload = self._payload_by_schema.get(schema.__name__, {})
        return RunnableLambda(lambda _input: payload)


class _SchemaSequenceLLM:
    def __init__(self, payload_sequence_by_schema: dict[str, list[dict]]):
        self._payload_sequence_by_schema = {
            key: list(values) for key, values in payload_sequence_by_schema.items()
        }

    def with_structured_output(self, schema):
        schema_name = schema.__name__

        def _invoke(_input):
            sequence = self._payload_sequence_by_schema.get(schema_name, [])
            if sequence:
                return sequence.pop(0)
            return {}

        return RunnableLambda(_invoke)


class _AnswerLLM:
    def __init__(self, answer_payload: dict):
        self._answer_payload = answer_payload

    def with_structured_output(self, _schema):
        return RunnableLambda(lambda _input: self._answer_payload)


def _mock_driver_with_company_catalog(rows):
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    session.run.return_value.data.return_value = rows
    return driver


def _analysis_payload(question: str, ticker: str | None = "NVDA") -> dict:
    return {
        "question": question,
        "company_ticker": ticker,
        "intent": {
            "intent": "fact_lookup",
            "needs_tables": True,
            "needs_graph_traversal": False,
            "needs_multi_period": False,
            "needs_company_filter": True,
            "confidence": 0.9,
            "rationale": "Financial filing lookup.",
        },
        "retrieval_plan": {
            "top_k_chunks": 8,
            "top_k_tables": 6,
            "hop_depth": 2,
            "required_section_types": [],
            "required_doc_types": [],
            "use_causal_edges": True,
            "use_prev_next_chunks": True,
            "use_company_filter": True,
        },
    }


def test_analyze_request_blocks_non_financial_question_from_llm_domain_gate():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
        {"ticker": "AAPL", "name": "Apple"},
    ])
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "non_financial",
                "domain_reason": "Cooking guidance is outside financial assistant scope.",
                "domain_confidence": "high",
                "domain_signals": ["lifestyle_request"],
                "clarification_question": None,
            }
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"](
        {
            "question": "How do I bake sourdough bread at home?",
            "messages": [],
        }
    )

    assert result["domain_status"] == "non_financial"
    assert result["blocked_reason"] == "non_financial"
    assert result["policy_flags"]["domain_gate_passed"] is False
    assert isinstance(result["final_answer"], dict)


def test_analyze_request_allows_financial_question_from_llm_domain_gate():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "financial_supported",
                "domain_reason": "Asks for company financial metric by period.",
                "domain_confidence": "high",
                "domain_signals": ["financial_metric", "company_period"],
                "clarification_question": None,
            },
            "QueryAnalysis": _analysis_payload("What was NVIDIA revenue in Q2 FY2025?", ticker="NVDA"),
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"](
        {
            "question": "What was NVIDIA revenue in Q2 FY2025?",
            "messages": [],
        }
    )

    assert result["domain_status"] == "financial_supported"
    assert result["policy_flags"]["domain_gate_passed"] is True
    assert result["blocked_reason"] is None


def test_analyze_request_marks_ambiguous_as_uncertain():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "uncertain",
                "domain_reason": "Company mention without explicit financial ask.",
                "domain_confidence": "medium",
                "domain_signals": ["company_only"],
                "clarification_question": "Which filing period or metric should I focus on?",
            }
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"](
        {
            "question": "Tell me about NVIDIA.",
            "messages": [],
        }
    )

    assert result["domain_status"] == "uncertain"
    assert result["blocked_reason"] == "domain_uncertain"
    assert result["policy_flags"]["domain_gate_passed"] is False
    assert isinstance(result["final_answer"], dict)


def test_analyze_request_allows_qualitative_ceo_filing_commentary():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    question = (
        "According to the CEO, why is Generative AI considered a new industrial revolution "
        "rather than just a new software capability?"
    )
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "financial_supported",
                "domain_reason": "Qualitative management commentary tied to filing results is in scope.",
                "domain_confidence": "medium",
                "domain_signals": ["ceo_commentary", "filing_context"],
                "clarification_question": None,
            },
            "QueryAnalysis": _analysis_payload(question, ticker="NVDA"),
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"]({"question": question, "messages": []})

    assert result["policy_flags"]["domain_gate_passed"] is True
    assert result["blocked_reason"] is None


def test_analyze_request_allows_q4_margin_outlook_question():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    question = "What are the anticipated GAAP and Non-GAAP gross margins projected in the Q4 FY2025 outlook?"
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "financial_supported",
                "domain_reason": "Gross margin outlook by period is a filing metric question.",
                "domain_confidence": "high",
                "domain_signals": ["gaap", "non_gaap", "outlook"],
                "clarification_question": None,
            },
            "QueryAnalysis": _analysis_payload(question, ticker="NVDA"),
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"]({"question": question, "messages": []})

    assert result["policy_flags"]["domain_gate_passed"] is True
    assert result["blocked_reason"] is None


def test_analyze_request_allows_networking_growth_driver_question():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    question = (
        "While Data Center compute revenue grew substantially, what was the sequential percentage "
        "change for Networking revenue, and what specific platform helped drive its year-over-year growth?"
    )
    llm = _SchemaAwareLLM(
        {
            "DomainDecision": {
                "domain_status": "financial_supported",
                "domain_reason": "Data Center and Networking growth drivers are in-scope filing performance analysis.",
                "domain_confidence": "high",
                "domain_signals": ["segment_revenue", "sequential_growth", "yoy_driver"],
                "clarification_question": None,
            },
            "QueryAnalysis": _analysis_payload(question, ticker="NVDA"),
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"]({"question": question, "messages": []})

    assert result["policy_flags"]["domain_gate_passed"] is True
    assert result["blocked_reason"] is None


def test_analyze_request_adjudicates_initial_non_financial_to_supported():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    question = (
        "What corporate event or transition occurred on June 7, 2024, "
        "that retroactively impacted all presented share and per-share amounts?"
    )
    llm = _SchemaSequenceLLM(
        {
            "DomainDecision": [
                {
                    "domain_status": "non_financial",
                    "domain_reason": "Misclassified as generic corporate trivia.",
                    "domain_confidence": "medium",
                    "domain_signals": ["corporate_event"],
                    "clarification_question": None,
                },
                {
                    "domain_status": "financial_supported",
                    "domain_reason": "Stock split and per-share adjustments are filing disclosures.",
                    "domain_confidence": "high",
                    "domain_signals": ["stock_split", "per_share_adjustment"],
                    "clarification_question": None,
                },
            ],
            "QueryAnalysis": [
                _analysis_payload(question, ticker="NVDA"),
            ],
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    result = nodes["analyze_request"]({"question": question, "messages": []})

    assert result["domain_status"] == "financial_supported"
    assert result["policy_flags"]["domain_gate_passed"] is True
    assert result["policy_flags"]["domain_adjudication_used"] is True
    assert result["blocked_reason"] is None


def test_analyze_request_clears_stale_block_state_between_turns():
    driver = _mock_driver_with_company_catalog([
        {"ticker": "NVDA", "name": "NVIDIA"},
    ])
    llm = _SchemaSequenceLLM(
        {
            "DomainDecision": [
                {
                    "domain_status": "non_financial",
                    "domain_reason": "Career guidance is outside financial assistant scope.",
                    "domain_confidence": "high",
                    "domain_signals": ["career"],
                    "clarification_question": None,
                },
                {
                    "domain_status": "non_financial",
                    "domain_reason": "Career guidance remains outside scope after adjudication.",
                    "domain_confidence": "high",
                    "domain_signals": ["career"],
                    "clarification_question": None,
                },
                {
                    "domain_status": "financial_supported",
                    "domain_reason": "Filing commitments question is in scope.",
                    "domain_confidence": "high",
                    "domain_signals": ["purchase_obligations"],
                    "clarification_question": None,
                },
            ],
            "QueryAnalysis": [
                _analysis_payload(
                    "Out of $13.2 billion in non-inventory purchase obligations, how much is dedicated to multi-year cloud service agreements?",
                    ticker="NVDA",
                )
            ],
        }
    )
    nodes = make_nodes(driver=driver, llm=llm, embedder=MagicMock())

    first_turn = nodes["analyze_request"](
        {
            "question": "i love to work for nvidia how i ge a offer from them?",
            "messages": [],
        }
    )
    assert first_turn["blocked_reason"] == "non_financial"
    assert isinstance(first_turn["final_answer"], dict)

    second_turn = nodes["analyze_request"](
        {
            "question": "Out of $13.2 billion in non-inventory purchase obligations, how much is dedicated to multi-year cloud service agreements?",
            "messages": [],
            "blocked_reason": first_turn["blocked_reason"],
            "final_answer": first_turn["final_answer"],
        }
    )

    assert second_turn["policy_flags"]["domain_gate_passed"] is True
    assert second_turn["blocked_reason"] is None
    assert second_turn["final_answer"] == {}


def test_assess_readiness_requests_clarification_for_underspecified_query():
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=_SchemaAwareLLM(
            {
                "ReadinessDecision": {
                    "answer_mode": "clarify",
                    "reason": "Missing fiscal year for third-quarter record revenue question.",
                    "missing_slots": ["year"],
                    "clarification_questions": [
                        "Which fiscal year should be used for the third quarter?"
                    ],
                    "ingest_recommendations": [],
                }
            }
        ),
        embedder=MagicMock(),
    )

    result = nodes["assess_readiness"](
        {
            "question": "What was NVIDIA's total record revenue for the third quarter?",
            "domain_status": "financial_supported",
            "policy_flags": {"domain_reason": "in-domain"},
            "intent": {},
        }
    )

    assert result["blocked_reason"] == "needs_clarification"
    assert result["policy_flags"]["readiness_passed"] is False
    assert result["policy_flags"]["answer_mode"] == "clarify"
    assert isinstance(result["final_answer"], dict)


def test_assess_readiness_requests_ingestion_for_missing_live_data():
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=_SchemaAwareLLM(
            {
                "ReadinessDecision": {
                    "answer_mode": "request_ingestion",
                    "reason": "Current corpus lacks present market-cap valuation data.",
                    "missing_slots": ["current_valuation", "ytd_market_cap_change"],
                    "clarification_questions": [],
                    "ingest_recommendations": [
                        "Ingest a recent source with current market capitalization.",
                        "Ingest a source covering year-to-date market value change.",
                    ],
                }
            }
        ),
        embedder=MagicMock(),
    )

    result = nodes["assess_readiness"](
        {
            "question": "How much has Nvidia's total market value grown this year, and what is its current total valuation?",
            "domain_status": "financial_supported",
            "policy_flags": {"domain_reason": "in-domain"},
            "intent": {},
        }
    )

    assert result["blocked_reason"] == "needs_additional_data_ingestion"
    assert result["policy_flags"]["readiness_passed"] is False
    assert result["policy_flags"]["answer_mode"] == "request_ingestion"
    assert isinstance(result["final_answer"], dict)


def test_assess_readiness_overrides_catalog_mismatch_when_company_filter_not_required():
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=_SchemaAwareLLM(
            {
                "ReadinessDecision": {
                    "answer_mode": "request_ingestion",
                    "reason": (
                        "The request is well-specified but company_ticker=None and the company is not in "
                        "the company catalog."
                    ),
                    "missing_slots": ["entity_data"],
                    "clarification_questions": [],
                    "ingest_recommendations": [
                        "Ingest filings for Foxconn first."
                    ],
                }
            }
        ),
        embedder=MagicMock(),
    )

    result = nodes["assess_readiness"](
        {
            "question": "How is Foxconn utilizing digital twins and industrial AI built on NVIDIA Omniverse?",
            "domain_status": "financial_supported",
            "policy_flags": {"domain_reason": "in-domain"},
            "company_ticker": None,
            "intent": {
                "needs_company_filter": False,
            },
        }
    )

    assert result["blocked_reason"] is None
    assert result["policy_flags"]["readiness_passed"] is True
    assert result["policy_flags"]["answer_mode"] == "answer"
    assert result["readiness_decision"]["answer_mode"] == "answer"


def test_quality_gate_blocks_low_evidence_quality():
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=_FakeLLM(),
        embedder=MagicMock(),
    )

    result = nodes["quality_gate"](
        {
            "intent": {"needs_tables": True},
            "evidence": [{"source_type": "chunk", "score": 0.2}],
            "retrieval_quality": {
                "evidence_count": 1,
                "avg_score": 0.2,
                "table_count": 0,
            },
        }
    )

    assert result["blocked_reason"] == "insufficient_evidence_quality"
    assert result["policy_flags"]["quality_gate_passed"] is False
    assert "missing_table_evidence" in result["policy_flags"]["quality_gate_failures"]
    assert result["evidence"] == []
    assert isinstance(result["final_answer"], dict)


def test_quality_gate_passes_when_evidence_is_sufficient():
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=_FakeLLM(),
        embedder=MagicMock(),
    )

    result = nodes["quality_gate"](
        {
            "intent": {"needs_tables": True},
            "evidence": [
                {"source_type": "chunk", "score": 0.8},
                {"source_type": "table", "score": 0.7},
                {"source_type": "chunk", "score": 0.6},
            ],
            "retrieval_quality": {
                "evidence_count": 3,
                "avg_score": 0.7,
                "table_count": 1,
            },
        }
    )

    assert result["policy_flags"]["quality_gate_passed"] is True
    assert result["policy_flags"]["quality_gate_failures"] == []
    assert "blocked_reason" not in result


def test_extract_numeric_claims_normalizes_numbers_and_percentages():
    text = "Revenue was 30,040 in Q2 and margin reached 74.5%, versus 70% last quarter."
    claims = _extract_numeric_claims(text)
    assert "30040" in claims
    assert "74.5%" in claims
    assert "70%" in claims


def test_synthesize_answer_blocks_when_claim_support_is_too_low():
    llm = _AnswerLLM(
        {
            "answer": "Revenue was 9999 and margin was 88%.",
            "bullets": [],
            "confidence": "high",
            "open_questions": [],
        }
    )
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=llm,
        embedder=MagicMock(),
    )

    result = nodes["synthesize_answer"](
        {
            "question": "What was revenue?",
            "intent": {},
            "retrieval_plan": {},
            "graph_paths": [],
            "evidence": [
                {
                    "text": "Revenue was 3000 and margin was 40%.",
                    "source_type": "chunk",
                    "source_id": "c1",
                    "score": 0.8,
                }
            ],
        }
    )

    assert result["blocked_reason"] == "low_claim_support"
    assert result["evaluation_signals"]["claim_support_ratio"] < 0.4
    assert result["evaluation_signals"]["confidence"] == "low"


def test_synthesize_answer_downgrades_confidence_on_partial_claim_support():
    llm = _AnswerLLM(
        {
            "answer": "Revenue was 3000 and margin was 88%.",
            "bullets": [],
            "confidence": "high",
            "open_questions": [],
        }
    )
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=llm,
        embedder=MagicMock(),
    )

    result = nodes["synthesize_answer"](
        {
            "question": "What was revenue?",
            "intent": {},
            "retrieval_plan": {},
            "graph_paths": [],
            "evidence": [
                {
                    "text": "Revenue was 3000 and margin was 40%.",
                    "source_type": "chunk",
                    "source_id": "c1",
                    "score": 0.8,
                },
                {
                    "text": "The company reported 3000 in total revenue.",
                    "source_type": "table",
                    "source_id": "t1",
                    "score": 0.7,
                },
            ],
        }
    )

    assert result["blocked_reason"] is None
    assert result["evaluation_signals"]["claim_total_count"] > 0
    assert result["evaluation_signals"]["unsupported_claim_count"] > 0
    assert result["evaluation_signals"]["confidence_overridden"] is True
    assert result["final_answer"]["confidence"] == "low"


def test_synthesize_answer_enables_visualization_for_trend_queries():
    llm = _AnswerLLM(
        {
            "answer": (
                "In Q2 FY2024, revenue was $10.0 billion. "
                "In Q2 FY2025, revenue was $15.0 billion."
            ),
            "bullets": ["Revenue increased significantly year-over-year."],
            "confidence": "high",
            "open_questions": [],
        }
    )
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=llm,
        embedder=MagicMock(),
    )

    result = nodes["synthesize_answer"](
        {
            "question": "Show revenue trend between Q2 FY2024 and Q2 FY2025.",
            "intent": {
                "intent": "trend_analysis",
                "needs_multi_period": True,
            },
            "retrieval_plan": {},
            "graph_paths": [],
            "evidence": [
                {
                    "text": "Q2 FY2024 revenue was $10.0 billion.",
                    "source_type": "chunk",
                    "source_id": "c1",
                    "score": 0.9,
                },
                {
                    "text": "Q2 FY2025 revenue was $15.0 billion.",
                    "source_type": "table",
                    "source_id": "t1",
                    "score": 0.85,
                },
            ],
        }
    )

    assert result["visualization"]["enabled"] is True
    assert result["visualization"]["chart_type"] == "line"
    assert len(result["visualization"]["data"]) >= 2


def test_synthesize_answer_disables_visualization_for_fact_lookup_queries():
    llm = _AnswerLLM(
        {
            "answer": "NVIDIA's stock trading symbol on Nasdaq is NVDA.",
            "bullets": [],
            "confidence": "high",
            "open_questions": [],
        }
    )
    nodes = make_nodes(
        driver=_mock_driver_with_company_catalog([]),
        llm=llm,
        embedder=MagicMock(),
    )

    result = nodes["synthesize_answer"](
        {
            "question": "What is NVIDIA's stock trading symbol?",
            "intent": {
                "intent": "fact_lookup",
                "needs_multi_period": False,
            },
            "retrieval_plan": {},
            "graph_paths": [],
            "evidence": [
                {
                    "text": "Common Stock trading symbol: NVDA.",
                    "source_type": "chunk",
                    "source_id": "c1",
                    "score": 0.9,
                }
            ],
        }
    )

    assert result["visualization"]["enabled"] is False
    assert result["visualization"]["reason"] == "intent_not_visualization_oriented"
