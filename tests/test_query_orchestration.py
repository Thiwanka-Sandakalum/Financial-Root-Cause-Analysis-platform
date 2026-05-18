"""Tests for query orchestration graph routing and response shaping."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.config import Settings
from src.query.answer_synthesizer import Citation, SynthesisOutput
from src.query.coverage_gate import CoverageReport
from src.query.orchestration import build_query_graph, run_query
from src.query.query_classifier import QueryIntent


class _FakeRetriever:
    def __init__(self, driver: object, settings: Settings) -> None:
        del driver
        del settings

    def invoke(self, query: str) -> list[Document]:
        del query
        return [
            Document(
                page_content="NVIDIA revenue in Q2 FY2024 was $26.0 billion.",
                metadata={
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "company_id": "NVDA",
                    "combined_score": 0.92,
                    "confidence": 0.95,
                    "period_start": "2024-04-01",
                    "period_end": "2024-06-30",
                },
            )
        ]


def _settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )


def test_run_query_returns_missing_data_when_coverage_fails(monkeypatch: Any) -> None:
    settings = _settings()

    monkeypatch.setattr(
        "src.query.nodes.classify_query",
        lambda query, settings: QueryIntent(
            intent_type="fact_lookup",
            companies=["NVDA"],
            metrics=["revenue"],
            question_text=query,
        ),
    )
    monkeypatch.setattr(
        "src.query.nodes.check_coverage",
        lambda **kwargs: CoverageReport(
            is_complete=False,
            missing_sources=["q2 earnings"],
            missing_message="Please upload Q2 earnings report.",
        ),
    )

    result = run_query(
        query="What was NVDA Q2 revenue?",
        company_id=None,
        period_start=None,
        period_end=None,
        settings=settings,
        driver=object(),
    )

    assert result["status"] == "missing_data"
    assert "upload" in result["missing_data_message"].lower()


def test_run_query_happy_path(monkeypatch: Any) -> None:
    settings = _settings()

    monkeypatch.setattr(
        "src.query.nodes.classify_query",
        lambda query, settings: QueryIntent(
            intent_type="fact_lookup",
            companies=["NVDA"],
            metrics=["revenue"],
            question_text=query,
        ),
    )
    monkeypatch.setattr(
        "src.query.nodes.check_coverage",
        lambda **kwargs: CoverageReport(
            is_complete=True,
            missing_sources=[],
            missing_message="",
        ),
    )
    monkeypatch.setattr("src.query.nodes.HybridRetriever", _FakeRetriever)
    monkeypatch.setattr(
        "src.query.nodes.expand_chunk_graph",
        lambda driver, chunk_ids, settings: [],
    )
    monkeypatch.setattr(
        "src.query.nodes.synthesize_answer",
        lambda query, ranked_evidence, settings: SynthesisOutput(
            answer="NVIDIA reported $26.0 billion revenue in Q2 FY2024 [1].",
            citations=[
                Citation(
                    chunk_id="c1",
                    doc_id="d1",
                    position=1,
                    raw_text="NVIDIA revenue in Q2 FY2024 was $26.0 billion.",
                    confidence=0.95,
                )
            ],
            confidence=0.9,
            reasoning="Direct value appears in evidence.",
        ),
    )
    monkeypatch.setattr(
        "src.query.nodes.validate_groundedness",
        lambda answer, citations, settings: type(
            "GroundednessStub",
            (),
            {"is_grounded": True, "unsupported_claims": [], "confidence": 0.92},
        )(),
    )

    result = run_query(
        query="What was NVDA Q2 revenue?",
        company_id=None,
        period_start="2024-04-01",
        period_end="2024-06-30",
        settings=settings,
        driver=object(),
    )

    assert result["status"] == "ok"
    assert result["is_grounded"] is True
    assert result["citations"][0]["chunk_id"] == "c1"
    assert "26.0" in result["answer"]


def test_graph_accepts_messages_only_input(monkeypatch: Any) -> None:
    settings = _settings()

    monkeypatch.setattr(
        "src.query.nodes.classify_query",
        lambda query, settings: QueryIntent(
            intent_type="fact_lookup",
            companies=["NVDA"],
            metrics=["revenue"],
            question_text=query,
        ),
    )
    monkeypatch.setattr(
        "src.query.nodes.check_coverage",
        lambda **kwargs: CoverageReport(
            is_complete=True,
            missing_sources=[],
            missing_message="",
        ),
    )
    monkeypatch.setattr("src.query.nodes.HybridRetriever", _FakeRetriever)
    monkeypatch.setattr(
        "src.query.nodes.expand_chunk_graph",
        lambda driver, chunk_ids, settings: [],
    )
    monkeypatch.setattr(
        "src.query.nodes.synthesize_answer",
        lambda query, ranked_evidence, settings: SynthesisOutput(
            answer="NVIDIA reported $26.0 billion revenue in Q2 FY2024 [1].",
            citations=[
                Citation(
                    chunk_id="c1",
                    doc_id="d1",
                    position=1,
                    raw_text="NVIDIA revenue in Q2 FY2024 was $26.0 billion.",
                    confidence=0.95,
                )
            ],
            confidence=0.9,
            reasoning="Direct value appears in evidence.",
        ),
    )
    monkeypatch.setattr(
        "src.query.nodes.validate_groundedness",
        lambda answer, citations, settings: type(
            "GroundednessStub",
            (),
            {"is_grounded": True, "unsupported_claims": [], "confidence": 0.92},
        )(),
    )

    graph = build_query_graph(settings, object())
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What was NVDA Q2 revenue?")],
            "company_id": None,
            "period_start": "2024-04-01",
            "period_end": "2024-06-30",
        }
    )

    final_response = result["final_response"]
    assert final_response["status"] == "ok"
    assert final_response["is_grounded"] is True
    assert final_response["citations"][0]["chunk_id"] == "c1"
    assert "26.0" in final_response["answer"]


# ---------------------------------------------------------------------------
# Regression tests that lock the behavior shapes that were broken in prod logs
# ---------------------------------------------------------------------------


def test_citation_marker_beyond_citation_list_is_flagged() -> None:
    """Answer referencing [16] when only 4 citations exist must be flagged.

    Before the fix this would silently pass groundedness with confidence=0.95.
    After the fix validate_groundedness must return is_grounded=False or
    confidence < 0.7 when marker indices exceed the citation count.
    """
    from src.query.answer_synthesizer import Citation
    from src.query.groundedness_validator import validate_groundedness

    citations = [
        Citation(chunk_id=f"c{i}", doc_id="d1", position=i, raw_text="Jensen Huang said accelerated computing.", confidence=0.9)
        for i in range(1, 5)  # only 4 citations
    ]
    # Answer references [16] which does not exist in the 4-item citation list
    answer = (
        "Jensen Huang described the shift to accelerated computing [1]. "
        "He emphasised that AI is a new computing model [16]."
    )
    check = validate_groundedness(answer, citations, settings=None)
    # Either the answer should be flagged as not grounded OR the confidence
    # should drop below 0.7 to signal the discrepancy.
    assert not check.is_grounded or check.confidence < 0.7, (
        f"Expected citation-drift to be flagged, got is_grounded={check.is_grounded} "
        f"confidence={check.confidence}"
    )


def test_citation_markers_renumbered_to_match_citation_list() -> None:
    """synthesize_answer post-processing must renumber markers to 1..N.

    When the LLM emits markers like [3][7][12] but the citation list only has
    3 items, the returned answer must contain only [1]..[3] after
    renumbering, and max marker index must equal len(citations).
    """
    import re

    from src.query.answer_synthesizer import Citation, SynthesisOutput, _renumber_citations

    raw_citations = [
        Citation(chunk_id=f"c{i}", doc_id="d1", position=i, raw_text="evidence text", confidence=0.9)
        for i in range(1, 4)  # 3 real citations
    ]
    # Simulate LLM emitting arbitrary (non-sequential) marker numbers
    raw_answer = "The company grew [3]. Accelerated computing [7] changed everything [12]."
    output = SynthesisOutput(answer=raw_answer, citations=raw_citations, confidence=0.9, reasoning="test")

    fixed = _renumber_citations(output)

    markers = list(map(int, re.findall(r"\[(\d+)\]", fixed.answer)))
    assert markers, "No markers found after renumbering"
    assert max(markers) <= len(fixed.citations), (
        f"Max marker {max(markers)} exceeds citation count {len(fixed.citations)}"
    )


def test_coverage_gate_open_query_still_checks_chunk_existence() -> None:
    """Coverage gate with company_id=None must still fail when DB is empty."""
    from unittest.mock import MagicMock

    from src.config import Settings
    from src.query.coverage_gate import check_coverage

    settings = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    # DB returns zero documents and chunks
    mock_session.run.return_value.single.return_value = {"doc_count": 0, "chunk_count": 0}

    report = check_coverage(
        driver=mock_driver,
        settings=settings,
        company_id=None,   # open-ended — no company filter
        period_start=None,
        period_end=None,
    )

    assert not report.is_complete, (
        "Coverage gate must report incomplete when doc_count=0, even with no company filter"
    )


def test_ranking_preserves_order_for_tied_scores() -> None:
    """Chunks with identical combined_score must be ordered by secondary key.

    Chunks that have a position metadata should rank above those without.
    """
    from src.config import Settings
    from src.query.evidence_ranker import rank_evidence

    settings = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )

    # Two chunks with identical combined_score — one has a position, one does not
    chunk_with_position = Document(
        page_content="Jensen Huang described the shift to accelerated computing.",
        metadata={"chunk_id": "pos", "doc_id": "d1", "combined_score": 0.9, "confidence": 0.9, "position": 3},
    )
    chunk_no_position = Document(
        page_content="Disclaimer: this document contains forward-looking statements.",
        metadata={"chunk_id": "nopos", "doc_id": "d1", "combined_score": 0.9, "confidence": 0.9},
    )

    ranked = rank_evidence(
        query_embedding=None,
        retrieved_chunks=[chunk_no_position, chunk_with_position],
        company_id=None,
        period_start=None,
        period_end=None,
        settings=settings,
    )

    assert ranked[0].metadata["chunk_id"] == "pos", (
        "Chunk with position metadata should rank above chunk without position when scores are tied"
    )
