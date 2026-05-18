"""Tests for evidence ranking utilities."""

from langchain_core.documents import Document

from src.config import Settings
from src.query.evidence_ranker import rank_evidence


def _settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )


def test_rank_evidence_orders_by_weighted_score() -> None:
    settings = _settings()

    chunks = [
        Document(
            page_content="A",
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "company_id": "NVDA",
                "combined_score": 0.9,
                "confidence": 0.8,
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
            },
        ),
        Document(
            page_content="B",
            metadata={
                "chunk_id": "c2",
                "doc_id": "d2",
                "company_id": "NVDA",
                "combined_score": 0.4,
                "confidence": 0.95,
                "period_start": "2022-01-01",
                "period_end": "2022-03-31",
            },
        ),
    ]

    ranked = rank_evidence(
        query_embedding=None,
        retrieved_chunks=chunks,
        company_id="NVDA",
        period_start="2024-01-01",
        period_end="2024-03-31",
        settings=settings,
    )

    assert len(ranked) == 2
    assert ranked[0].metadata["chunk_id"] == "c1"
    assert ranked[0].metadata["final_score"] > ranked[1].metadata["final_score"]


def test_rank_evidence_applies_top_k_limit() -> None:
    settings = _settings()
    settings.query_evidence_top_k = 1

    chunks = [
        Document(page_content="A", metadata={"chunk_id": "c1", "doc_id": "d1", "combined_score": 0.8}),
        Document(page_content="B", metadata={"chunk_id": "c2", "doc_id": "d2", "combined_score": 0.7}),
    ]

    ranked = rank_evidence(
        query_embedding=None,
        retrieved_chunks=chunks,
        company_id=None,
        period_start=None,
        period_end=None,
        settings=settings,
    )

    assert len(ranked) == 1
    assert ranked[0].metadata["chunk_id"] == "c1"
