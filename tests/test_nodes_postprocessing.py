from query.agent_query.nodes import (
    deterministic_citations_from_evidence,
    infer_ticker_from_evidence,
)


def test_infer_ticker_from_evidence_returns_majority_ticker():
    evidence = [
        {"ticker": "NVDA"},
        {"ticker": "nvda"},
        {"ticker": "AAPL"},
    ]

    inferred = infer_ticker_from_evidence(evidence)

    assert inferred == "NVDA"


def test_infer_ticker_from_evidence_requires_confident_majority():
    evidence = [
        {"ticker": "NVDA"},
        {"ticker": "AAPL"},
        {"ticker": "MSFT"},
    ]

    inferred = infer_ticker_from_evidence(evidence)

    assert inferred is None


def test_deterministic_citations_from_evidence_maps_fields_correctly():
    evidence = [
        {
            "source_type": "table",
            "source_id": "doc_sec_tbl_1",
            "doc_id": "doc_1",
            "section_id": "sec_1",
            "section_title": "Revenue table",
            "page": None,
            "score": 0.91,
        },
        {
            "source_type": "chunk",
            "source_id": "doc_chunk_4",
            "doc_id": "doc_1",
            "section_id": "sec_2",
            "doc_type": "8-K",
            "page": 16,
            "score": 0.88,
        },
    ]

    citations = deterministic_citations_from_evidence(evidence)

    assert citations == [
        {
            "source_type": "table",
            "source_id": "doc_sec_tbl_1",
            "doc_id": "doc_1",
            "section_id": "sec_1",
            "page": None,
            "score": 0.91,
            "title": "Revenue table",
        },
        {
            "source_type": "chunk",
            "source_id": "doc_chunk_4",
            "doc_id": "doc_1",
            "section_id": "sec_2",
            "page": 16,
            "score": 0.88,
            "title": "8-K",
        },
    ]


def test_deterministic_citations_from_evidence_dedupes_and_limits():
    evidence = [
        {"source_type": "table", "source_id": "t1"},
        {"source_type": "table", "source_id": "t1"},
        {"source_type": "chunk", "source_id": "c1"},
    ]

    citations = deterministic_citations_from_evidence(evidence, max_citations=1)

    assert len(citations) == 1
    assert citations[0]["source_id"] == "t1"
