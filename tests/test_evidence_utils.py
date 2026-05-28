from pydantic import BaseModel
from query.agent_query.evidence_utils import (
    state_dict,
    serialize_docs,
    serialize_graph_paths,
    dedupe_documents,
    evidence_item,
)


class DummyModel(BaseModel):
    name: str
    value: int


def test_state_dict():
    # Test with pydantic object
    model = DummyModel(name="test", value=42)
    assert state_dict(model) == {"name": "test", "value": 42}

    # Test with normal dict
    d = {"key": "val"}
    assert state_dict(d) == d

    # Test with invalid object
    assert state_dict("not-a-dict") == {}


def test_serialize_docs():
    docs = [
        {
            "page_content": "This is content 1",
            "metadata": {
                "source_type": "chunk",
                "doc_type": "10-Q",
                "period": "Q3 FY25",
                "section_type": "MD&A",
                "score": 0.85,
            },
        },
        {
            "page_content": "This is content 2",
            "metadata": {
                "source_type": "table",
                "doc_type": "10-K",
                "period": "FY24",
                "section_type": "FINANCIALS",
                "score": 0.92,
            },
        },
    ]
    serialized = serialize_docs(docs)
    assert "[chunk | 10-Q | Q3 FY25 | MD&A | score=0.85]" in serialized
    assert "This is content 1" in serialized
    assert "[table | 10-K | FY24 | FINANCIALS | score=0.92]" in serialized
    assert "This is content 2" in serialized


def test_serialize_graph_paths():
    paths = [
        {"node": "CompanyA", "relationship": "FILED", "target": "Doc1"},
        {"node": "Doc1", "relationship": "CONTAINS", "target": "Sec1"},
    ]
    serialized = serialize_graph_paths(paths)
    assert "{'node': 'CompanyA'" in serialized
    assert "{'node': 'Doc1'" in serialized


def test_dedupe_documents():
    docs = [
        {
            "metadata": {"source_type": "chunk", "source_id": "chunk_1", "score": 0.8},
        },
        {
            "metadata": {
                "source_type": "chunk",
                "source_id": "chunk_1",
                "score": 0.9,
            },  # Duplicate, higher score (though keep first seen order or sort?)
        },
        {
            "metadata": {"source_type": "table", "source_id": "table_1", "score": 0.5},
        },
        {
            "metadata": {"source_type": "chunk", "source_id": "chunk_2", "score": 0.95},
        },
    ]

    deduped = dedupe_documents(docs)
    # Dedupe uses (source_type, source_id)
    # First seen of chunk_1 is kept (score 0.8).
    # Then table_1 (0.5), chunk_2 (0.95).
    # Then sorted by score desc.
    # Scores: chunk_2 (0.95), chunk_1 (0.8), table_1 (0.5).
    assert len(deduped) == 3
    assert deduped[0]["metadata"]["source_id"] == "chunk_2"
    assert deduped[1]["metadata"]["source_id"] == "chunk_1"
    assert deduped[1]["metadata"]["score"] == 0.8  # First seen chunk_1 is kept
    assert deduped[2]["metadata"]["source_id"] == "table_1"


def test_evidence_item():
    doc = {
        "page_content": "Raw content text",
        "metadata": {
            "source_type": "chunk",
            "source_id": "c1",
            "doc_id": "d1",
            "doc_type": "10-Q",
            "period": "Q1 FY25",
            "section_id": "s1",
            "section_type": "MD&A",
            "page": 4,
            "score": 0.88,
            "ticker": "AAPL",
        },
    }
    item = evidence_item(doc)
    assert item == {
        "source_type": "chunk",
        "source_id": "c1",
        "doc_id": "d1",
        "doc_type": "10-Q",
        "period": "Q1 FY25",
        "section_id": "s1",
        "section_type": "MD&A",
        "page": 4,
        "score": 0.88,
        "ticker": "AAPL",
        "text": "Raw content text",
    }


