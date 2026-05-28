from types import SimpleNamespace
from unittest.mock import patch
from query.core.neo4j_query_retriever import (
    retrieve_chunk_documents,
    retrieve_table_documents,
    _normalize_text,
    _retriever_doc,
)


def test_normalize_text():
    assert _normalize_text(" AAPL  ") == "AAPL"
    assert _normalize_text(None) == ""


def test_retriever_doc():
    doc = _retriever_doc("chunk", "c1", "content text", {"ticker": "AAPL"})
    assert doc == {
        "page_content": "content text",
        "type": "Document",
        "metadata": {
            "source_type": "chunk",
            "source_id": "c1",
            "ticker": "AAPL",
        },
    }


class _FakeRetriever:
    def __init__(self, items):
        self._items = items

    def search(self, **kwargs):
        return SimpleNamespace(items=self._items, metadata={})


def test_retrieve_chunk_documents_with_prebuilt_retriever():
    fake_items = [
        SimpleNamespace(
            content="AAPL Q4 revenues beat expectations",
            metadata={
                "source_id": "chunk_doc_1",
                "page": 2,
                "section_type": "EARNINGS",
                "section_id": "sec_1",
                "section_title": "Earnings summary",
                "doc_id": "doc_1",
                "doc_type": "10-Q",
                "period": "Q4 FY25",
                "ticker": "AAPL",
                "company": "Apple Inc",
                "score": 0.95,
            },
        )
    ]

    embedding = [0.1] * 1536
    with patch(
        "query.core.neo4j_query_retriever._build_chunk_retriever",
        return_value=_FakeRetriever(fake_items),
    ):
        docs = retrieve_chunk_documents(
            driver=object(),
            embedding=embedding,
            top_k=1,
            company_ticker="AAPL",
            time_filter="Q4 FY25",
            use_causal_edges=True,
        )

    assert len(docs) == 1
    assert docs[0]["page_content"] == "AAPL Q4 revenues beat expectations"
    assert docs[0]["metadata"]["ticker"] == "AAPL"
    assert docs[0]["metadata"]["score"] == 0.95


def test_retrieve_table_documents_with_prebuilt_retriever():
    fake_items = [
        SimpleNamespace(
            content="Revenue by quarter",
            metadata={
                "source_id": "table_doc_1",
                "sequence": 3,
                "section_type": "TABLE",
                "section_id": "sec_2",
                "section_title": "Revenue table",
                "doc_id": "doc_1",
                "doc_type": "10-Q",
                "period": "Q4 FY25",
                "ticker": "AAPL",
                "company": "Apple Inc",
                "score": 0.88,
            },
        )
    ]

    embedding = [0.1] * 1536
    with patch(
        "query.core.neo4j_query_retriever._build_table_retriever",
        return_value=_FakeRetriever(fake_items),
    ):
        docs = retrieve_table_documents(
            driver=object(),
            embedding=embedding,
            top_k=1,
            company_ticker="AAPL",
            time_filter="Q4 FY25",
        )

    assert len(docs) == 1
    assert docs[0]["page_content"] == "Revenue by quarter"
    assert docs[0]["metadata"]["source_id"] == "table_doc_1"

