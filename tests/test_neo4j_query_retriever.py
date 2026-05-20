from unittest.mock import MagicMock, patch
from retrieval.neo4j_query_retriever import (
    retrieve_chunk_documents,
    expand_graph_context,
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


@patch("retrieval.neo4j_query_retriever._SEARCH_VECTOR_SUPPORTED", None)
def test_retrieve_chunk_documents_and_fallback():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Vector search mock results
    mock_result_vector = MagicMock()
    mock_result_vector.data.return_value = [
        {"node": {"id": "chunk_doc_1"}, "score": 0.95}
    ]

    # Detail search mock results
    mock_result_detail = MagicMock()
    mock_detail_record = {
        "chunk_id": "chunk_doc_1",
        "text": "AAPL Q4 revenues beat expectations",
        "page": 2,
        "section_type": "EARNINGS",
        "section_id": "sec_1",
        "section_title": "Earnings summary",
        "doc_id": "doc_1",
        "doc_type": "10-Q",
        "period": "Q4 FY25",
        "ticker": "AAPL",
        "company": "Apple Inc",
    }
    mock_result_detail.single.return_value = mock_detail_record

    # Define side effect for session.run based on query text
    def run_side_effect(query, *args, **kwargs):
        if "SEARCH INDEX" in query or "queryNodes" in query:
            return mock_result_vector
        else:
            return mock_result_detail

    mock_session.run.side_effect = run_side_effect

    embedding = [0.1] * 1536
    docs = retrieve_chunk_documents(
        driver=mock_driver,
        embedding=embedding,
        top_k=1,
        company_ticker="AAPL",
        time_filter="Q4 FY25",
    )

    assert len(docs) == 1
    assert docs[0]["page_content"] == "AAPL Q4 revenues beat expectations"
    assert docs[0]["metadata"]["ticker"] == "AAPL"
    assert docs[0]["metadata"]["score"] == 0.95

    # Filter mock call args to find vector search call
    called_queries = [call[0][0] for call in mock_session.run.call_args_list]
    assert any("SEARCH INDEX chunk_embeddings" in q for q in called_queries)


def test_expand_graph_context():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_result = MagicMock()
    mock_record = {
        "chunk_id": "c1",
        "section_id": "s1",
        "section_title": "Risk section",
        "doc_id": "d1",
        "doc_type": "10-K",
        "period": "FY24",
        "prev_chunks": ["c0"],
        "next_chunks": ["c2"],
        "entities": ["Apple", "Cook"],
        "causal_links": ["CAUSED:RevenueDecrease"],
    }
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    source_docs = [
        {
            "metadata": {
                "source_type": "chunk",
                "source_id": "c1",
            }
        }
    ]

    paths = expand_graph_context(
        driver=mock_driver,
        source_documents=source_docs,
        hop_depth=2,
        use_causal_edges=True,
    )

    assert len(paths) == 1
    assert paths[0]["chunk_id"] == "c1"
    assert paths[0]["entities"] == ["Apple", "Cook"]
    assert paths[0]["hop_depth"] == 2
