from unittest.mock import MagicMock
from graph.schema import create_schema


def test_create_schema():
    mock_driver = MagicMock()
    mock_session = MagicMock()

    # Configure the mock driver to return the mock session when called as a context manager
    mock_driver.session.return_value.__enter__.return_value = mock_session

    create_schema(mock_driver, embedding_dim=1536)

    # Assert session was created
    mock_driver.session.assert_called_once()

    # Assert constraints and vector indexes were run
    assert mock_session.run.call_count >= 3

    # Check that the last run call had the right embedding_dim passed
    vector_index_calls = [
        call
        for call in mock_session.run.call_args_list
        if "CREATE VECTOR INDEX" in call[0][0]
    ]
    assert len(vector_index_calls) >= 2
    for call in vector_index_calls:
        assert call[1].get("embedding_dim") == 1536
