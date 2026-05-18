"""Tests for query understanding and classification."""

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.query.query_classifier import QueryIntent, classify_query


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )


def test_query_intent_model() -> None:
    """Test QueryIntent pydantic model."""
    intent = QueryIntent(
        intent_type="fact_lookup",
        companies=["AAPL"],
        metrics=["revenue"],
        question_text="What was Apple's revenue in Q3?",
    )

    assert intent.intent_type == "fact_lookup"
    assert "AAPL" in intent.companies
    assert "revenue" in intent.metrics


def test_query_intent_with_time_range() -> None:
    """Test QueryIntent with temporal information."""
    from src.query.query_classifier import DateRange
    from datetime import date

    intent = QueryIntent(
        intent_type="temporal_analysis",
        companies=["MSFT"],
        metrics=["eps"],
        time_range=DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31)),
        question_text="How did Microsoft's EPS change in 2023?",
    )

    assert intent.time_range is not None
    assert intent.time_range.start.isoformat() == "2023-01-01"
    assert intent.time_range.end.isoformat() == "2023-12-31"


def test_date_range_rejects_malformed_dates() -> None:
    """Malformed model output should be discarded before it reaches routing."""
    from src.query.query_classifier import DateRange

    with pytest.raises(ValidationError):
        DateRange(
            start="ca/Los_Angeles]/P3M/2024-07-28T00:00:00Z-07:00[America/Los_Angeles]",
            end="2024-07-28T00:00:00Z-07:00[America/Los_Angeles]/P3M",
        )
