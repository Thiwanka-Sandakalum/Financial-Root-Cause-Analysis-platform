"""Tests for coverage gate and data availability checks."""

import pytest

from src.config import Settings
from src.query.coverage_gate import CoverageReport, check_coverage


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


def test_coverage_report_model() -> None:
    """Test CoverageReport dataclass."""
    report = CoverageReport(
        is_complete=True,
        missing_sources=[],
        missing_message="",
    )

    assert report.is_complete is True
    assert len(report.missing_sources) == 0
    assert report.missing_message == ""


def test_coverage_report_incomplete() -> None:
    """Test incomplete coverage report."""
    report = CoverageReport(
        is_complete=False,
        missing_sources=["Q4 earnings report"],
        missing_message="Please upload Q4 earnings report for AAPL",
    )

    assert report.is_complete is False
    assert len(report.missing_sources) > 0
    assert "Q4" in report.missing_message
