from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.chunker import ChunkingError, chunk_document
from src.ingestion.models import DocumentInput, ParsedDocument
from src.ingestion.parser import DocumentParseError, UnsupportedDocumentError, parse_document


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


def test_parse_text_document(tmp_path: Path, settings: Settings) -> None:
    file_path = tmp_path / "earnings.txt"
    file_path.write_text("Revenue increased\n\nOperating margin improved\n", encoding="utf-8")

    parsed = parse_document(
        DocumentInput(
            file_path=str(file_path),
            company_id="AAPL",
            source_type="earnings_transcript",
        ),
        settings,
    )

    assert parsed.company_id == "AAPL"
    assert parsed.source_type == "earnings_transcript"
    assert parsed.text == "Revenue increased\nOperating margin improved"
    assert parsed.metadata["file_type"] == "text"


def test_parse_rejects_unsupported_extension(tmp_path: Path, settings: Settings) -> None:
    file_path = tmp_path / "earnings.csv"
    file_path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        parse_document(
            DocumentInput(
                file_path=str(file_path),
                company_id="AAPL",
                source_type="spreadsheet",
            ),
            settings,
        )


def test_parse_rejects_empty_text_file(tmp_path: Path, settings: Settings) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(DocumentParseError):
        parse_document(
            DocumentInput(
                file_path=str(file_path),
                company_id="AAPL",
                source_type="notes",
            ),
            settings,
        )


def test_chunk_document_preserves_metadata(settings: Settings) -> None:
    parsed = ParsedDocument(
        doc_id="doc_123",
        company_id="AAPL",
        source_type="annual_report",
        file_path="/tmp/report.txt",
        title="Annual Report",
        text=" ".join(f"word{i}" for i in range(2500)),
        metadata={"period_start": "2024-01-01", "period_end": "2024-12-31"},
    )

    chunks = chunk_document(parsed, settings)

    assert len(chunks) >= 2
    assert chunks[0].metadata["period_start"] == "2024-01-01"
    assert chunks[0].metadata["title"] == "Annual Report"
    assert chunks[0].position == 0
    assert chunks[1].metadata["chunk_start_token"] < chunks[0].metadata["chunk_end_token"]


def test_chunk_document_rejects_empty_content(settings: Settings) -> None:
    parsed = ParsedDocument(
        doc_id="doc_empty",
        company_id="AAPL",
        source_type="annual_report",
        file_path="/tmp/empty.txt",
        title="Empty",
        text="",
        metadata={},
    )

    with pytest.raises(ChunkingError):
        chunk_document(parsed, settings)
