from pathlib import Path

from src.config import Settings
from src.ingestion.service import IngestionRequest, IngestionUploadRequest, submit_ingestion


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )


def test_submit_ingestion_normalizes_uploads_and_closes_owned_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_path = tmp_path / "report.txt"
    file_path.write_text("revenue up", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_client = _FakeClient()

    def fake_client_factory(settings: Settings) -> _FakeClient:
        captured["settings"] = settings
        return fake_client

    def fake_run_ingestion(*, uploads, settings, neo4j_client):
        captured["uploads"] = uploads
        captured["run_settings"] = settings
        captured["neo4j_client"] = neo4j_client
        return {"status": "completed", "processed_documents": 1}

    monkeypatch.setattr("src.ingestion.service.Neo4jClient", fake_client_factory)
    monkeypatch.setattr("src.ingestion.service.run_ingestion", fake_run_ingestion)

    result = submit_ingestion(
        IngestionRequest(
            uploads=[
                IngestionUploadRequest(
                    file_path=str(file_path),
                    company_id="AAPL",
                    source_type="annual_report",
                    period_start="2024-01-01",
                    period_end="2024-12-31",
                    upload_id="upload_1",
                )
            ]
        ),
        settings=_settings(),
    )

    assert result["status"] == "completed"
    assert captured["uploads"] == [
        {
            "file_path": str(file_path.resolve()),
            "company_id": "AAPL",
            "source_type": "annual_report",
            "auto_detect": False,
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "upload_id": "upload_1",
        }
    ]
    assert captured["neo4j_client"] is fake_client
    assert fake_client.closed is True


def test_submit_ingestion_reuses_supplied_client(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "report.txt"
    file_path.write_text("revenue up", encoding="utf-8")

    fake_client = _FakeClient()

    def fake_run_ingestion(*, uploads, settings, neo4j_client):
        return {
            "status": "completed",
            "processed_documents": len(uploads),
            "client_reused": neo4j_client is fake_client,
        }

    monkeypatch.setattr("src.ingestion.service.run_ingestion", fake_run_ingestion)

    result = submit_ingestion(
        IngestionRequest(
            uploads=[
                IngestionUploadRequest(
                    file_path=str(file_path),
                    company_id="AAPL",
                )
            ]
        ),
        settings=_settings(),
        neo4j_client=fake_client,
    )

    assert result["status"] == "completed"
    assert result["client_reused"] is True
    assert fake_client.closed is False