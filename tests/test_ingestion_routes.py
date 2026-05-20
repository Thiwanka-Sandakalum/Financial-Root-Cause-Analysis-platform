import pytest
from fastapi.testclient import TestClient
from ingestion.api.main import create_app

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_start_ingestion(client, monkeypatch):
    # Mock create_job and background task
    async def mock_create_job(driver, document_ids):
        return "job123"
    
    client.app.dependency_overrides = {}
    from ingestion.api.db_helpers import create_job as orig_create_job
    import ingestion.api.db_helpers
    ingestion.api.db_helpers.create_job = mock_create_job

    payload = {
        "document_ids": ["doc1", "doc2"],
        "skip_if_exists": False,
        "force_reprocess": False
    }
    response = client.post("/api/v1/ingest/start", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "job123"
    assert data["status"] == "QUEUED"
    assert data["documents_queued"] == 2

    # Restore
    ingestion.api.db_helpers.create_job = orig_create_job

def test_list_ingestion_jobs(client, monkeypatch):
    # Mock list_jobs
    async def mock_list_jobs(driver, status=None, limit=50, offset=0):
        return [
            {"job_id": "job123", "status": "QUEUED", "created_at": "2024-05-20T00:00:00Z"}
        ]
    from ingestion.api.db_helpers import list_jobs as orig_list_jobs
    import ingestion.api.db_helpers
    ingestion.api.db_helpers.list_jobs = mock_list_jobs

    response = client.get("/api/v1/ingest/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)
    assert data["jobs"][0]["job_id"] == "job123"

    # Restore
    ingestion.api.db_helpers.list_jobs = orig_list_jobs
