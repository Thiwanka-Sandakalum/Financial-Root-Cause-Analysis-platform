from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import ingestion.api.routes.documents as documents_routes


@pytest.mark.asyncio
async def test_persist_upload_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(documents_routes, "_UPLOAD_DIR", tmp_path)

    payload = b"sample upload payload"
    upload = UploadFile(filename="report.txt", file=BytesIO(payload))

    file_path, size_bytes = await documents_routes._persist_upload(upload, "doc-123")

    saved_path = Path(file_path)
    assert saved_path.exists()
    assert saved_path.name == "doc-123.txt"
    assert saved_path.read_bytes() == payload
    assert size_bytes == len(payload)


@pytest.mark.asyncio
async def test_upload_document_requires_filename():
    upload = UploadFile(filename=None, file=BytesIO(b"x"))

    with pytest.raises(HTTPException) as exc_info:
        await documents_routes.upload_document(
            file=upload,
            company_ticker="ADS",
            company_name="Adidas AG",
            doc_type="Annual Report",
            fiscal_period="2025",
            auto_ingest=False,
            driver=None,
        )

    assert exc_info.value.status_code == 400
    assert "filename" in str(exc_info.value.detail).lower()
