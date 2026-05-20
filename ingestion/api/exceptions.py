"""Custom exceptions for the ingestion API."""
from datetime import datetime

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse


class IngestError(HTTPException):
    """Base ingestion error."""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 400,
    ):
        self.message = message
        self.code = code
        super().__init__(status_code=status_code, detail=message)


class DocumentNotFoundError(IngestError):
    """Document not found error."""
    
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document '{doc_id}' not found",
            code="DOCUMENT_NOT_FOUND",
            status_code=404,
        )


class JobNotFoundError(IngestError):
    """Job not found error."""
    
    def __init__(self, job_id: str):
        super().__init__(
            message=f"Job '{job_id}' not found",
            code="JOB_NOT_FOUND",
            status_code=404,
        )


class InvalidJobStateError(IngestError):
    """Invalid job state error."""
    
    def __init__(self, job_id: str, current_status: str):
        super().__init__(
            message=f"Cannot transition job '{job_id}' from '{current_status}' state",
            code="INVALID_JOB_STATE",
            status_code=409,
        )


async def ingest_error_handler(request: Request, exc: IngestError) -> JSONResponse:
    """Handle ingestion errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        },
    )
