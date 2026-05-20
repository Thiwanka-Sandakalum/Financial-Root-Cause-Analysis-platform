"""Pydantic request/response models for the ingestion API."""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document ingestion status."""
    UPLOADED = "UPLOADED"
    INGESTING = "INGESTING"
    INGESTED = "INGESTED"
    FAILED = "FAILED"


class JobStatus(str, Enum):
    """Ingestion job status."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class IngestionStage(str, Enum):
    """Ingestion pipeline stages."""
    UPLOADING = "UPLOADING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    EXTRACTION = "EXTRACTION"
    WRITING = "WRITING"


# ─────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────


class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    
    company_ticker: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1, max_length=255)
    doc_type: str = Field(default="Annual Report")
    fiscal_period: str = Field(default="2025")
    auto_ingest: bool = Field(default=False)


class IngestionRequest(BaseModel):
    """Start ingestion request."""
    
    document_ids: List[str] = Field(..., min_items=1)
    skip_if_exists: bool = Field(default=True)
    force_reprocess: bool = Field(default=False)


# ─────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    """Document response."""
    
    id: str
    filename: str
    ticker: str
    company_name: str
    doc_type: str
    period: str
    created_at: datetime
    status: DocumentStatus
    stats: Dict = Field(default_factory=dict)


class JobErrorResponse(BaseModel):
    """Job error response."""
    
    stage: str
    message: str
    severity: str  # "error" | "warning"
    timestamp: datetime


class JobResponse(BaseModel):
    """Ingestion job response."""
    
    id: str
    status: JobStatus
    stage: Optional[IngestionStage] = None
    progress: int = Field(ge=0, le=100)
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    stats: Dict = Field(default_factory=dict)
    errors: List[JobErrorResponse] = Field(default_factory=list)


class JobStartResponse(BaseModel):
    """Job start response."""
    
    job_id: str
    status: JobStatus
    created_at: datetime
    documents_queued: int


class CancelJobResponse(BaseModel):
    """Cancel job response."""
    
    id: str
    status: JobStatus
    message: str


# ─────────────────────────────────────────────────────────────────────────
# List Response Models
# ─────────────────────────────────────────────────────────────────────────


class PaginatedJobResponse(BaseModel):
    """Paginated job response."""
    
    total: int
    limit: int
    offset: int
    jobs: List[JobResponse]


class ListDocumentsResponse(BaseModel):
    """List documents response."""
    
    total: int
    limit: int
    offset: int
    documents: List[DocumentResponse]
