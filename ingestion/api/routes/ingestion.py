"""Ingestion pipeline routes."""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from neo4j import AsyncDriver

from ingestion.api.db_helpers import (
    create_job,
    get_job,
    list_jobs,
    update_job_status,
)
from ingestion.api.dependencies import get_neo4j_driver, get_llm, get_embedder
from ingestion.api.models import (
    IngestionRequest,
    JobResponse,
    JobStartResponse,
    JobStatus,
    PaginatedJobResponse,
    CancelJobResponse,
)
from ingestion.api.exceptions import JobNotFoundError
from ingestion.api.tasks import ingest_pipeline_task

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("/start", response_model=JobStartResponse, status_code=202)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    driver: AsyncDriver = Depends(get_neo4j_driver),
    llm = Depends(get_llm),
    embedder = Depends(get_embedder),
) -> JobStartResponse:
    """
    Start a new ingestion job.
    
    Args:
        request: Ingestion request with document IDs
        background_tasks: FastAPI background tasks
        driver: Neo4j driver
        llm: LLM client
        embedder: Embeddings client
        
    Returns:
        Job creation confirmation
    """
    job_id = await create_job(
        driver=driver,
        document_ids=request.document_ids,
    )

    # Queue background task
    background_tasks.add_task(
        ingest_pipeline_task,
        job_id=job_id,
        document_ids=request.document_ids,
        driver=driver,
        llm=llm,
        embedder=embedder,
        skip_if_exists=request.skip_if_exists,
        force_reprocess=request.force_reprocess,
    )

    return JobStartResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
        documents_queued=len(request.document_ids),
    )


@router.get("/jobs", response_model=PaginatedJobResponse)
async def list_ingestion_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> PaginatedJobResponse:
    """
    List ingestion jobs with optional filters.
    
    Args:
        status: Filter by job status (QUEUED, RUNNING, COMPLETED, FAILED)
        limit: Result limit (default: 50)
        offset: Result offset (default: 0)
        driver: Neo4j driver
        
    Returns:
        Paginated list of jobs
    """
    total, jobs = await list_jobs(
        driver=driver,
        status=status,
        limit=limit,
        offset=offset,
    )

    return PaginatedJobResponse(
        total=total,
        limit=limit,
        offset=offset,
        jobs=jobs,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_ingestion_job(
    job_id: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> JobResponse:
    """
    Get ingestion job status.
    
    Args:
        job_id: Job ID
        driver: Neo4j driver
        
    Returns:
        Job status and details
        
    Raises:
        JobNotFoundError: If job not found
    """
    job = await get_job(driver, job_id)

    if not job:
        raise JobNotFoundError(job_id)

    return job


@router.post("/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_ingestion_job(
    job_id: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> CancelJobResponse:
    """
    Cancel an ingestion job.
    
    Args:
        job_id: Job ID
        driver: Neo4j driver
        
    Returns:
        Cancel confirmation
        
    Raises:
        JobNotFoundError: If job not found
    """
    job = await get_job(driver, job_id)

    if not job:
        raise JobNotFoundError(job_id)

    # Update job status
    await update_job_status(
        driver=driver,
        job_id=job_id,
        status=JobStatus.CANCELLED.value,
        message="Job cancelled by user",
    )

    return CancelJobResponse(
        id=job_id,
        status=JobStatus.CANCELLED,
        message="Job cancelled successfully",
    )
