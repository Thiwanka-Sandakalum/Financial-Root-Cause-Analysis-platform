"""Neo4j database helpers for job and document management."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver

from ingestion.api.models import JobErrorResponse, JobResponse, JobStatus, IngestionStage


async def create_job(
    driver: AsyncDriver,
    document_ids: List[str],
) -> str:
    """
    Create a new ingestion job in Neo4j.
    
    Args:
        driver: Neo4j driver
        document_ids: List of document IDs to ingest
        
    Returns:
        Job ID
    """
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    query = """
    CREATE (j:Job {
        id: $job_id,
        status: 'QUEUED',
        progress: 0,
        created_at: $created_at,
        documents_count: $doc_count
    })
    RETURN j.id as job_id
    """

    async with driver.session() as session:
        result = await session.run(
            query,
            job_id=job_id,
            created_at=now,
            doc_count=len(document_ids),
        )
        await result.single()

    return job_id


async def get_job(driver: AsyncDriver, job_id: str) -> Optional[JobResponse]:
    """
    Get job status from Neo4j.
    
    Args:
        driver: Neo4j driver
        job_id: Job ID
        
    Returns:
        JobResponse or None if not found
    """
    query = """
    MATCH (j:Job {id: $job_id})
    RETURN {
        id: j.id,
        status: j.status,
        stage: j.stage,
        progress: j.progress,
        message: j.message,
        created_at: j.created_at,
        started_at: j.started_at,
        estimated_completion: j.estimated_completion,
        stats: j.stats
    } as job_data
    """

    async with driver.session() as session:
        result = await session.run(query, job_id=job_id)
        record = await result.single()

    if not record:
        return None

    job_data = record.get("job_data")
    
    # Parse timestamps
    created_at = datetime.fromisoformat(job_data.get("created_at", datetime.utcnow().isoformat()))
    started_at = None
    if job_data.get("started_at"):
        started_at = datetime.fromisoformat(job_data["started_at"])
    estimated_completion = None
    if job_data.get("estimated_completion"):
        estimated_completion = datetime.fromisoformat(job_data["estimated_completion"])

    return JobResponse(
        id=job_data.get("id"),
        status=JobStatus(job_data.get("status", "QUEUED")),
        stage=IngestionStage(job_data.get("stage")) if job_data.get("stage") else None,
        progress=int(job_data.get("progress", 0)),
        message=job_data.get("message"),
        created_at=created_at,
        started_at=started_at,
        estimated_completion=estimated_completion,
        stats=job_data.get("stats", {}),
        errors=job_data.get("errors", []),
    )


async def update_job_status(
    driver: AsyncDriver,
    job_id: str,
    status: str,
    stage: Optional[str] = None,
    progress: int = 0,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update job status in Neo4j.
    
    Args:
        driver: Neo4j driver
        job_id: Job ID
        status: Job status
        stage: Current ingestion stage
        progress: Progress percentage (0-100)
        message: Status message
        error_message: Error message if failed
    """
    now = datetime.utcnow().isoformat()
    
    update_dict = {
        "status": status,
        "progress": progress,
        "updated_at": now,
    }
    
    if stage:
        update_dict["stage"] = stage
    if message:
        update_dict["message"] = message
    if status == "RUNNING" and not update_dict.get("started_at"):
        update_dict["started_at"] = now
    if error_message:
        update_dict["error_message"] = error_message

    query = """
    MATCH (j:Job {id: $job_id})
    SET j += $updates
    RETURN j.id
    """

    async with driver.session() as session:
        await session.run(query, job_id=job_id, updates=update_dict)


async def list_jobs(
    driver: AsyncDriver,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, List[JobResponse]]:
    """
    List ingestion jobs from Neo4j.
    
    Args:
        driver: Neo4j driver
        status: Filter by status (optional)
        limit: Limit results
        offset: Offset results
        
    Returns:
        Tuple of (total_count, jobs)
    """
    where_clause = "WHERE j:Job"
    if status:
        where_clause += f" AND j.status = '{status}'"

    count_query = f"{where_clause} RETURN count(j) as total"
    list_query = f"""
    {where_clause}
    RETURN {{
        id: j.id,
        status: j.status,
        stage: j.stage,
        progress: j.progress,
        message: j.message,
        created_at: j.created_at,
        started_at: j.started_at,
        estimated_completion: j.estimated_completion,
        stats: j.stats
    }} as job_data
    ORDER BY j.created_at DESC
    SKIP {offset}
    LIMIT {limit}
    """

    async with driver.session() as session:
        count_result = await session.run(count_query)
        count_record = await count_result.single()
        total = count_record.get("total", 0) if count_record else 0

        list_result = await session.run(list_query)
        records = await list_result.all()

    jobs = []
    for record in records:
        job_data = record.get("job_data")
        created_at = datetime.fromisoformat(job_data.get("created_at", datetime.utcnow().isoformat()))
        started_at = None
        if job_data.get("started_at"):
            started_at = datetime.fromisoformat(job_data["started_at"])

        jobs.append(
            JobResponse(
                id=job_data.get("id"),
                status=JobStatus(job_data.get("status", "QUEUED")),
                stage=IngestionStage(job_data.get("stage")) if job_data.get("stage") else None,
                progress=int(job_data.get("progress", 0)),
                message=job_data.get("message"),
                created_at=created_at,
                started_at=started_at,
                stats=job_data.get("stats", {}),
            )
        )

    return total, jobs


async def create_document(
    driver: AsyncDriver,
    doc_id: str,
    filename: str,
    ticker: str,
    company_name: str,
    doc_type: str,
    period: str,
) -> None:
    """
    Create a document node in Neo4j.
    
    Args:
        driver: Neo4j driver
        doc_id: Document ID
        filename: File name
        ticker: Company ticker
        company_name: Company name
        doc_type: Document type
        period: Fiscal period
    """
    now = datetime.utcnow().isoformat()

    query = """
    CREATE (d:Document {
        id: $doc_id,
        filename: $filename,
        ticker: $ticker,
        company_name: $company_name,
        doc_type: $doc_type,
        period: $period,
        status: 'UPLOADED',
        created_at: $created_at
    })
    RETURN d.id
    """

    async with driver.session() as session:
        await session.run(
            query,
            doc_id=doc_id,
            filename=filename,
            ticker=ticker,
            company_name=company_name,
            doc_type=doc_type,
            period=period,
            created_at=now,
        )
