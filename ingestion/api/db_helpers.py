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

    stats = job_data.get("stats") or {}
    errors = job_data.get("errors") or []

    return JobResponse(
        id=job_data.get("id"),
        status=JobStatus(job_data.get("status", "QUEUED")),
        stage=IngestionStage(job_data.get("stage")) if job_data.get("stage") else None,
        progress=int(job_data.get("progress", 0)),
        message=job_data.get("message"),
        created_at=created_at,
        started_at=started_at,
        estimated_completion=estimated_completion,
        stats=stats,
        errors=errors,
    )


async def update_job_status(
    driver: AsyncDriver,
    job_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: int = 0,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
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
        "progress": progress,
        "updated_at": now,
    }

    if status is not None:
        update_dict["status"] = status
    
    if stage:
        update_dict["stage"] = stage
    if message:
        update_dict["message"] = message
    if status == "RUNNING" and not update_dict.get("started_at"):
        update_dict["started_at"] = now
    if error_message:
        update_dict["error_message"] = error_message
    if stats is not None:
        update_dict["stats"] = stats

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
    match_clause = "MATCH (j:Job)"
    where_clause = ""
    params = {}
    if status:
        where_clause = " WHERE j.status = $status"
        params["status"] = status

    count_query = f"{match_clause}{where_clause} RETURN count(j) as total"
    list_query = f"""
    {match_clause}{where_clause}
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
    SKIP $offset
    LIMIT $limit
    """
    params["offset"] = offset
    params["limit"] = limit

    async with driver.session() as session:
        count_result = await session.run(count_query, params)
        count_record = await count_result.single()
        total = count_record.get("total", 0) if count_record else 0

        list_result = await session.run(list_query, params)
        records = await list_result.data()

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
                stats=job_data.get("stats") or {},
            )
        )

    return total, jobs


async def create_document(
    driver: AsyncDriver,
    doc_id: str,
    filename: str,
    file_path: str,
    mime_type: str,
    size_bytes: int,
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
        file_path: Persisted absolute path for the uploaded file
        mime_type: Uploaded file MIME type
        size_bytes: Uploaded file size in bytes
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
        file_path: $file_path,
        mime_type: $mime_type,
        size_bytes: $size_bytes,
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
            file_path=file_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            ticker=ticker,
            company_name=company_name,
            doc_type=doc_type,
            period=period,
            created_at=now,
        )


async def update_document_status(
    driver: AsyncDriver,
    doc_id: str,
    status: str,
    stats: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update document processing status and optional stats/error details."""
    updates: Dict[str, Any] = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if stats is not None:
        updates["stats"] = stats
    if error_message:
        updates["error_message"] = error_message

    query = """
    MATCH (d:Document {id: $doc_id})
    SET d += $updates
    RETURN d.id
    """

    async with driver.session() as session:
        await session.run(query, doc_id=doc_id, updates=updates)


async def document_has_graph_data(driver: AsyncDriver, doc_id: str) -> bool:
    """Return whether a document already has generated section/chunk graph data."""
    query = """
    MATCH (d:Document {id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section)
    OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
    RETURN count(DISTINCT s) AS sections, count(DISTINCT c) AS chunks
    """

    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id)
        record = await result.single()

    if not record:
        return False
    sections = int(record.get("sections", 0) or 0)
    chunks = int(record.get("chunks", 0) or 0)
    return sections > 0 or chunks > 0


async def get_document_graph_stats(driver: AsyncDriver, doc_id: str) -> Dict[str, int]:
    """Return section/chunk/table counts currently attached to a document."""
    query = """
    MATCH (d:Document {id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section)
    OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (s)-[:HAS_TABLE]->(t:Table)
    RETURN count(DISTINCT s) AS sections,
           count(DISTINCT c) AS chunks,
           count(DISTINCT t) AS tables
    """

    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id)
        record = await result.single()

    if not record:
        return {"sections": 0, "chunks": 0, "tables": 0}

    return {
        "sections": int(record.get("sections", 0) or 0),
        "chunks": int(record.get("chunks", 0) or 0),
        "tables": int(record.get("tables", 0) or 0),
    }


async def get_documents_for_ingestion(
    driver: AsyncDriver,
    document_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Fetch document records keyed by document ID for ingestion validation."""
    if not document_ids:
        return {}

    query = """
    MATCH (d:Document)
    WHERE d.id IN $document_ids
    RETURN {
        id: d.id,
        filename: d.filename,
        file_path: d.file_path,
        ticker: d.ticker,
        company_name: d.company_name,
        doc_type: d.doc_type,
        period: d.period,
        status: d.status
    } AS doc_data
    """

    async with driver.session() as session:
        result = await session.run(query, document_ids=document_ids)
        records = await result.data()

    documents: Dict[str, Dict[str, Any]] = {}
    for record in records:
        doc_data = record.get("doc_data") or {}
        doc_id = doc_data.get("id")
        if doc_id:
            documents[doc_id] = doc_data
    return documents
