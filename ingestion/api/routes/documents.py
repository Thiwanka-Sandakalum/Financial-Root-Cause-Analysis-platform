"""Document management routes."""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from neo4j import AsyncDriver

from ingestion.api.db_helpers import create_document
from ingestion.api.dependencies import get_neo4j_driver
from ingestion.api.models import DocumentUploadRequest, DocumentResponse, ListDocumentsResponse
from ingestion.api.exceptions import DocumentNotFoundError

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=dict, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    company_ticker: str = Form(...),
    company_name: str = Form(...),
    doc_type: str = Form(default="Annual Report"),
    fiscal_period: str = Form(default="2025"),
    auto_ingest: bool = Form(default=False),
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> dict:
    """
    Upload a new document.
    
    Args:
        file: PDF or TXT file to upload
        company_ticker: Company ticker symbol
        company_name: Company name
        doc_type: Document type (default: Annual Report)
        fiscal_period: Fiscal period (default: 2025)
        auto_ingest: Auto-start ingestion (default: False)
        driver: Neo4j driver
        
    Returns:
        Document upload confirmation with optional job ID
    """
    doc_id = str(uuid.uuid4())
    
    # Create document in Neo4j
    await create_document(
        driver=driver,
        doc_id=doc_id,
        filename=file.filename or "unknown",
        ticker=company_ticker,
        company_name=company_name,
        doc_type=doc_type,
        period=fiscal_period,
    )

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "status": "UPLOADED",
        "auto_ingest_job_id": None,  # Would be populated if auto_ingest=True
    }


@router.get("", response_model=ListDocumentsResponse)
async def list_documents(
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    period: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> ListDocumentsResponse:
    """
    List documents with optional filters.
    
    Args:
        ticker: Filter by company ticker
        doc_type: Filter by document type
        period: Filter by fiscal period
        limit: Result limit (default: 20)
        offset: Result offset (default: 0)
        driver: Neo4j driver
        
    Returns:
        Paginated list of documents
    """
    # Build dynamic Cypher query
    where_conditions = ["d:Document"]
    params = {}
    
    if ticker:
        where_conditions.append("d.ticker = $ticker")
        params["ticker"] = ticker
    if doc_type:
        where_conditions.append("d.doc_type = $doc_type")
        params["doc_type"] = doc_type
    if period:
        where_conditions.append("d.period = $period")
        params["period"] = period

    where_clause = " AND ".join(where_conditions)
    
    count_query = f"MATCH (d:Document) WHERE {where_clause} RETURN count(d) as total"
    list_query = f"""
    MATCH (d:Document)
    WHERE {where_clause}
    RETURN {{
        id: d.id,
        filename: d.filename,
        ticker: d.ticker,
        company_name: d.company_name,
        doc_type: d.doc_type,
        period: d.period,
        created_at: d.created_at,
        status: d.status,
        stats: d.stats
    }} as doc_data
    ORDER BY d.created_at DESC
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
        records = await list_result.all()

    documents = [DocumentResponse(**record.get("doc_data")) for record in records]

    return ListDocumentsResponse(
        total=total,
        limit=limit,
        offset=offset,
        documents=documents,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> DocumentResponse:
    """
    Get document details.
    
    Args:
        doc_id: Document ID
        driver: Neo4j driver
        
    Returns:
        Document details
        
    Raises:
        DocumentNotFoundError: If document not found
    """
    query = """
    MATCH (d:Document {id: $doc_id})
    RETURN {
        id: d.id,
        filename: d.filename,
        ticker: d.ticker,
        company_name: d.company_name,
        doc_type: d.doc_type,
        period: d.period,
        created_at: d.created_at,
        status: d.status,
        stats: d.stats
    } as doc_data
    """

    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id)
        record = await result.single()

    if not record:
        raise DocumentNotFoundError(doc_id)

    return DocumentResponse(**record.get("doc_data"))


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    cascade: bool = True,
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> None:
    """
    Delete a document.
    
    Args:
        doc_id: Document ID
        cascade: Delete linked entities (default: True)
        driver: Neo4j driver
        
    Raises:
        DocumentNotFoundError: If document not found
    """
    if cascade:
        query = """
        MATCH (d:Document {id: $doc_id})
        DETACH DELETE d
        RETURN count(d) as deleted
        """
    else:
        query = """
        MATCH (d:Document {id: $doc_id})
        DELETE d
        RETURN count(d) as deleted
        """

    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id)
        record = await result.single()
        deleted = record.get("deleted", 0) if record else 0

    if deleted == 0:
        raise DocumentNotFoundError(doc_id)
