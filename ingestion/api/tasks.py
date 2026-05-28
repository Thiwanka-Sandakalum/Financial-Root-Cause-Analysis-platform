"""Background tasks for ingestion pipeline."""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import AsyncDriver, GraphDatabase

from ingestion.api.db_helpers import (
    document_has_graph_data,
    get_document_graph_stats,
    update_document_status,
    update_job_status,
)
from ingestion.api.models import DocumentStatus, IngestionStage, JobStatus
from ingestion.pipeline.graph_writer import ingest_document

logger = logging.getLogger(__name__)


# In-memory runtime cache only for transient process-local visibility.
RUNTIME_JOB_CACHE: Dict[str, Dict[str, Any]] = {}


def _set_runtime_state(job_id: str, **updates: Any) -> None:
    state = RUNTIME_JOB_CACHE.setdefault(job_id, {})
    state.update(updates)


def _run_ingest_document_sync(
    *,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    file_path: str,
    company_ticker: str,
    company_name: str,
    doc_type: str,
    fiscal_period: str,
    doc_id: str,
    llm: ChatGoogleGenerativeAI,
    embedder: GoogleGenerativeAIEmbeddings,
) -> None:
    """Execute the sync graph writer in a worker thread."""
    sync_driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )
    try:
        ingest_document(
            file_path=file_path,
            company_ticker=company_ticker,
            company_name=company_name,
            doc_type=doc_type,
            fiscal_period=fiscal_period,
            driver=sync_driver,
            embedder=embedder,
            llm=llm,
            doc_id=doc_id,
        )
    finally:
        sync_driver.close()


async def ingest_pipeline_task(
    job_id: str,
    document_ids: List[str],
    document_rows: List[Dict[str, Any]],
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    driver: AsyncDriver,
    llm: ChatGoogleGenerativeAI,
    embedder: GoogleGenerativeAIEmbeddings,
    skip_if_exists: bool = True,
    force_reprocess: bool = False,
) -> None:
    """
    Background task for ingestion pipeline.
    
    Stages:
    1. PARSING - Extract sections and structure from documents
    2. CHUNKING - Split content into meaningful chunks
    3. EMBEDDING - Generate embeddings for chunks
    4. EXTRACTION - Extract entities and relationships
    5. WRITING - Write to Neo4j
    
    Args:
        job_id: Job ID
        document_ids: List of document IDs to ingest
        driver: Neo4j driver
        llm: LLM client
        embedder: Embeddings client
        skip_if_exists: Skip if already ingested
        force_reprocess: Force reprocessing
    """
    total_docs = len(document_rows)
    processed_docs = 0
    skipped_docs = 0
    failed_docs = 0

    _set_runtime_state(
        job_id,
        total_docs=total_docs,
        processed_docs=0,
        skipped_docs=0,
        failed_docs=0,
        active_document_id=None,
        status=JobStatus.RUNNING.value,
    )

    try:
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=JobStatus.RUNNING.value,
            stage=IngestionStage.PARSING.value,
            progress=0,
            message=f"Starting ingestion for {total_docs} document(s)...",
            stats={
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "skipped_documents": skipped_docs,
                "failed_documents": failed_docs,
            },
        )

        for idx, doc in enumerate(document_rows, start=1):
            doc_id = doc.get("id", "")
            file_path = doc.get("file_path", "")
            doc_label = doc.get("filename", doc_id)

            _set_runtime_state(job_id, active_document_id=doc_id, active_filename=doc_label)

            await update_document_status(
                driver=driver,
                doc_id=doc_id,
                status=DocumentStatus.INGESTING.value,
            )

            if not Path(file_path).exists():
                failed_docs += 1
                await update_document_status(
                    driver=driver,
                    doc_id=doc_id,
                    status=DocumentStatus.FAILED.value,
                    error_message=f"File not found on disk: {file_path}",
                )
                continue

            has_graph_data = await document_has_graph_data(driver, doc_id)
            if has_graph_data and skip_if_exists and not force_reprocess:
                skipped_docs += 1
                processed_docs += 1
                await update_document_status(
                    driver=driver,
                    doc_id=doc_id,
                    status=DocumentStatus.INGESTED.value,
                    stats={"skipped": True},
                )
                progress = int(processed_docs * 100 / max(total_docs, 1))
                await update_job_status(
                    driver=driver,
                    job_id=job_id,
                    status=JobStatus.RUNNING.value,
                    stage=IngestionStage.WRITING.value,
                    progress=progress,
                    message=f"Skipped already-ingested document {doc_label}",
                    stats={
                        "total_documents": total_docs,
                        "processed_documents": processed_docs,
                        "skipped_documents": skipped_docs,
                        "failed_documents": failed_docs,
                    },
                )
                continue

            try:
                await update_job_status(
                    driver=driver,
                    job_id=job_id,
                    status=JobStatus.RUNNING.value,
                    stage=IngestionStage.PARSING.value,
                    progress=int((idx - 1) * 100 / max(total_docs, 1)),
                    message=f"Processing document {idx}/{total_docs}: {doc_label}",
                )

                await asyncio.to_thread(
                    _run_ingest_document_sync,
                    neo4j_uri=neo4j_uri,
                    neo4j_username=neo4j_username,
                    neo4j_password=neo4j_password,
                    file_path=file_path,
                    company_ticker=doc.get("ticker") or "UNKNOWN",
                    company_name=doc.get("company_name") or "Unknown Company",
                    doc_type=doc.get("doc_type") or "Document",
                    fiscal_period=doc.get("period") or "Unknown",
                    doc_id=doc_id,
                    llm=llm,
                    embedder=embedder,
                )

                doc_stats = await get_document_graph_stats(driver, doc_id)
                processed_docs += 1
                await update_document_status(
                    driver=driver,
                    doc_id=doc_id,
                    status=DocumentStatus.INGESTED.value,
                    stats=doc_stats,
                )
                progress = int(processed_docs * 100 / max(total_docs, 1))
                await update_job_status(
                    driver=driver,
                    job_id=job_id,
                    status=JobStatus.RUNNING.value,
                    stage=IngestionStage.WRITING.value,
                    progress=progress,
                    message=f"Completed document {idx}/{total_docs}: {doc_label}",
                    stats={
                        "total_documents": total_docs,
                        "processed_documents": processed_docs,
                        "skipped_documents": skipped_docs,
                        "failed_documents": failed_docs,
                    },
                )

            except Exception as doc_exc:
                failed_docs += 1
                logger.error(
                    "[%s] Ingestion failed for document %s: %s",
                    job_id,
                    doc_id,
                    doc_exc,
                    exc_info=True,
                )
                await update_document_status(
                    driver=driver,
                    doc_id=doc_id,
                    status=DocumentStatus.FAILED.value,
                    error_message=str(doc_exc),
                )

        final_status = JobStatus.COMPLETED if failed_docs == 0 else JobStatus.FAILED
        final_message = (
            "Ingestion completed successfully"
            if failed_docs == 0
            else f"Ingestion finished with {failed_docs} failed document(s)"
        )
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=final_status.value,
            stage=IngestionStage.WRITING.value,
            progress=100,
            message=final_message,
            stats={
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "skipped_documents": skipped_docs,
                "failed_documents": failed_docs,
            },
        )
        _set_runtime_state(
            job_id,
            status=final_status.value,
            processed_docs=processed_docs,
            skipped_docs=skipped_docs,
            failed_docs=failed_docs,
            active_document_id=None,
        )

    except Exception as e:
        logger.error(f"[{job_id}] Ingestion failed: {str(e)}", exc_info=True)
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=JobStatus.FAILED.value,
            progress=0,
            error_message=str(e),
            stats={
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "skipped_documents": skipped_docs,
                "failed_documents": failed_docs,
            },
        )
        _set_runtime_state(job_id, status=JobStatus.FAILED.value, active_document_id=None)
        raise
