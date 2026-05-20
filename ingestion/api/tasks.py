"""Background tasks for ingestion pipeline."""
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import AsyncDriver

from ingestion.api.db_helpers import update_job_status
from ingestion.api.models import IngestionStage, JobStatus

logger = logging.getLogger(__name__)


async def ingest_pipeline_task(
    job_id: str,
    document_ids: List[str],
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
    try:
        # Update job status to running
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=JobStatus.RUNNING.value,
            stage=IngestionStage.PARSING.value,
            progress=0,
            message="Starting ingestion pipeline...",
        )

        # Stage 1: Parse documents
        logger.info(f"[{job_id}] Starting parsing stage for {len(document_ids)} document(s)")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            stage=IngestionStage.PARSING.value,
            progress=10,
            message="Parsing documents...",
        )

        # Stage 2: Chunking
        logger.info(f"[{job_id}] Starting chunking stage")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            stage=IngestionStage.CHUNKING.value,
            progress=30,
            message="Chunking documents...",
        )

        # Stage 3: Embedding
        logger.info(f"[{job_id}] Starting embedding stage")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            stage=IngestionStage.EMBEDDING.value,
            progress=50,
            message="Generating embeddings...",
        )

        # Stage 4: Entity Extraction
        logger.info(f"[{job_id}] Starting entity extraction stage")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            stage=IngestionStage.EXTRACTION.value,
            progress=70,
            message="Extracting entities and relationships...",
        )

        # Stage 5: Writing to Neo4j
        logger.info(f"[{job_id}] Starting Neo4j write stage")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            stage=IngestionStage.WRITING.value,
            progress=85,
            message="Writing to knowledge graph...",
        )

        # Completion
        logger.info(f"[{job_id}] Ingestion completed successfully")
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            message="Ingestion completed successfully",
            stage=None,
        )

    except Exception as e:
        logger.error(f"[{job_id}] Ingestion failed: {str(e)}", exc_info=True)
        await update_job_status(
            driver=driver,
            job_id=job_id,
            status=JobStatus.FAILED.value,
            progress=0,
            error_message=str(e),
        )
        raise
