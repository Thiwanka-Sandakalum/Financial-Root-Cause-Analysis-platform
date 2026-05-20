"""
ingestion — Document ingestion package.

Sub-packages:
  ingestion.pipeline  Core parsing / chunking / entity-extraction / Neo4j write
  ingestion.api       FastAPI REST service for the ingestion pipeline
"""

# Re-export the main orchestration entry point for backwards compatibility
from ingestion.pipeline.graph_writer import ingest_document

__all__ = ["ingest_document"]
