"""Ingestion service: public API for submitting documents to the pipeline.

Provides:
- ``IngestionUploadRequest`` — per-file input model
- ``IngestionRequest`` — collection of uploads
- ``submit_ingestion()`` — entry point that manages the Neo4j client lifecycle
- ``main()`` — CLI entry point for the ``ingest-documents`` script
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.db.neo4j_client import Neo4jClient
from src.db.neo4j_setup import initialize_neo4j_schema
from src.ingestion.orchestration import run_ingestion


class IngestionUploadRequest(BaseModel):
    """Per-file upload parameters."""

    file_path: str
    company_id: str | None = None
    source_type: str | None = None
    title: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    upload_id: str | None = None
    auto_detect: bool = False


class IngestionRequest(BaseModel):
    """Request to ingest one or more documents."""

    uploads: list[IngestionUploadRequest] = Field(default_factory=list)


def submit_ingestion(
    request: IngestionRequest,
    settings: Settings | None = None,
    neo4j_client: Any | None = None,
) -> dict[str, Any]:
    """Run the ingestion pipeline for the given request.

    Manages the Neo4j client lifecycle: creates and closes a client unless
    one is provided by the caller.  When the caller supplies ``neo4j_client``
    it is reused and not closed on return.
    """
    if settings is None:
        settings = get_settings()

    owns_client = neo4j_client is None
    if owns_client:
        neo4j_client = Neo4jClient(settings)

    try:
        # Ensure constraints/indexes exist so first-time ingestion doesn't fail
        # when retrieval/post-processing expects vector/fulltext indexes.
        has_verify = hasattr(neo4j_client, "verify_connectivity")
        has_driver = hasattr(neo4j_client, "driver")
        if has_verify and has_driver:
            neo4j_client.verify_connectivity()
            initialize_neo4j_schema(neo4j_client.driver, settings)

        uploads = [
            {
                "file_path": str(Path(u.file_path).resolve()),
                "company_id": u.company_id,
                "source_type": u.source_type,
                "auto_detect": u.auto_detect,
                "period_start": u.period_start,
                "period_end": u.period_end,
                "upload_id": u.upload_id,
            }
            for u in request.uploads
        ]
        return run_ingestion(
            uploads=uploads,
            settings=settings,
            neo4j_client=neo4j_client,
        )
    finally:
        if owns_client:
            neo4j_client.close()


def main() -> None:
    """CLI entry point for the ``ingest-documents`` script."""
    parser = argparse.ArgumentParser(description="Ingest documents into RootAlpha")
    parser.add_argument("files", nargs="+", help="Paths to documents to ingest")
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--auto-detect", action="store_true", default=False)
    parser.add_argument("--period-start", default=None)
    parser.add_argument("--period-end", default=None)
    args = parser.parse_args()

    settings = get_settings()
    request = IngestionRequest(
        uploads=[
            IngestionUploadRequest(
                file_path=f,
                source_type=args.source_type,
                auto_detect=args.auto_detect,
                period_start=args.period_start,
                period_end=args.period_end,
            )
            for f in args.files
        ]
    )

    result = submit_ingestion(request, settings=settings)

    import json

    print(json.dumps(result, indent=2))
    if result.get("status") not in {"completed", "completed_with_warnings"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
