# RootAlpha Ingestion Pipeline Architecture

## Purpose

This document describes the production-grade architecture of the RootAlpha ingestion pipeline.
It is written as the target operating model for the system: reliable, restart-safe, observable,
and intentionally lightweight.

The ingestion pipeline is responsible for:

- accepting uploaded source documents
- persisting document metadata and job state
- parsing and structuring financial content
- generating chunks, embeddings, and extracted entities
- writing the final knowledge graph into Neo4j

## Design Goals

The pipeline is designed around five constraints:

1. Neo4j is the only persistent database.
2. No additional operational infrastructure is required for queueing or state management.
3. Transient execution state may live in process memory only.
4. The system must be safe to retry and safe to restart.
5. Every ingestion run must be inspectable through API status and graph state.

## High-Level Architecture

The system is split into four logical layers:

1. API layer
- Accepts uploads and ingestion requests.
- Performs request validation.
- Persists document and job metadata in Neo4j.
- Exposes job and document status endpoints.

2. Runtime orchestration layer
- Resolves document records to local files.
- Manages in-memory runtime state for active jobs.
- Bridges async API execution to the synchronous graph-writing pipeline.
- Updates job and document lifecycle state in Neo4j.

3. Pipeline layer
- Parses PDF or text documents into sections.
- Chunks narrative content.
- Embeds chunks and tables.
- Extracts entities and relationships.
- Writes graph nodes and relationships into Neo4j.

4. Persistence layer
- Stores all durable job and document state in Neo4j.
- Stores uploaded source files on local disk.
- Uses local file paths as the durable pointer from API metadata to pipeline execution.

## Component Map

```text
Client
  |
  v
FastAPI API
  |- /api/v1/documents/upload
  |- /api/v1/ingest/start
  |- /api/v1/ingest/jobs
  |
  v
Neo4j
  |- Document nodes
  |- Job nodes
  |- Graph entities and relationships
  |
  v
Background ingestion task
  |- in-memory runtime cache
  |- sync pipeline bridge
  |
  v
Ingestion pipeline
  |- parser
  |- chunker
  |- embedder
  |- entity extractor
  |- graph writer
  |
  v
Local upload storage
```

## Core Principles

### 1. Neo4j as the system of record

Neo4j stores every durable state transition relevant to ingestion:

- uploaded documents
- ingestion jobs
- document lifecycle state
- processing statistics
- final graph structure

This removes the need for a second persistent operational store.

### 2. In-memory runtime state only for execution convenience

The API process may keep a small in-memory cache for active job context, such as:

- active document id
- currently running filename
- per-process counters

This cache is not authoritative. If the process restarts, Neo4j remains the source of truth.

### 3. Durable file-to-document mapping

Uploaded files are written to local disk immediately and referenced from the `Document` node.
The pipeline never depends on the raw multipart request after upload completes.

### 4. Idempotent graph writes

Graph writes are designed to tolerate retries.
Stable identifiers and schema-aware `MERGE` operations are used wherever the graph model allows.

### 5. Observable execution

Every job exposes:

- status
- stage
- progress
- message
- start time
- failure reason
- output statistics

This allows API consumers and operators to understand what the pipeline is doing without direct
log access.

## End-to-End Flow

### Step 1. Upload

The client sends a multipart upload request.

The API performs the following actions:

1. validates the request fields
2. assigns a `document_id`
3. stores the file under `data/uploads/<document_id>.<ext>`
4. calculates persisted file metadata
5. creates a `Document` node in Neo4j

Persisted document metadata includes:

- `id`
- `filename`
- `file_path`
- `mime_type`
- `size_bytes`
- `ticker`
- `company_name`
- `doc_type`
- `period`
- `status`
- `created_at`

### Step 2. Start ingestion

The client submits one or more `document_ids` to the ingestion endpoint.

The API:

1. verifies every document exists
2. verifies every document has a stored file path
3. verifies the file is still present on disk
4. creates a `Job` node in Neo4j
5. schedules a background ingestion task

### Step 3. Runtime orchestration

The background task:

1. marks the job `RUNNING`
2. marks each document `INGESTING`
3. checks whether graph data already exists for the document
4. skips, retries, or processes the document according to request flags
5. calls the synchronous ingestion pipeline through a thread bridge
6. updates Neo4j with final job and document state

### Step 4. Parsing

The pipeline chooses the parser based on file type:

- PDF files use LlamaParse plus section classification
- text or markdown files use direct text parsing

The parser produces a list of `ParsedSection` objects containing:

- `section_type`
- `title`
- `content`
- `tables`
- `headings`
- `page_start`
- `page_end`

### Step 5. Chunking

Narrative content is split into retrieval-friendly chunks.
Tables remain separate first-class objects rather than being flattened into chunk text.

Each chunk carries:

- a stable id
- parent section id
- section type
- sequence position
- source page
- source document id

### Step 6. Embedding

Chunk text and table text are embedded in batch form.
This produces vector representations used for later retrieval and graph-aware question answering.

### Step 7. Entity extraction

Selected sections such as MD&A, risk, or earnings commentary are passed to the extraction layer.
This produces:

- typed entities
- typed relationships

The extractor enforces a controlled label set to avoid schema drift.

### Step 8. Graph writing

The graph writer persists:

- `Company`
- `Document`
- `Section`
- `Chunk`
- `Table`
- extracted entity nodes
- semantic relationships
- sequential `NEXT_CHUNK` links

The final graph becomes the durable retrieval substrate for downstream query and reasoning flows.

## Data Model

### Document node

Represents a source file accepted by the system.

Key properties:

- `id`
- `filename`
- `file_path`
- `mime_type`
- `size_bytes`
- `ticker`
- `company_name`
- `doc_type`
- `period`
- `status`
- `stats`
- `created_at`
- `updated_at`
- `error_message`

### Job node

Represents one ingestion request over one or more documents.

Key properties:

- `id`
- `status`
- `stage`
- `progress`
- `message`
- `documents_count`
- `started_at`
- `created_at`
- `updated_at`
- `stats`
- `error_message`

### Graph content

Graph content produced by ingestion includes:

- `Company`
- `Document`
- `Section`
- `Chunk`
- `Table`
- `FinancialMetric`
- `FinancialEvent`
- `Product`
- `Executive`
- `RiskFactor`
- `MacroEvent`

## Status Model

### Document lifecycle

Document state transitions:

```text
UPLOADED -> INGESTING -> INGESTED
UPLOADED -> INGESTING -> FAILED
```

### Job lifecycle

Job state transitions:

```text
QUEUED -> RUNNING -> COMPLETED
QUEUED -> RUNNING -> FAILED
RUNNING -> CANCELLED
```

### Stage model

The pipeline exposes these logical stages:

1. `UPLOADING`
2. `PARSING`
3. `CHUNKING`
4. `EMBEDDING`
5. `EXTRACTION`
6. `WRITING`

Stages are externally visible even when internal implementation details evolve.

## Reliability Model

### Restart behavior

Because Neo4j stores the durable record of jobs and documents, the system can recover from API
process restarts without losing authoritative state.

If in-memory runtime state is lost:

- the API can still report the last persisted job state from Neo4j
- operators can reconcile orphaned `RUNNING` jobs
- jobs can be retried using existing `Document.file_path` metadata

### Retry behavior

The pipeline includes bounded retry behavior around external provider calls.
Retries are intended only for transient failures such as:

- temporary LLM provider unavailability
- temporary parsing provider failures
- transient embedding API issues

Retries do not replace idempotent write design.

### Skip and reprocess semantics

The ingestion API supports two explicit policies:

- `skip_if_exists=true`
- `force_reprocess=true`

These determine whether a document with existing graph content is skipped or re-run.

## Operational Guarantees

The architecture aims to provide the following guarantees:

1. Every accepted upload is either present on disk or the request fails.
2. Every ingestion request is either represented by a `Job` node or rejected before execution.
3. A job failure is reflected in Neo4j, not only in logs.
4. Document and job lifecycle state remain queryable through the API.
5. A retryable ingestion does not require a second persistent database.

## Failure Handling

The system distinguishes four failure classes:

### 1. Request validation failure

Examples:

- missing document id
- missing file path
- file removed from local disk

These fail before a job is executed.

### 2. External provider failure

Examples:

- LlamaParse API failure
- Gemini model access failure
- embedding API failure

These are retried when reasonable and recorded on the affected job or document.

### 3. Graph write failure

Examples:

- schema constraint violations
- invalid merge keys
- transaction errors

These fail the current document and surface through the job state.

### 4. Partial pipeline failure

If one document in a multi-document job fails, the job captures aggregate outcome through stats.
The system records per-document success or failure so operators can retry only the affected items.

## Security and Safety

The ingestion architecture applies the following safeguards:

- uploaded files are written under a controlled directory
- persisted file names are generated from internal identifiers
- external provider credentials are loaded from environment configuration
- no user-supplied path is trusted as a storage location
- request validation occurs before execution begins

## Observability

Production operation depends on three signals:

### API-visible status

Available through:

- document listing and detail endpoints
- job listing and detail endpoints

### Structured logs

Logs should include:

- `job_id`
- `document_id`
- stage
- provider failure details
- retry attempts
- final counts

### Graph verification

Operators can confirm successful ingestion directly in Neo4j by checking:

- document status
- linked sections
- linked chunks
- linked tables
- company linkage

## Testing Strategy

The ingestion pipeline is validated at three levels:

### 1. Unit tests

Validate isolated logic such as:

- upload persistence
- request validation
- helper behavior
- lifecycle state handling

### 2. API tests

Validate:

- upload success and failure cases
- ingestion start validation
- stable job response shapes
- document and job state transitions

### 3. Integration tests

Validate end-to-end ingestion against Neo4j by asserting:

- document creation
- section creation
- chunk creation
- graph linkage
- failure reporting

## Deployment Model

This design intentionally avoids infrastructure-heavy dependencies.

Required runtime components:

- FastAPI application
- Neo4j database
- local disk for uploaded files
- access to parsing, embedding, and extraction providers

Not required:

- Redis
- Postgres
- Celery
- Kafka
- separate queue database

This makes the system operationally small while still preserving durable execution state.

## Recommended Repository Layout

```text
ingestion/
├── ARCHITECTURE.md
├── api/
│   ├── main.py
│   ├── routes/
│   ├── tasks.py
│   ├── db_helpers.py
│   └── config.py
├── pipeline/
│   ├── parser.py
│   ├── chunker.py
│   ├── entity_extractor.py
│   └── graph_writer.py
└── data/
    └── uploads/
```

## Summary

The RootAlpha ingestion pipeline is built around a simple principle: keep the architecture small,
but make state durable and execution explicit.

Neo4j is the only persistent database.
Local disk stores uploaded source documents.
FastAPI provides ingestion control and visibility.
The pipeline converts source documents into graph-ready financial knowledge.

This architecture is intentionally lightweight, but it is structured to support reliable ingestion,
clear operational status, and production-oriented graph building without introducing unnecessary
infrastructure.