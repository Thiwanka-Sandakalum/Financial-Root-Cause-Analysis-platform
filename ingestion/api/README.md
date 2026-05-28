# RootAlpha Ingestion API

FastAPI implementation of the document ingestion and knowledge graph pipeline following best practices for clean architecture, separation of concerns, and maintainability.

For the full ingestion system design, see [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Architecture Overview

```
ingestion/api/
├── config.py           # Configuration & environment management
├── dependencies.py     # Dependency injection (Neo4j, LLM, Embedder)
├── models.py           # Pydantic request/response schemas
├── exceptions.py       # Custom exceptions & error handlers
├── db_helpers.py       # Neo4j database operations
├── tasks.py            # Background ingestion tasks
├── main.py             # FastAPI application factory
├── run.py              # Server startup script
└── routes/
    ├── documents.py    # Document management endpoints
    └── ingestion.py    # Ingestion pipeline endpoints
```

## Key Design Principles

### 1. Dependency Injection
All external services are injected via FastAPI's `Depends()`:
```python
async def my_endpoint(
    driver: AsyncDriver = Depends(get_neo4j_driver),
    llm: ChatGoogleGenerativeAI = Depends(get_llm),
):
    # ...
```

### 2. Clean Separation of Concerns
- **Models**: Request/Response validation
- **Routes**: Endpoint logic
- **DB Helpers**: Neo4j operations
- **Tasks**: Background processing
- **Config**: Environment management

### 3. Neo4j for Job State
- All job state persisted in Neo4j (no separate queue DB)
- Queryable job history with filters
- Real-time progress updates
- Automatic cleanup with graph relationships

### 4. Async-First Architecture
- Non-blocking I/O with `async/await`
- Neo4j async driver for concurrency
- FastAPI `BackgroundTasks` for fire-and-forget jobs

### 5. Structured Error Handling
Custom exception hierarchy with consistent response format:
```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "Job 'xyz' not found",
    "timestamp": "2026-05-20T10:30:00"
  }
}
```

## Installation

### 1. Install dependencies
```bash
pip install fastapi uvicorn neo4j pydantic pydantic-settings
pip install langchain-google-genai langchain-text-splitters
```

### 2. Configure environment (.env)
```bash
# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM & Embeddings
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=models/embedding-001

# Optional
API_KEY=your-api-key-here
LLAMA_CLOUD_API_KEY=your-llama-key-here
```

### 3. Start the server
```bash
python ingestion/api/run.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Docs
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- OpenAPI JSON: `http://localhost:8000/api/openapi.json`

### Health Check
```bash
GET /health
```

### Document Management

#### Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <pdf_or_txt_file>
company_ticker: "NVDA"
company_name: "NVIDIA Corporation"
doc_type: "Annual Report"
fiscal_period: "2025"
auto_ingest: false
```

Response:
```json
{
  "document_id": "doc-uuid",
  "filename": "annual-report.pdf",
  "status": "UPLOADED",
  "auto_ingest_job_id": null
}
```

#### List Documents
```bash
GET /api/v1/documents?ticker=NVDA&limit=20&offset=0
```

#### Get Document
```bash
GET /api/v1/documents/{doc_id}
```

#### Delete Document
```bash
DELETE /api/v1/documents/{doc_id}?cascade=true
```

### Ingestion Pipeline

#### Start Ingestion Job
```bash
POST /api/v1/ingest/start
Content-Type: application/json

{
  "document_ids": ["doc-1", "doc-2"],
  "skip_if_exists": true,
  "force_reprocess": false
}
```

Response (202 Accepted):
```json
{
  "job_id": "job-uuid",
  "status": "QUEUED",
  "created_at": "2026-05-20T10:30:00",
  "documents_queued": 2
}
```

#### List Jobs
```bash
GET /api/v1/ingest/jobs?status=RUNNING&limit=50&offset=0
```

#### Get Job Status
```bash
GET /api/v1/ingest/jobs/{job_id}
```

Response:
```json
{
  "id": "job-uuid",
  "status": "RUNNING",
  "stage": "EXTRACTION",
  "progress": 65,
  "message": "Extracting entities...",
  "created_at": "2026-05-20T10:30:00",
  "started_at": "2026-05-20T10:30:05",
  "estimated_completion": null,
  "stats": {
    "documents_processed": 1,
    "chunks_created": 342
  },
  "errors": []
}
```

#### Cancel Job
```bash
POST /api/v1/ingest/jobs/{job_id}/cancel
```

## Ingestion Pipeline Stages

1. **PARSING** - Extract sections and structure from documents
2. **CHUNKING** - Split content into meaningful chunks with 2048 tokens
3. **EMBEDDING** - Generate embeddings for chunks
4. **EXTRACTION** - Extract entities and relationships from key sections
5. **WRITING** - Write chunks, entities, and relationships to Neo4j

## Background Task Processing

Jobs are queued and processed asynchronously:
- FastAPI `BackgroundTasks` for execution
- Neo4j stores job state and progress
- Real-time status updates via REST polling

## Error Handling

All errors follow a consistent structure:

```python
# Example: Document not found
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document 'doc-id' not found",
    "timestamp": "2026-05-20T10:30:00"
  }
}
```

Custom exceptions:
- `DocumentNotFoundError` (404)
- `JobNotFoundError` (404)
- `InvalidJobStateError` (409)
- `IngestError` (400)

## Testing

Example test structure:
```python
# tests/test_documents_api.py
import pytest
from fastapi.testclient import TestClient
from ingestion.api.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Production Deployment

### With Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker ingestion.api.main:app
```

### With Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "ingestion/api/run.py"]
```

### Environment Variables
Set all variables in `.env` or via cloud deployment platform (AWS, GCP, etc.)

## Contributing

### Code Style
- Use type hints everywhere
- Follow PEP 8
- Keep functions focused and simple
- Document with docstrings

### Adding New Endpoints
1. Create model in `models.py`
2. Add route in `routes/*.py`
3. Add DB helper in `db_helpers.py` if needed
4. Write tests

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Neo4j Async Driver](https://neo4j.com/docs/python-manual/current/)
- [Pydantic v2](https://docs.pydantic.dev/2.0/)
- [API Design Spec](../API_DESIGN.md)
