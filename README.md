# RootAlpha — Financial RAG Pipeline

Graph-augmented RAG system for SEC filings (10-Q, 8-K) and financial news.  
**Stack**: LangGraph · Neo4j · Gemini · pdfplumber · Python 3.11 · uv

---

## Project Structure

```
RootAlpha/
├── langgraph.json            # LangGraph Cloud deployment manifest
├── pyproject.toml            # Dependencies (managed by uv)
├── .env.example              # Required env vars template
│
├── data/
│   └── nvidia_investor_data_2024/
│       ├── NVIDIA_Q2_2024_10Q_Report.pdf
│       ├── NVIDIA_Q3_2024_10Q_Report.pdf
│       ├── NVIDIA_8K_Aug2024_EarningsReport.pdf
│       ├── NVIDIA_8K_Nov2024_EarningsReport.pdf
│       ├── NVIDIA_8K_Nov2024_MaterialEvent.pdf
│       └── news_text_2024/       # Plain-text news articles
│
├── src/
│   ├── config.py             # AppSettings (pydantic-settings)
│   ├── logging_config.py     # Structured JSON logging
│   ├── main.py               # FastAPI app + lifespan
│   │
│   ├── db/
│   │   ├── neo4j_client.py   # Driver singleton + session helper
│   │   └── neo4j_setup.py    # Constraints, fulltext & vector indexes
│   │
│   ├── ingestion/
│   │   ├── models.py         # Core dataclasses: DocumentInput, ParsedDocument,
│   │   │                     #   ChunkRecord, ExtractionRecord
│   │   ├── parser.py         # PDF/TXT → ParsedDocument (pdfplumber, section detection)
│   │   ├── classifier.py     # Regex + LLM metadata classification
│   │   ├── chunker.py        # ParsedDocument → list[ChunkRecord]
│   │   ├── embedder.py       # Batch embed chunks (gemini-embedding-001, 768-dim)
│   │   ├── raptor.py         # RAPTOR section summaries (Sarthi et al. 2024)
│   │   ├── extractor.py      # LLM extraction: entities, events, metrics,
│   │   │                     #   causal links, event chains
│   │   ├── graph_writer.py   # Persist to Neo4j (Documents, Chunks, Sections,
│   │   │                     #   Entities, Events, Metrics, edges)
│   │   ├── orchestration.py  # LangGraph StateGraph — fan-out per document
│   │   └── service.py        # submit_ingestion() entry point + CLI
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Vector (ANN) + fulltext retrieval, score merge
│   │   └── graph_expander.py    # Hop-1 graph context expansion from seed chunks
│   │
│   ├── query/
│   │   ├── query_classifier.py      # Route: factoid / comparative / trend / causal
│   │   ├── answer_synthesizer.py    # Gemini Pro synthesis over ranked evidence
│   │   ├── evidence_ranker.py       # Re-rank retrieved chunks by relevance
│   │   ├── coverage_gate.py         # Check answer covers required financial categories
│   │   ├── groundedness_validator.py# Verify answer is grounded in retrieved evidence
│   │   ├── orchestration.py         # LangGraph StateGraph — query pipeline
│   │   └── service.py               # ask() entry point
│   │
│   └── llm/
│       └── gemini.py         # LLM client factory (Flash + Pro models)
│
└── tests/
    ├── conftest.py
    ├── golden_qa.json                        # Ground-truth Q&A for eval
    ├── eval_ragas.py                         # RAGAS offline evaluation harness
    ├── test_ingestion_parser_chunker.py
    ├── test_ingestion_extractor.py
    ├── test_ingestion_service.py
    ├── test_graph_writer_routing.py
    ├── test_hybrid_retriever.py
    ├── test_query_classifier.py
    ├── test_query_orchestration.py
    ├── test_answer_synthesizer.py
    ├── test_evidence_ranker.py
    ├── test_coverage_gate.py
    └── test_groundedness_validator.py
```

---

## Graph Schema

```
(Company)-[:PUBLISHED]->(Document)-[:CONTAINS]->(Section)-[:HAS_CHUNK]->(Chunk)
(Document)-[:FILED_FOR]->(Quarter)-[:HAS_METRIC]->(Metric)
(Chunk)-[:PART_OF]->(Document)
(Chunk)-[:NEXT_CHUNK]->(Chunk)
(prose Chunk)-[:PARENT_OF]->(table Chunk)
(Chunk)-[:HAS_ENTITY]->(Entity)
(Chunk)-[:HAS_EVENT]->(Event)
(Chunk)-[:HAS_METRIC]->(Metric)
(Event)-[:CAUSED]->(Metric)
(Event)-[:LED_TO]->(Event)
(Chunk)-[:SIMILAR]-(Chunk)
```

**Node properties (key)**

| Node | Key properties |
|------|----------------|
| `Chunk` | `chunk_id`, `doc_id`, `company_id`, `text`, `position`, `token_count`, `section_title`, `section_id`, `page_number`, `chunk_type` (`prose`\|`table`\|`footnote`\|`title`), `parent_chunk_id`, `period_start`, `period_end`, `embedding` (768-dim) |
| `Section` | `section_id`, `title`, `doc_id`, `summary`, `summary_embedding` (768-dim, RAPTOR) |
| `Document` | `doc_id`, `company_id`, `title`, `period_start`, `period_end`, `file_type` |
| `Company` | `company_id`, `ticker` |
| `Quarter` | `label`, `period_start`, `period_end` |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and fill env vars
cp .env.example .env

# Ingest a document
uv run python -m src.ingestion.service data/nvidia_investor_data_2024/NVIDIA_Q3_2024_10Q_Report.pdf

# Run tests
uv run pytest tests/ -q
```

---

## Key Design Decisions

- **RAPTOR summaries** (`raptor.py`): Each `Section` node gets an LLM-generated summary + embedding, enabling coarse-grained section retrieval before drilling into individual chunks.
- **Table chunks**: Tables extracted by pdfplumber are stored as separate `ChunkRecord` nodes (`chunk_type="table"`) linked to their prose parent via `[:PARENT_OF]`.
- **Two-stage retrieval**: `hybrid_retriever` merges vector ANN + BM25 fulltext results; `graph_expander` hops one step into entity/event/metric neighbours for context enrichment.
- **LangGraph fan-out**: Each document is processed in a parallel `Send` branch — parse → classify → chunk → embed → RAPTOR → extract → write.
