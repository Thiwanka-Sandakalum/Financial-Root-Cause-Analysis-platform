<div align="center">

# RootAlpha

**Multi-Agent GraphRAG Platform for Financial Root-Cause Analysis**

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00?style=flat-square&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-FF6F00?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![Neo4j](https://img.shields.io/badge/Neo4j_Enterprise-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Vertex AI](https://img.shields.io/badge/Vertex_AI-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Demo Video](https://img.shields.io/badge/Demo_Video-YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/uc_bf1roLrs?si=Tusse26CBd6SNBsB)

Ingests quarterly reports → builds a causal knowledge graph → returns numerically-verified root-cause analysis in under **15 seconds**.

🎬 **[Watch the 5-Minute Demo Video on YouTube](https://youtu.be/uc_bf1roLrs?si=Tusse26CBd6SNBsB)**

</div>

---

## Overview

Every quarter, financial analysts spend hours manually cross-referencing dense, 120-page quarterly reports to understand the drivers behind margin compressions, revenue misses, and balance sheet shifts. While generic AI tools exacerbate this by hallucinating numbers and missing cross-section connections, **RootAlpha** automates financial root-cause analysis in seconds by reading reports like a senior analyst. By isolating tables and building a causal knowledge graph (linking metrics to events, event risks, and macro conditions), RootAlpha maps cause-and-effect relationships explicitly, providing interactive page citations and strictly hard-blocking any numeric claims that cannot be mathematically verified against the source text.

---

## Architecture

RootAlpha runs in two stages: an **asynchronous document ingestion pipeline** that extracts tables and constructs the knowledge graph in Neo4j, and a stateful **LangGraph query agent** that executes hybrid retrieval (vector search + Cypher graph traversal), validates response quality, and verifies numeric claims.

### ☁️ Cloud Architecture
![Cloud Architecture](docs/diagrams/arct_diagram.png)

### 📄 Ingestion Pipeline
```mermaid
flowchart LR
    PDF["📄 PDF"] --> LP["LlamaParse"]
    LP --> CL["Classifier"]
    CL --> NR["Narrative Split"]
    CL --> TB["Tables"]
    NR & TB --> EM["text-embedding-004\n768-dim Vectors"]
    EM --> CW["Cypher Writer"]
    CW --> NEO[("Neo4j\nGraph DB")]
```

### 🤖 Query Agent State Machine
```mermaid
flowchart TD
    User(["User Prompt"]) --> State["State Initialization"]

    subgraph LangGraph ["LangGraph Agentic Controller"]
        State --> N1["1 · Analyze Request"]
        N1 --> N2["2 · Assess Readiness"]
        N2 --> B1{"Context\nComplete?"}

        B1 -- No --> Halt(["Halt · Suggest Ingestion"])
        B1 -- Yes --> N3["3 · Plan Retrieval"]
        N3 --> N4["4 · Retrieve"]

        subgraph Neo4j ["Neo4j Integration"]
            N4 --> VS[("Vector Index\nVectorCypherRetriever")]
            N4 --> GT[("Graph Traversal\nNeighborhood Expansion")]
        end

        VS & GT --> N5["5 · Merge & Rank Evidence"]
        N5 --> N6["6 · Quality Gate"]
        N6 --> B2{"Meets\nThresholds?"}

        B2 -- No --> Fallback(["Fallback · Insufficient Context"])
        B2 -- Yes --> N7["7 · Synthesize Answer"]

        subgraph Grounding ["Safety & Grounding"]
            N7 --> Diag["Numeric Claim Diagnostics"]
            Diag --> B3{"Support\nRatio ≥ 40%?"}
            B3 -- No --> Safe["Safe Grounding Fallback"]
            B3 -- Yes --> Final["Final Answer Synthesis"]
        end
    end

    Safe & Final --> N8["8 · Build Visualization Spec"]
    N8 --> Out(["Response + Chart JSON"])
```

---

## Tech Stack

* **Agent Orchestration**: LangGraph, LangChain
* **Graph Database**: Neo4j Enterprise (APOC, Vector Index, `VectorCypherRetriever`)
* **LLMs & Embeddings**: Google Vertex AI (`gemini-2.5-flash`, `text-embedding-004`)
* **PDF Parsing**: LlamaParse (layout classification, table isolation)
* **Backend API**: Python 3.11, FastAPI, Uvicorn
* **Frontend UI**: React, Vite, Tailwind CSS, Recharts
* **Testing & DevOps**: Docker Compose, Pytest, Ruff, Mypy, LangSmith, Phoenix

---

## Evaluation

RootAlpha enforces anti-hallucination at the numeric level. If fewer than **40%** of generated numeric claims are traceable to retrieved context, the system hard-blocks and returns a safe grounding fallback.

Below is the execution output of the automated Ragas evaluation suite:

![Ragas Evaluation Results](docs/image.png)

---

## Quick Start

### Prerequisites
* Docker & Docker Compose
* `gcloud` CLI (authenticated with a project hosting Vertex AI APIs)
* LlamaParse API Key

### Deploy with Docker (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/your-username/financial-root-cause-analysis.git
cd financial-root-cause-analysis

# 2. Configure environment
cp .env.example .env
# Set GCP_PROJECT_ID and LLAMA_PARSE_API_KEY in .env

# 3. Authenticate Google Application Default Credentials (ADC)
gcloud auth application-default login --project your-gcp-project-id

# 4. Spin up containerized microservices
docker-compose up --build
```

| Service | Local URL |
|---|---|
| LangGraph Agent UI | http://localhost:8080 |
| FastAPI Ingestion Docs | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 — `neo4j` / `password123` |

---

## Key Design Decisions

* **Neo4j Graph Database**: Replacing traditional vector stores with a knowledge graph allowed explicit representation of causal connections (`CAUSED`, `IMPACTED`, `DEPENDS_ON`), resolving multi-document root-cause tracing issues.
* **Stateful LangGraph Agent**: Using a state-machine framework enables cyclic correction and fallback routing (like triggering quality gate re-runs or requesting user input), which is impossible with linear chains.
* **Numeric Guardrails**: Injecting regex-based validation of generated numbers against raw source context guarantees strict grounding for high-stakes financial analysis.