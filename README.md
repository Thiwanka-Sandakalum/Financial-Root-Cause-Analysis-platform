# RootAlpha Financial Root-Cause Analysis Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Neo4j-Graph_Database-blue.svg" alt="Neo4j">
  <img src="https://img.shields.io/badge/Vertex_AI-Gemini_2.5_Flash-green.svg" alt="Vertex AI">
  <img src="https://img.shields.io/badge/LlamaParse-PDF_Parsing-yellow.svg" alt="LlamaParse">
</p>

A production-grade, multi-agent AI framework utilizing **GraphRAG** to ingest complex financial quarterly reports, map causal entity relationships, and generate structurally grounded, numerically verified risk assessments.

---

## 1. System Architecture Diagrams

This system decouples knowledge graph construction (Ingestion) from user query execution (LangGraph Query Agent), using Neo4j as the single source of truth to power hybrid semantic and relational searches.

### A. Ingestion Pipeline Architecture
This pipeline parses raw PDFs, classifies financial sections, chunk-embeds narratives, extracts domain-specific entities, and constructs the Neo4j Knowledge Graph.

```mermaid
flowchart TB
    PDF[PDF] --> LP[LlamaParse]
    LP --> Classify[Classifier]
    Classify --> Narrative[Narrative Split]
    Classify --> Tables[Tables]
    Narrative --> Embed[Embedding]
    Tables --> Embed
    Embed --> DBWriter[Cypher Writer]
    DBWriter --> Neo4j[(Neo4j)]
```

### B. Query Agent & RAG Pipeline (LangGraph Workflow)
A state-driven workflow capable of cyclic self-correction, strict semantic/relational retrieval, and automated quality gate enforcement.

```mermaid
flowchart TD
    User([User Prompt]) --> State[State Initialization]
    
    subgraph LangGraph Agentic Controller
        State --> Node1[1. Analyze Request]
        Node1 -->|Classify Domain/Ticker/Intent| Node2[2. Assess Readiness]
        Node2 -->|Check Missing Context| Node2Branch{Context Complete?}
        
        Node2Branch -->|No| AskClarification(["Halt: Suggest Ingestion / Ask User"])
        Node2Branch -->|Yes| Node3[3. Plan Retrieval Tools]
        
        Node3 -->|Select Chunks, Tables, or Graph Traversal| Node4[4. Retrieve with Tools]
        
        subgraph Neo4j Integration
            Node4 -->|VectorCypherRetriever| VSearch[("Neo4j Vector Index")]
            Node4 -->|Neighborhood Expansion| GTraversal[("Neo4j Graph Database")]
        end
        
        VSearch & GTraversal -->|Retrieved Contexts| Node5[5. Merge & Rank Evidence]
        Node5 --> Node6[6. Quality Gate Node]
        
        Node6 -->|Verify Scores & Chunk Counts| GateBranch{Meets Thresholds?}
        GateBranch -->|No: Lower Scores| FailFallback(["Fallback: Insufficient Context"])
        GateBranch -->|Yes| Node7[7. Synthesize Answer]
        
        subgraph Safety & Grounding
            Node7 --> Diagnostics[Numeric Claim Diagnostics]
            Diagnostics --> SupportBranch{Support Ratio >= 40%?}
            SupportBranch -->|No| AnswerBlock[Safe Grounding Fallback]
            SupportBranch -->|Yes| AnswerApprove[Final Answer Synthesis]
        end
    end
    
    AnswerBlock & AnswerApprove --> Node8[8. Build Visualization Spec]
    Node8 --> FinalOutput(["Response + Chart JSON"])
```

### C. System Production Cloud Architecture
A highly available, production-grade cloud layout utilizing Google Cloud Platform (GCP) and containerized microservices to guarantee scalability, isolation, and secure connections.

![ARCT Diagram](docs/diagrams/arct_diagram.png)

---

## 2. Business Problem & Solution Impact

**Problem:** Financial analysts and investors spend hours manually cross-referencing dense, 100+ page quarterly reports (MD&A, Notes, Financials) to identify root causes behind revenue shifts, risk factors, and metric fluctuations. Standard LLM approaches hallucinate numbers and fail to connect relational events.

**Solution:** An autonomous GraphRAG system that parses documents into structured graphs, querying a localized Neo4j database to trace financial metrics back to their causal macro or micro events, returning numerically verified root-cause analysis in under 15 seconds.

---

## 3. Technology Stack & Frameworks Used

This project implements industry-standard engineering practices and a modern AI toolchain:

### Core Frameworks & Language
* **Python 3.11** ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white): The programming baseline for safety, asynchronous bindings, and syntax clarity.
* **FastAPI & Uvicorn** ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-purple?style=flat-square): Async REST endpoints for pipeline orchestration and frontend interface support.

### Agentic Orchestration & RAG
* **LangGraph** ![LangGraph](https://img.shields.io/badge/LangGraph-orange?style=flat-square) & **LangChain** ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white): Orchestrates our deterministic state machine, enabling complex routing, retries, and cycle mitigation.
* **`neo4j-graphrag` (VectorCypherRetriever)** ![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat-square&logo=neo4j&logoColor=white) ![RAG](https://img.shields.io/badge/RAG-blue?style=flat-square): Merges vector searches with graph-relational queries, enriching semantic matches with structural graph context.

### Ingest & Parsing Pipeline
* **LlamaParse (Llama Cloud)** ![LlamaParse](https://img.shields.io/badge/LlamaParse-black?style=flat-square): Advanced API-based parser used to classify document layout sections and isolate complex financial tables.
* **Vertex AI** ![Vertex AI](https://img.shields.io/badge/Vertex_AI-4285F4?style=flat-square&logo=google-cloud&logoColor=white) (`gemini-2.5-flash` / `text-embedding-004`): Projects narratives and tabular data into 768-dimensional vector spaces and performs reasoning.

### Database & Storage
* **Neo4j Enterprise Graph Database** ![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat-square&logo=neo4j&logoColor=white): Powers causal modeling (`CAUSED`, `IMPACTED`, `DEPENDS_ON`), document structures (`CONTAINS`), and vector indexing.
* **Google Cloud Storage (GCS)** ![GCS](https://img.shields.io/badge/GCS-4285F4?style=flat-square&logo=google-cloud-storage&logoColor=white): Acts as the landing zone for raw financial PDF documents in production.

### Evaluation, Observability & Tooling
* **Ragas** ![Ragas](https://img.shields.io/badge/Ragas-FF6F61?style=flat-square) & **Pandas** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white): Evaluates context recall, faithfulness, and answer relevance on a synthetic/curated test suite.
* **LangSmith** ![LangSmith](https://img.shields.io/badge/LangSmith-orange?style=flat-square) & **Arize Phoenix** ![Arize Phoenix](https://img.shields.io/badge/Phoenix-blue?style=flat-square): Distributed tracing, logging, and token usage accounting.
* **Docker** ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white): Containerizes the application stack for local execution and production cloud deployments.
* **Pytest** ![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white): Implements unit and integration test suites.
* **Ruff** ![Ruff](https://img.shields.io/badge/Ruff-black?style=flat-square) & **Mypy** ![Mypy](https://img.shields.io/badge/Mypy-blue?style=flat-square): Static checking, formatting, and linting tools.

---

## 4. Evaluation & Performance Metrics (Critical)

Financial AI applications require strict anti-hallucination mechanisms. This system integrates custom **Numeric Claim Diagnostics** that parse generated outputs against retrieved evidence.

* **Numeric Grounding:** If the LLM generates numeric claims where less than 40% are found in the direct context, the system triggers a hard block and returns a safe fallback message.
* **Performance Dashboard:**

| Metric Evaluation | Target Goal | Optimization Strategy | Status |
|---|---|---|---|
| Numeric Grounding Ratio | > 0.90 | Extracted tables as dedicated graph nodes to preserve row/col associations. | ✅ Passed |
| Context Relevance (GraphRAG) | > 0.85 | Hybrid Cypher queries fetching neighbor nodes (e.g. `MacroEvent -> IMPACTED -> Metric`). | ✅ Passed |
| Average Graph Traversal Latency | < 3.00s | Cypher query indexing on Ticker & Document ID. | ⚠️ Optimizing |
| Quality Gate Rejection Rate | < 5.0% | Strict LLM system prompt engineering & query expansion. | ✅ Passed |

---

## 5. Local Setup & Environment Configuration

You can configure and run the entire ecosystem (Neo4j Graph Database, FastAPI Ingestion Server, and the LangGraph Agent Studio) either using **Docker & Docker Compose** (recommended) or in a **Native Virtual Environment**.

The pipeline integrates with **Google Cloud Vertex AI** for embedding and inference APIs. Ensure your Google Cloud credentials are configured.

### Option A: Containerized Setup (Docker & Docker Compose)

This approach automatically spins up a local Neo4j database, pre-configured with APOC plugins, mounts your credentials, and starts the FastAPI ingestion service and LangGraph agent server.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/financial-root-cause-analysis.git
   cd financial-root-cause-analysis
   ```

2. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Open .env and specify your GCP Project ID and LlamaParse API Key.
   ```

3. **Authenticate Google Application Default Credentials (ADC):**
   ```bash
   # Ensure ADC credentials exist locally on your host machine
   gcloud auth application-default login --project development-498315
   ```

4. **Spin up the services:**
   ```bash
   # Mounts your ADC credentials from the host (~/.config/gcloud) into the containers
   docker-compose up --build
   ```

5. **Access the Services:**
   * **LangGraph Agent Dev UI:** [http://localhost:8080](http://localhost:8080)
   * **FastAPI Ingestion Server Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
   * **Neo4j Browser Console:** [http://localhost:7474](http://localhost:7474) (Username: `neo4j`, Password: `password123`)

---

### Option B: Native Setup (Local Python Env)

1. **Install Dependencies using uv:**
   ```bash
   # Install uv package manager if not present
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Synchronize virtual env and dependencies
   uv sync
   source .venv/bin/activate
   ```

2. **Run a Local Neo4j Instance:**
   Ensure Docker is running a standalone instance or use Neo4j Desktop:
   ```bash
   docker run \
       --name rootalpha-neo4j \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password123 \
       -e NEO4J_PLUGINS='["apoc"]' \
       -d neo4j:5.24.0-community
   ```

3. **Authenticate GCP Credentials:**
   ```bash
   gcloud auth application-default login --project development-498315
   ```

4. **Start the FastAPI Ingestion Engine:**
   ```bash
   python -m ingestion.api.run
   ```

5. **Start the LangGraph Agent Server:**
   ```bash
   langgraph dev
   ```

---

## 6. Engineering Trade-Offs & Future Scope

* **Graph Database vs. Pure Vector Store:** Switched the underlying knowledge base from a standard vector database (e.g., Pinecone) to Neo4j. *Trade-off:* Increased document ingestion and indexing latency significantly, but reduced LLM hallucinations regarding company structures and causal event chains by enabling explicit relationship traversal.
* **LangGraph vs. Linear Chains:** Opted for a LangGraph state machine over basic LangChain pipelines. *Trade-off:* Higher boilerplate complexity and steeper learning curve, but allows for crucial "Quality Gate" nodes that intercept ungrounded outputs and force the agent to re-plan its retrieval strategy.
* **Future Scope:** Implement multi-threading in the LangGraph retrieval nodes to execute the chunk vector search and Neo4j relational expansion concurrently rather than sequentially, targeting a 1.5s overall latency reduction. Explore transitioning from Vertex AI APIs to local quantized models (e.g., Llama 3) for fallback mechanisms to reduce external dependencies and cost.
