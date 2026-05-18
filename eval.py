"""
RootAlpha RAGAS Evaluation
==========================
Evaluates the GraphRAG pipeline using RAGAS metrics, with automatic
trace logging to LangSmith when LANGCHAIN_TRACING_V2=true is set.

Docs:
  https://docs.ragas.io/en/stable/howtos/integrations/langchain/
  https://docs.ragas.io/en/stable/howtos/integrations/langsmith/

Usage
-----
# Without reference answers (Faithfulness + ResponseRelevancy only):
python eval.py

# With reference answers file (all 4 metrics):
python eval.py --testset eval_testset.json

# Specify LangSmith project:
python eval.py --project rootalpha-eval

Testset JSON format (list of objects):
[
  {
    "question": "What was NVIDIA revenue in Q2 FY2025?",
    "reference": "NVIDIA total revenue in Q2 FY2025 was $30,040 million.",
    "ticker": "NVDA",
    "period": null
  },
  ...
]
If "reference" is omitted, FactualCorrectness and LLMContextRecall are skipped.
"""
from __future__ import annotations

import argparse
import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

from agent.query import build_query_graph
from graph.schema import create_schema
from runtime import configure_runtime

# ── default test questions (no reference answers → limited metrics) ──────────
DEFAULT_QUESTIONS: list[dict] = [
    {
        "question": "What was NVIDIA total revenue in Q2 FY2025 compared to Q1 FY2025?",
        "ticker": None,
        "period": None,
    },
    {
        "question": "What were the key risk factors mentioned in NVIDIA's latest filing?",
        "ticker": None,
        "period": None,
    },
    {
        "question": "Why did NVIDIA gross margin change this quarter?",
        "ticker": None,
        "period": None,
    },
]


def _context_strings(state: dict) -> list[str]:
    """Extract plain-text context strings from graph state for RAGAS."""
    contexts: list[str] = []
    for doc in state.get("chunk_hits") or []:
        text = doc.get("page_content") or ""
        if text.strip():
            contexts.append(text.strip())
    for doc in state.get("table_hits") or []:
        text = doc.get("page_content") or ""
        if text.strip():
            contexts.append(text.strip())
    return contexts or ["[no context retrieved]"]


def run_graph(graph, question: str, ticker: str | None, period: str | None) -> dict:
    """Run the RootAlpha graph and return the full state dict."""
    return graph.invoke(
        {
            "question": question,
            "company_ticker": ticker,
            "time_filter": period,
            "messages": [],
            "chunk_hits": [],
            "table_hits": [],
            "graph_paths": [],
            "evidence": [],
            "gaps": [],
        },
        config={"configurable": {"thread_id": f"eval-{hash(question)}"}},
    )


def build_dataset(graph, test_cases: list[dict]) -> tuple[list[dict], bool]:
    """Run every test case through the graph and build a RAGAS sample list."""
    samples: list[dict] = []
    has_reference = False

    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        print(f"  [{i}/{len(test_cases)}] {q[:70]}...")
        state = run_graph(graph, q, case.get("ticker"), case.get("period"))
        final_answer = state.get("final_answer") or {}
        response = final_answer.get("answer") or ""
        contexts = _context_strings(state)

        sample: dict = {
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": response,
        }
        if "reference" in case and case["reference"]:
            sample["reference"] = case["reference"]
            has_reference = True

        samples.append(sample)

    return samples, has_reference


def main() -> None:
    configure_runtime()

    # ── LangSmith tracing: set env vars before any LangChain import ──────────
    # These are already in .env if you use `langgraph dev`.
    # To enable: add to .env:
    #   LANGCHAIN_TRACING_V2=true
    #   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    #   LANGCHAIN_API_KEY=<your-key>
    #   LANGCHAIN_PROJECT=rootalpha-eval

    parser = argparse.ArgumentParser(description="Evaluate RootAlpha with RAGAS")
    parser.add_argument(
        "--testset",
        default=None,
        help="Path to JSON testset file. If omitted, uses built-in default questions.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="LangSmith project name (overrides LANGCHAIN_PROJECT env var).",
    )
    args = parser.parse_args()

    if args.project:
        os.environ["LANGCHAIN_PROJECT"] = args.project

    # ── load test cases ───────────────────────────────────────────────────────
    if args.testset:
        with open(args.testset) as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases from {args.testset}")
    else:
        test_cases = DEFAULT_QUESTIONS
        print(f"Using {len(test_cases)} built-in default questions (no reference answers)")

    # ── build graph ───────────────────────────────────────────────────────────
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_FAST_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.0,
    )
    embedder = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    create_schema(driver)
    graph = build_query_graph(driver=driver, llm=llm, embedder=embedder)

    # ── collect answers + contexts ────────────────────────────────────────────
    print("\nRunning graph on test cases...")
    samples, has_reference = build_dataset(graph, test_cases)
    driver.close()

    # ── build RAGAS EvaluationDataset ─────────────────────────────────────────
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError:
        print(
            "\nragas is not installed. Run:\n"
            "  pip install ragas\n"
            "then retry."
        )
        return

    evaluation_dataset = EvaluationDataset.from_list(samples)

    # ── choose metrics based on whether reference answers are present ─────────
    metrics = [Faithfulness(), ResponseRelevancy()]
    if has_reference:
        from ragas.metrics import FactualCorrectness, LLMContextRecall
        metrics += [LLMContextRecall(), FactualCorrectness()]
        print("\nMetrics: Faithfulness, ResponseRelevancy, LLMContextRecall, FactualCorrectness")
    else:
        print("\nMetrics: Faithfulness, ResponseRelevancy")
        print("(Add 'reference' fields to your testset for LLMContextRecall + FactualCorrectness)")

    # Wrap the same Gemini LLM as the RAGAS evaluator LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.0,
        )
    )

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    if tracing:
        print(f"LangSmith tracing ON → project: {project}")
    else:
        print("LangSmith tracing OFF (set LANGCHAIN_TRACING_V2=true in .env to enable)")

    # ── run evaluation ────────────────────────────────────────────────────────
    print("\nEvaluating...")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    scores = result.scores if hasattr(result, "scores") else {}
    # ragas returns a dict-like object
    result_dict = dict(result) if not isinstance(result, dict) else result
    for metric, score in result_dict.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {score:.4f}  [{bar}]")
    print("=" * 50)

    if tracing:
        print(f"\nTraces logged to LangSmith project: {project}")
        print("View at: https://smith.langchain.com")


if __name__ == "__main__":
    main()
