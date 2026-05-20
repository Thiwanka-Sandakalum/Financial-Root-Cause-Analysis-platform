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
import warnings
from typing import cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

from agent.query import build_query_graph
from graph.schema import create_schema

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


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    # Keep logs readable while using legacy-compatible Ragas adapters.
    warnings.filterwarnings(
        "ignore",
        message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"LangchainLLMWrapper is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"LangchainEmbeddingsWrapper is deprecated.*",
        category=DeprecationWarning,
    )

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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Force-enable LangSmith tracing for this run.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Force-disable LangSmith tracing for this run.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Fail (exit with exit code 1) if average score of all evaluated metrics is under this value.",
    )
    args = parser.parse_args(argv)

    if args.project:
        os.environ["LANGCHAIN_PROJECT"] = args.project
        os.environ["LANGSMITH_PROJECT"] = args.project

    # Ragas/LangChain tracing reads LANGCHAIN_* vars; mirror LangSmith-only envs.
    if not os.getenv("LANGCHAIN_PROJECT") and os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

    if args.trace and args.no_trace:
        raise SystemExit("Choose only one of --trace or --no-trace")

    env_trace_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    env_trace_ls = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    tracing = args.trace or (not args.no_trace and (env_trace_v2 or env_trace_ls))

    if tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # ── load test cases ───────────────────────────────────────────────────────
    if args.testset:
        with open(args.testset) as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases from {args.testset}")
    else:
        test_cases = DEFAULT_QUESTIONS
        print(
            f"Using {len(test_cases)} built-in default questions (no reference answers)"
        )

    # ── build graph ───────────────────────────────────────────────────────────
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_FAST_MODEL")
        or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.0,
    )
    embedder = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    create_schema(driver)
    graph = build_query_graph(driver=driver, llm=llm, embedder=embedder)

    # ── collect answers + contexts ────────────────────────────────────────────
    print("\nRunning graph on test cases...")
    samples, has_reference = build_dataset(graph, test_cases)
    driver.close()

    # ── build RAGAS EvaluationDataset ─────────────────────────────────────────
    # Docs: https://docs.ragas.io/en/stable/howtos/integrations/langchain/
    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.dataset_schema import EvaluationResult
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            FactualCorrectness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        print("\nragas is not installed. Run:\n  pip install ragas\nthen retry.")
        return

    evaluation_dataset = EvaluationDataset.from_list(samples)

    # Wrap the same Gemini LLM as the RAGAS evaluator LLM
    # Matches the official docs pattern: LangchainLLMWrapper(llm)
    # Reuses the already-instantiated LangChain LLM — no second model init needed.
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedder)

    # ── choose metrics based on whether reference answers are present ─────────
    # Without reference: only generation-side metrics (no ground truth needed)
    # With reference: full suite including retrieval + factual correctness
    # strictness=1 avoids extra internal generations and speeds up small smoke runs.
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1)]
    if has_reference:
        metrics += [
            LLMContextRecall(),
            FactualCorrectness(),
            LLMContextPrecisionWithReference(),
        ]
        print(
            "\nMetrics: Faithfulness, ResponseRelevancy, LLMContextRecall, FactualCorrectness, LLMContextPrecisionWithReference"
        )
    else:
        print("\nMetrics: Faithfulness, ResponseRelevancy")
        print(
            "(Add 'reference' fields to your testset for LLMContextRecall + FactualCorrectness + LLMContextPrecisionWithReference)"
        )

    project = os.getenv("LANGCHAIN_PROJECT", "default")
    if tracing:
        has_api_key = bool(
            os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        )
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        print(f"LangSmith tracing ON → project: {project}")
        print(f"LangSmith endpoint: {endpoint}")
        if not has_api_key:
            print(
                "Warning: LANGCHAIN_API_KEY/LANGSMITH_API_KEY is not set; traces may not be uploaded."
            )
    else:
        print(
            "LangSmith tracing OFF (use --trace or set LANGCHAIN_TRACING_V2=true in .env to enable)"
        )

    # ── run evaluation ────────────────────────────────────────────────────────
    print("\nEvaluating...")
    run_config = RunConfig(timeout=300, max_retries=2, max_workers=8)
    result = cast(
        EvaluationResult,
        evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            return_executor=False,
        ),
    )

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    # In this ragas version, aggregate values are stored in _repr_dict (floats),
    # while _scores_dict contains per-sample lists.
    result_dict = getattr(result, "_repr_dict", {})
    if not isinstance(result_dict, dict) or not result_dict:
        result_dict = getattr(result, "_scores_dict", {})

    scores_found = []

    if isinstance(result_dict, dict) and result_dict:
        for metric, score in result_dict.items():
            if isinstance(score, list):
                valid = [x for x in score if isinstance(x, (int, float))]
                score = float(sum(valid) / len(valid)) if valid else 0.0
            score = float(score)
            scores_found.append(score)
            clamped = max(0.0, min(1.0, score))
            bar = "█" * int(clamped * 20) + "░" * (20 - int(clamped * 20))
            print(f"  {metric:<35} {score:.4f}  [{bar}]")
    else:
        print("  (No aggregate scores returned)")
    print("=" * 50)

    # ── per-question breakdown → CSV ──────────────────────────────────────────
    try:
        df = result.to_pandas()
        # attach category if present in test cases
        categories = [c.get("category", "") for c in test_cases]
        if len(categories) == len(df):
            df.insert(0, "category", categories)
        os.makedirs("evals/results", exist_ok=True)
        csv_path = "evals/results/eval_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nPer-question scores saved to: {csv_path}")
        # print weakest questions per metric
        metric_cols = [
            c
            for c in df.columns
            if c
            not in (
                "user_input",
                "response",
                "reference",
                "retrieved_contexts",
                "category",
            )
        ]
        if metric_cols:
            print("\nWeakest questions per metric:")
            for col in metric_cols:
                if col in df.columns and df[col].notna().any():
                    worst = df.loc[df[col].idxmin()]
                    print(f"  {col}: {worst['user_input'][:80]}  → {worst[col]:.4f}")
    except Exception as exc:
        print(f"\n(Could not export per-question CSV: {exc})")

    if tracing:
        print(f"\nTraces logged to LangSmith project: {project}")
        print("View at: https://smith.langchain.com")

    if args.fail_under is not None:
        avg_score = sum(scores_found) / len(scores_found) if scores_found else 0.0
        print(
            f"\nQuality Gate Check: Average Ragas score is {avg_score:.4f} (Required: {args.fail_under:.4f})"
        )
        if avg_score < args.fail_under:
            print("❌ Quality Gate Failed! Evaluation average is under the threshold.")
            raise SystemExit(1)
        else:
            print("✅ Quality Gate Passed!")


if __name__ == "__main__":
    main()
