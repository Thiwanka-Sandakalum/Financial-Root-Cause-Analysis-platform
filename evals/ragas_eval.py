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
import pandas as pd

from query.agent_query.graph import build_query_graph
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

EXPECTED_DOMAIN_VALUES = {"financial_supported", "non_financial"}


def _normalize_expected_domain(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    value = raw.strip().lower()
    if value in EXPECTED_DOMAIN_VALUES:
        return value
    return ""


def _infer_expected_domain(case: dict) -> str:
    explicit = _normalize_expected_domain(case.get("expected_domain"))
    if explicit:
        return explicit

    category = str(case.get("category") or "").strip().lower()
    if category in {"robustness", "non_financial", "out_of_scope"}:
        return "non_financial"
    return "financial_supported"


def _state_diagnostics(state: dict) -> dict:
    final_answer = state.get("final_answer") or {}
    citations = final_answer.get("citations") if isinstance(final_answer, dict) else []
    if not isinstance(citations, list):
        citations = []
    evidence = state.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []

    retrieval_quality = state.get("retrieval_quality") or {}
    if not isinstance(retrieval_quality, dict):
        retrieval_quality = {}

    evaluation_signals = state.get("evaluation_signals") or {}
    if not isinstance(evaluation_signals, dict):
        evaluation_signals = {}

    policy_flags = state.get("policy_flags") or {}
    if not isinstance(policy_flags, dict):
        policy_flags = {}

    domain_status = state.get("domain_status")
    if not isinstance(domain_status, str):
        domain_status = ""

    domain_reason = policy_flags.get("domain_reason")
    if not isinstance(domain_reason, str):
        domain_reason = ""

    domain_confidence = policy_flags.get("domain_confidence")
    if not isinstance(domain_confidence, str):
        domain_confidence = ""

    domain_signals = policy_flags.get("domain_signals")
    if isinstance(domain_signals, list):
        domain_signals_text = ", ".join(str(item).strip() for item in domain_signals if str(item).strip())
    else:
        domain_signals_text = ""

    return {
        "retry_count": int(state.get("retry_count") or 0),
        "error_class": state.get("error_class") or "",
        "blocked_reason": state.get("blocked_reason") or "",
        "domain_status": domain_status,
        "domain_gate_passed": bool(policy_flags.get("domain_gate_passed") or False),
        "domain_confidence": domain_confidence,
        "domain_reason": domain_reason,
        "domain_signals": domain_signals_text,
        "domain_adjudication_used": bool(policy_flags.get("domain_adjudication_used") or False),
        "retrieval_evidence_count": int(retrieval_quality.get("evidence_count") or len(evidence)),
        "retrieval_avg_score": float(retrieval_quality.get("avg_score") or 0.0),
        "citation_count": len(citations),
        "citation_coverage": float(evaluation_signals.get("citation_coverage") or 0.0),
        "answer_confidence": str(evaluation_signals.get("confidence") or final_answer.get("confidence") or ""),
        "claim_support_ratio": float(evaluation_signals.get("claim_support_ratio") or 0.0),
        "claim_total_count": int(evaluation_signals.get("claim_total_count") or 0),
        "unsupported_claim_count": int(evaluation_signals.get("unsupported_claim_count") or 0),
        "confidence_overridden": bool(evaluation_signals.get("confidence_overridden") or False),
    }


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
            "retry_count": 0,
        },
        config={"configurable": {"thread_id": f"eval-{hash(question)}"}},
    )


def build_dataset(graph, test_cases: list[dict]) -> tuple[list[dict], bool, list[dict]]:
    """Run every test case through the graph and build a RAGAS sample list."""
    samples: list[dict] = []
    diagnostics: list[dict] = []
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
        diagnostics.append(_state_diagnostics(state))

    return samples, has_reference, diagnostics


def _compute_domain_gate_summary(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty or "expected_domain" not in df.columns:
        return {}

    labels = df["expected_domain"].fillna("").astype(str).str.strip().str.lower()
    valid_mask = labels.isin(EXPECTED_DOMAIN_VALUES)
    if not valid_mask.any():
        return {}

    eval_df = df.loc[valid_mask].copy()
    eval_labels = labels.loc[valid_mask]
    blocked_reason = eval_df.get("blocked_reason", pd.Series([""] * len(eval_df))).fillna("").astype(str)

    blocked = blocked_reason != ""
    predicted_non_financial = blocked
    actual_non_financial = eval_labels == "non_financial"

    tp = int((predicted_non_financial & actual_non_financial).sum())
    fp = int((predicted_non_financial & ~actual_non_financial).sum())
    tn = int((~predicted_non_financial & ~actual_non_financial).sum())
    fn = int((~predicted_non_financial & actual_non_financial).sum())

    total = int(len(eval_df))
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    false_block_rate = fp / int((~actual_non_financial).sum()) if int((~actual_non_financial).sum()) else 0.0
    false_allow_rate = fn / int(actual_non_financial.sum()) if int(actual_non_financial.sum()) else 0.0

    return {
        "labeled_samples": total,
        "accuracy": accuracy,
        "precision_non_financial": precision,
        "recall_non_financial": recall,
        "f1_non_financial": f1,
        "false_block_rate": false_block_rate,
        "false_allow_rate": false_allow_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


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
    samples, has_reference, diagnostics = build_dataset(graph, test_cases)
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
        if diagnostics and len(diagnostics) == len(df):
            diag_df = pd.DataFrame(diagnostics)
            df = pd.concat([df, diag_df], axis=1)
        # attach category if present in test cases
        categories = [c.get("category", "") for c in test_cases]
        if len(categories) == len(df):
            df.insert(0, "category", categories)
        expected_domain = [_infer_expected_domain(c) for c in test_cases]
        if len(expected_domain) == len(df):
            insert_at = 1 if "category" in df.columns else 0
            df.insert(insert_at, "expected_domain", expected_domain)
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
                "expected_domain",
                "retry_count",
                "error_class",
                "blocked_reason",
                "domain_status",
                "domain_gate_passed",
                "domain_confidence",
                "domain_reason",
                "domain_signals",
                "domain_adjudication_used",
                "retrieval_evidence_count",
                "retrieval_avg_score",
                "citation_count",
                "citation_coverage",
                "answer_confidence",
                "claim_support_ratio",
                "claim_total_count",
                "unsupported_claim_count",
                "confidence_overridden",
            )
        ]
        if metric_cols:
            print("\nWeakest questions per metric:")
            for col in metric_cols:
                if col in df.columns and df[col].notna().any():
                    worst = df.loc[df[col].idxmin()]
                    print(f"  {col}: {worst['user_input'][:80]}  → {worst[col]:.4f}")

        domain_summary = _compute_domain_gate_summary(df)
        if domain_summary:
            print("\nDomain Gate QA Summary (labeled):")
            print(f"  labeled_samples: {int(domain_summary['labeled_samples'])}")
            print(f"  accuracy: {float(domain_summary['accuracy']):.4f}")
            print(f"  precision_non_financial: {float(domain_summary['precision_non_financial']):.4f}")
            print(f"  recall_non_financial: {float(domain_summary['recall_non_financial']):.4f}")
            print(f"  f1_non_financial: {float(domain_summary['f1_non_financial']):.4f}")
            print(f"  false_block_rate: {float(domain_summary['false_block_rate']):.4f}")
            print(f"  false_allow_rate: {float(domain_summary['false_allow_rate']):.4f}")

            summary_path = "evals/results/domain_gate_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(domain_summary, f, indent=2)
            print(f"Domain gate summary saved to: {summary_path}")
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
