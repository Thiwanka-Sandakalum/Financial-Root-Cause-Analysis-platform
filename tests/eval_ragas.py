"""RAGAS evaluation harness for the RootAlpha financial RAG pipeline.

Metrics evaluated (all 0–1, higher is better):
  - Faithfulness        : are answer claims supported by retrieved chunks?
  - ContextUtilization  : are the most relevant chunks ranked first?
  - AnswerRelevancy     : does the answer address the question?
  - ContextRecall       : did retrieval surface all info needed to answer?
  - FactualCorrectness  : is the answer factually accurate vs. reference?

Run:
    uv run python tests/eval_ragas.py

LangSmith tracing:
    The pipeline's LangChain calls are traced automatically via LANGCHAIN_TRACING_V2
    and LANGSMITH_API_KEY (already set in .env).
    Ragas evaluator LLM calls go through litellm, which also picks up LangSmith
    tracing via the same LANGCHAIN_TRACING_V2 / LANGSMITH_API_KEY env vars.

Results are printed to the console and saved to tests/ragas_results.csv.

Golden dataset: edit tests/golden_qa.json to add/update (question, reference_answer, company_id)
entries sourced from your ingested documents.  reference_answer is required for
ContextRecall and FactualCorrectness; samples without one skip those two metrics.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Golden QA dataset
# Maintained in tests/golden_qa.json — add/edit entries there.
# Fields: question (str), company_id (str), reference_answer (str | null)
# ---------------------------------------------------------------------------

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"


@dataclass
class GoldenSample:
    question: str
    company_id: str
    reference_answer: str | None = None


def _load_golden_qa(path: Path = GOLDEN_QA_PATH) -> list[GoldenSample]:
    with path.open() as f:
        raw = json.load(f)
    return [
        GoldenSample(
            question=item["question"],
            company_id=item["company_id"],
            reference_answer=item.get("reference_answer"),
        )
        for item in raw
    ]

# ---------------------------------------------------------------------------
# Pipeline runner — returns (response_text, retrieved_contexts)
# ---------------------------------------------------------------------------

def _run_pipeline(
    question: str,
    company_id: str,
) -> tuple[str, list[str]]:
    """Invoke the query pipeline and return the answer + retrieved chunk texts."""
    # Lazy imports: pipeline setup requires env vars / Neo4j to be live.
    from src.config import get_settings
    from src.db.neo4j_client import Neo4jClient
    from src.query.orchestration import QueryPipeline

    settings = get_settings()
    client = Neo4jClient(settings)
    client.verify_connectivity()

    pipeline = QueryPipeline(settings, client)

    # Call compiled_graph directly to get the full state (including ranked_evidence).
    raw_state = pipeline.compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "company_id": company_id,
            "period_start": None,
            "period_end": None,
        }
    )

    final_response: dict[str, Any] = raw_state.get("final_response", {})
    print(f"  raw final_response: {final_response}")
    answer: str = final_response.get("answer", "") or ""

    # The graph uses output_schema=OutputState which hides ranked_evidence.
    # Extract context snippets from citations instead — each citation carries
    # the raw_text evidence chunk that was used to write the answer.
    citations = final_response.get("citations", [])
    retrieved_contexts: list[str] = []
    for c in citations:
        text = c.get("raw_text", "") if isinstance(c, dict) else getattr(c, "raw_text", "")
        if text:
            retrieved_contexts.append(str(text))

    client.close()
    return answer, retrieved_contexts


# ---------------------------------------------------------------------------
# RAGAS evaluator setup
# ragas 0.4.x collections metrics require InstructorLLM via llm_factory —
# LangchainLLMWrapper is not supported by these metrics.
#
# Non-Vertex path: OpenAI-compat Gemini endpoint via openai client.
# Vertex AI path:  litellm with vertex_ai/ model prefix + ADC credentials.
#
# LangSmith tracing: the pipeline itself runs through LangChain, so
# LANGCHAIN_TRACING_V2=true (already in .env) traces all pipeline calls.
# Ragas metric LLM calls go through litellm and appear in LangSmith only
# if litellm's LANGSMITH_* env vars are set (they are, see .env).
# ---------------------------------------------------------------------------

def _build_evaluator_llm():
    """Build a ragas InstructorLLM (via llm_factory) for evaluator metrics.

    Vertex AI path: litellm routes to VertexAI using Application Default
    Credentials (ADC). No API key required — uses gcloud auth.
    """
    from ragas.llms import llm_factory
    from src.config import get_settings

    settings = get_settings()

    if settings.gemini_use_vertexai:
        import os
        from litellm import OpenAI as LiteLLMClient
        os.environ["VERTEXAI_PROJECT"] = settings.google_cloud_project or ""
        os.environ["VERTEXAI_LOCATION"] = settings.google_cloud_location
        client = LiteLLMClient(api_key="placeholder")  # ADC used; key ignored for VertexAI
        return llm_factory(
            f"vertex_ai/{settings.gemini_fast_model}",
            client=client,
            adapter="litellm",
        )
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return llm_factory(settings.gemini_fast_model, client=client)


def _build_evaluator_embeddings():
    """Build ragas LiteLLMEmbeddings for embedding-dependent metrics (AnswerRelevancy)."""
    from ragas.embeddings.base import embedding_factory
    from src.config import get_settings

    settings = get_settings()

    if settings.gemini_use_vertexai:
        import os
        os.environ["VERTEXAI_PROJECT"] = settings.google_cloud_project or ""
        os.environ["VERTEXAI_LOCATION"] = settings.google_cloud_location
        return embedding_factory(
            "litellm",
            model=f"vertex_ai/{settings.gemini_embedding_model}",
        )
    else:
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        client = genai.Client(api_key=settings.gemini_api_key)
        return GoogleEmbeddings(client=client, model=settings.gemini_embedding_model)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import pandas as pd
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics.collections import (
            ContextRecall,
            ContextUtilization,
            Faithfulness,
            AnswerRelevancy,
            FactualCorrectness,
        )
    except ImportError:
        print(
            "ragas is not installed.  Run:\n"
            "  uv sync --group dev\n"
            "then retry."
        )
        sys.exit(1)

    evaluator_llm = _build_evaluator_llm()
    evaluator_embeddings = _build_evaluator_embeddings()

    # ragas.metrics.collections metrics — use batch_score() directly.
    # ragas.evaluate() only accepts ragas.metrics.base.Metric subclasses;
    # collections metrics inherit from SimpleBaseMetric (different hierarchy).
    #
    # LangSmith tracing: because evaluator_llm is a LangchainLLMWrapper, every
    # LLM call made by these metrics is automatically traced when
    # LANGCHAIN_TRACING_V2=true is set in the environment.
    #
    # Metric input signatures:
    #   Faithfulness:        user_input, response, retrieved_contexts
    #   ContextUtilization:  user_input, response, retrieved_contexts
    #   AnswerRelevancy:     user_input, response
    #   ContextRecall:       user_input, retrieved_contexts, reference
    #   FactualCorrectness:  response, reference
    faithfulness = Faithfulness(llm=evaluator_llm)
    context_utilization = ContextUtilization(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_recall = ContextRecall(llm=evaluator_llm)
    factual_correctness = FactualCorrectness(llm=evaluator_llm)

    samples: list[SingleTurnSample] = []
    golden_qa = _load_golden_qa()

    for idx, item in enumerate(golden_qa, 1):
        print(f"[{idx}/{len(golden_qa)}] Running pipeline for: {item.question!r}")
        answer, retrieved_contexts = _run_pipeline(item.question, item.company_id)

        if not answer:
            print(f"  WARNING: empty answer — skipping sample.")
            continue

        if not retrieved_contexts:
            print(f"  WARNING: no retrieved contexts — sample will score 0 on retrieval metrics.")

        sample = SingleTurnSample(
            user_input=item.question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=item.reference_answer,
        )
        samples.append(sample)
        print(f"  answer[:120]: {answer[:120]!r}")
        print(f"  contexts retrieved: {len(retrieved_contexts)}")

    if not samples:
        print(f"No samples collected from {GOLDEN_QA_PATH} — aborting evaluation.")
        sys.exit(1)

    print(f"\nRunning RAGAS evaluation on {len(samples)} sample(s) ...")

    # --- Score each metric via batch_score() ---

    # Inputs for context-aware metrics
    ctx_inputs = [
        {
            "user_input": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
        }
        for s in samples
    ]
    # Inputs for AnswerRelevancy (no retrieved_contexts needed)
    rel_inputs = [
        {"user_input": s.user_input, "response": s.response}
        for s in samples
    ]

    results_data: dict[str, list] = {
        "question": [s.user_input for s in samples],
        "answer": [s.response[:120] for s in samples],
        "contexts_count": [len(s.retrieved_contexts) for s in samples],
    }

    for metric, inputs in [
        (faithfulness, ctx_inputs),
        (context_utilization, ctx_inputs),
        (answer_relevancy, rel_inputs),
    ]:
        print(f"  Scoring {metric.name} ...")
        scores = metric.batch_score(inputs)
        results_data[metric.name] = [r.value for r in scores]

    # ContextRecall and FactualCorrectness only for samples with a reference answer
    samples_with_ref = [
        (i, s) for i, s in enumerate(samples) if s.reference is not None
    ]
    if samples_with_ref:
        ref_count = len(samples_with_ref)
        print(f"  Scoring {context_recall.name} (on {ref_count} sample(s) with reference) ...")
        recall_inputs = [
            {
                "user_input": s.user_input,
                "retrieved_contexts": s.retrieved_contexts,
                "reference": s.reference,
            }
            for _, s in samples_with_ref
        ]
        recall_scores = context_recall.batch_score(recall_inputs)
        recall_values: list[float | None] = [None] * len(samples)
        for (orig_idx, _), score_result in zip(samples_with_ref, recall_scores):
            recall_values[orig_idx] = score_result.value
        results_data[context_recall.name] = recall_values

        print(f"  Scoring {factual_correctness.name} (on {ref_count} sample(s) with reference) ...")
        fc_inputs = [
            {"response": s.response, "reference": s.reference}
            for _, s in samples_with_ref
        ]
        fc_scores = factual_correctness.batch_score(fc_inputs)
        fc_values: list[float | None] = [None] * len(samples)
        for (orig_idx, _), score_result in zip(samples_with_ref, fc_scores):
            fc_values[orig_idx] = score_result.value
        results_data[factual_correctness.name] = fc_values

    df = pd.DataFrame(results_data)
    print("\n=== RAGAS Results ===")
    print(df.to_string())

    out_path = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
