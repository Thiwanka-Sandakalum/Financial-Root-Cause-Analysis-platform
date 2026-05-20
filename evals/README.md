# RootAlpha Evaluations

This project now uses a uv-first workflow.

## LangSmith Observability (Ragas Metrics Tracing)

Set the standard LangSmith environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=rootalpha-eval
```

Then run eval with tracing enabled:

```bash
uv run rootalpha-eval --trace --testset evals/datasets/eval_testset_2.json
```

Notes:

- The evaluator normalizes both `LANGCHAIN_TRACING_V2` and `LANGSMITH_TRACING`.
- `--trace` and `--no-trace` override env vars for one run.
- Metric runs appear in LangSmith tracing for the configured project.

## Run evaluation

With default built-in questions:

```bash
uv run rootalpha-eval
```

With a dataset file:

```bash
uv run rootalpha-eval --testset evals/datasets/eval_testset_2.json
```

Disable tracing for local smoke runs:

```bash
LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false LANGCHAIN_PROJECT= uv run rootalpha-eval --testset evals/datasets/eval_testset_2.json
```

Results are written to:

- `evals/results/eval_results.csv`
