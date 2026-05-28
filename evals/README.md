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

## Domain Gate Diagnostics in CSV/Report

Evaluation exports now include domain-control diagnostics:

- `domain_status`
- `domain_gate_passed`
- `domain_confidence`
- `domain_reason`
- `domain_signals`
- `domain_adjudication_used`

To compute labeled false-block/false-allow metrics in the markdown report,
add `expected_domain` to each test case in your dataset:

```json
{
	"question": "What are the anticipated GAAP and Non-GAAP gross margins projected in the Q4 FY2025 outlook?",
	"reference": "GAAP and non-GAAP gross margins are expected to be 74.4% and 75.0%, respectively, plus or minus 50 basis points.",
	"ticker": "NVDA",
	"period": null,
	"expected_domain": "financial_supported"
}
```

Supported labels for `expected_domain`:

- `financial_supported`
- `non_financial`

If `expected_domain` is omitted, eval will infer defaults:

- `category` in `robustness`, `non_financial`, `out_of_scope` -> `non_financial`
- otherwise -> `financial_supported`

Additional domain QA artifact generated on eval runs:

- `evals/results/domain_gate_summary.json`

This summary includes:

- `accuracy`
- `precision_non_financial`
- `recall_non_financial`
- `f1_non_financial`
- `false_block_rate`
- `false_allow_rate`
