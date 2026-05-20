from __future__ import annotations

from evals.ragas_eval import main


def eval_default() -> None:
    """Run quick eval with built-in questions."""
    main([])


def eval_reference() -> None:
    """Run eval against the reference testset."""
    main(["--testset", "evals/datasets/eval_testset_2.json"])


def eval_reference_trace() -> None:
    """Run reference eval with tracing forced on."""
    main(["--trace", "--testset", "evals/datasets/eval_testset_2.json"])
