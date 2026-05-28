from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return parsed


@dataclass(frozen=True)
class GuardrailThresholds:
    quality_min_evidence_count: int = 2
    quality_min_avg_score: float = 0.35
    claim_support_block_threshold: float = 0.40
    claim_support_downgrade_threshold: float = 0.80
    citation_coverage_high_conf_floor: float = 0.50


GUARDRAIL_THRESHOLDS = GuardrailThresholds(
    quality_min_evidence_count=_env_int("RA_QUALITY_MIN_EVIDENCE_COUNT", 2, minimum=1),
    quality_min_avg_score=_env_float("RA_QUALITY_MIN_AVG_SCORE", 0.35),
    claim_support_block_threshold=_env_float("RA_CLAIM_SUPPORT_BLOCK_THRESHOLD", 0.40),
    claim_support_downgrade_threshold=_env_float("RA_CLAIM_SUPPORT_DOWNGRADE_THRESHOLD", 0.80),
    citation_coverage_high_conf_floor=_env_float("RA_CITATION_COVERAGE_HIGH_CONF_FLOOR", 0.50),
)
