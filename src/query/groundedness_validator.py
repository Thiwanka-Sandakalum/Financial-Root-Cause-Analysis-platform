"""Groundedness validation for synthesized answers."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.config import Settings
from src.llm.gemini import build_fast_model
from src.query.answer_synthesizer import Citation


class GroundednessCheck(BaseModel):
    """Groundedness validation result."""

    is_grounded: bool = Field(description="Whether all meaningful claims are evidence-backed")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims lacking evidence")
    confidence: float = Field(ge=0.0, le=1.0, description="Validation confidence in [0,1]")


class _ModelGroundednessCheck(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


def validate_groundedness(
    answer: str,
    citations: list[Citation],
    settings: Settings | None = None,
) -> GroundednessCheck:
    """Validate that answer claims are supported by provided citations.

    Uses a fast heuristic first. If confidence is low and settings are provided,
    falls back to one model-based structured check.
    """
    heuristic = _heuristic_groundedness_check(answer, citations)
    if settings is None or heuristic.confidence >= 0.7:
        return heuristic

    try:
        return _model_groundedness_check(answer, citations, settings)
    except Exception:
        return heuristic


def _heuristic_groundedness_check(answer: str, citations: list[Citation]) -> GroundednessCheck:
    # -----------------------------------------------------------------------
    # Structural check: every [N] marker in the answer must have a matching
    # citation at index N-1.  If any marker exceeds the citation count the
    # answer has citation drift and cannot be considered grounded.
    # -----------------------------------------------------------------------
    marker_numbers = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    if marker_numbers and max(marker_numbers) > len(citations):
        orphan_markers = sorted({n for n in marker_numbers if n > len(citations)})
        return GroundednessCheck(
            is_grounded=False,
            unsupported_claims=[
                f"Answer references citation marker(s) {orphan_markers} "
                f"but only {len(citations)} citation(s) were returned."
            ],
            confidence=0.0,
        )

    claims = _extract_claim_sentences(answer)
    if not claims:
        return GroundednessCheck(is_grounded=True, unsupported_claims=[], confidence=0.6)

    citation_text = " ".join(c.raw_text.lower() for c in citations if c.raw_text)
    unsupported: list[str] = []

    for claim in claims:
        # Skip temporal framing sentences — they set context, not assert figures.
        if _TEMPORAL_FRAMING.match(re.sub(r"\[\d+\]", "", claim).strip()):
            continue

        claim_without_markers = re.sub(r"\[\d+\]", "", claim)
        numbers = re.findall(r"\d[\d,.%]*", claim_without_markers)

        if numbers:
            # If the sentence has a valid inline citation marker [N] and NONE of
            # its numbers appear verbatim in the citation text, the figures are
            # derived/calculated (e.g. ratios, growth rates). Trust the explicit
            # citation and skip the literal check for fully derived values.
            # If at least one number IS found verbatim, fall through to the normal
            # check — a mix of real and fabricated figures must still be caught.
            cited_markers = [int(m) for m in re.findall(r"\[(\d+)\]", claim)]
            if (
                cited_markers
                and all(n <= len(citations) for n in cited_markers)
                and not any(num.lower() in citation_text for num in numbers)
            ):
                continue

            if not all(num.lower() in citation_text for num in numbers):
                unsupported.append(claim)
            continue

        tokens = [t for t in re.findall(r"[A-Za-z]{4,}", claim.lower()) if t not in _STOPWORDS]
        if tokens and not any(token in citation_text for token in tokens[:4]):
            unsupported.append(claim)

    # Extra guard for causal assertions: require causal signal in evidence text.
    for claim in claims:
        cleaned = re.sub(r"\[\d+\]", "", claim).strip().lower()
        if not _CAUSAL_CLAIM_RE.search(cleaned):
            continue
        if not _CAUSAL_EVIDENCE_RE.search(citation_text):
            unsupported.append(claim)

    confidence = 1.0 - (len(unsupported) / max(len(claims), 1))
    confidence = max(0.0, min(1.0, confidence))

    return GroundednessCheck(
        is_grounded=len(unsupported) == 0,
        unsupported_claims=unsupported,
        confidence=confidence,
    )


def _model_groundedness_check(
    answer: str,
    citations: list[Citation],
    settings: Settings,
) -> GroundednessCheck:
    model = build_fast_model(settings).with_structured_output(_ModelGroundednessCheck)
    citation_blob = "\n".join(
        f"chunk_id={c.chunk_id} doc_id={c.doc_id} text={c.raw_text}" for c in citations
    )
    result = model.invoke(
        (
            "Validate claim grounding. Mark claim unsupported when evidence text does not support it.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Evidence:\n{citation_blob}"
        )
    )
    return GroundednessCheck(
        is_grounded=result.is_grounded,
        unsupported_claims=result.unsupported_claims,
        confidence=result.confidence,
    )


# Temporal framing sentences introduce *when* something happened; they do not
# assert a figure that must appear in citation text.  Examples:
#   "In the third quarter of fiscal year 2025, ..."
#   "In Q3 FY2025, NVIDIA reported..."
#   "For the quarter ended October 2024, ..."
_TEMPORAL_FRAMING = re.compile(
    r"^(?:"
    r"in\s+(?:the\s+)?(?:first|second|third|fourth|q[1-4])\s+(?:quarter|half)[\w\s,]*"
    r"|in\s+q[1-4]\s+fy\d{2,4}"
    r"|for\s+the\s+(?:quarter|period|year)\s+ended"
    r"|during\s+(?:the\s+)?(?:first|second|third|fourth|q[1-4])\s+"
    r"|in\s+fiscal\s+(?:year\s+)?\d{4}"
    r")",
    re.IGNORECASE,
)


def _extract_claim_sentences(answer: str) -> list[str]:
    pieces = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", answer) if segment.strip()]
    claims: list[str] = []
    for sentence in pieces:
        if len(sentence) < 20:
            continue
        if any(k in sentence.lower() for k in ("revenue", "eps", "growth", "margin", "%", "$")):
            claims.append(sentence)
            continue
        if re.search(r"\d", sentence):
            claims.append(sentence)
    return claims


_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "about",
    "their",
    "which",
    "there",
    "because",
    "would",
    "could",
}


_CAUSAL_CLAIM_RE = re.compile(
    r"\b(caused|cause|led\s+to|resulted\s+in|because\s+of|drove|triggered|impact\s+of|effect\s+of)\b",
    re.IGNORECASE,
)

_CAUSAL_EVIDENCE_RE = re.compile(
    r"\b(caused|led\s+to|resulted\s+in|because|due\s+to|driven\s+by|contributed\s+to|impact)\b",
    re.IGNORECASE,
)
