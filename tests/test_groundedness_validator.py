"""Tests for groundedness validator heuristics."""

from src.query.answer_synthesizer import Citation
from src.query.groundedness_validator import validate_groundedness


def test_groundedness_detects_supported_numeric_claim() -> None:
    answer = "Revenue was $26.0 billion in Q2 FY2024 [1]."
    citations = [
        Citation(
            chunk_id="c1",
            doc_id="d1",
            position=3,
            raw_text="NVIDIA reported revenue of $26.0 billion for Q2 FY2024.",
            confidence=0.94,
        )
    ]

    result = validate_groundedness(answer, citations)

    assert result.is_grounded is True
    assert result.unsupported_claims == []


def test_groundedness_flags_unsupported_numeric_claim() -> None:
    answer = "Revenue was $30.0 billion in Q2 FY2024 [1]."
    citations = [
        Citation(
            chunk_id="c1",
            doc_id="d1",
            position=3,
            raw_text="NVIDIA reported revenue of $26.0 billion for Q2 FY2024.",
            confidence=0.94,
        )
    ]

    result = validate_groundedness(answer, citations)

    assert result.is_grounded is False
    assert len(result.unsupported_claims) == 1
