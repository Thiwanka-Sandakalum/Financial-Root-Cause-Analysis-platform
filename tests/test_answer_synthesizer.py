"""Tests for synthesis output schemas and evidence payload builder."""

from langchain_core.documents import Document

from src.query.answer_synthesizer import Citation, SynthesisOutput, _build_evidence_payload


def test_synthesis_models_validate() -> None:
    citation = Citation(
        chunk_id="c1",
        doc_id="d1",
        position=2,
        raw_text="Revenue increased 15% year over year.",
        confidence=0.9,
    )

    output = SynthesisOutput(
        answer="Revenue increased 15% [1].",
        citations=[citation],
        confidence=0.87,
        reasoning="Claim matches reported figure in earnings excerpt.",
    )

    assert output.citations[0].chunk_id == "c1"
    assert output.confidence > 0.8


def test_build_evidence_payload_contains_core_fields() -> None:
    docs = [
        Document(
            page_content="Quarterly revenue reached 26.0B.",
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "position": 4,
                "final_score": 0.93,
            },
        )
    ]

    payload = _build_evidence_payload(docs)

    assert "chunk_id=c1" in payload
    assert "doc_id=d1" in payload
    assert "score=0.9300" in payload
