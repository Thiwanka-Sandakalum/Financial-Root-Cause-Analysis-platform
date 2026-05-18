"""Answer synthesis with explicit citations using Gemini strong model."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import Settings
from src.llm.gemini import build_strong_model


class Citation(BaseModel):
    """Reference to a chunk used as evidence."""

    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Source document identifier")
    position: int | None = Field(default=None, description="Chunk position within document")
    raw_text: str = Field(description="Evidence text snippet used by the answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Evidence confidence in [0,1]")


class SynthesisOutput(BaseModel):
    """Structured grounded answer output."""

    answer: str = Field(description="Grounded answer with inline citation markers like [1], [2] immediately after every factual claim")
    citations: list[Citation] = Field(default_factory=list, description="One entry per unique inline marker, mapping [n] to its source chunk")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in [0,1]")
    reasoning: str = Field(
        description=(
            "1–2 sentences explaining which evidence supports the answer and why. "
            "Do not repeat the answer; describe the evidence-to-claim mapping."
        )
    )


def synthesize_answer(
    query: str,
    ranked_evidence: list[Document],
    settings: Settings,
    raptor_sections: list[dict[str, str | float]] | None = None,
    causal_chain: list[Document] | None = None,
    intent_type: str = "fact_lookup",
) -> SynthesisOutput:
    """Synthesize answer text from ranked evidence chunks.

    The model is instructed to only use provided evidence and to cite factual claims.
    """
    if not ranked_evidence:
        return SynthesisOutput(
            answer="I do not have enough evidence to answer this question yet.",
            citations=[],
            confidence=0.0,
            reasoning="No evidence chunks were provided to the synthesizer.",
        )

    evidence_payload = _build_evidence_payload(ranked_evidence[: settings.query_synthesis_top_k])
    section_context = _build_section_context(raptor_sections or [])
    causal_context = _build_causal_context(causal_chain or [])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                    "system",
                    (
                        "You are a financial analysis engine. "
                        "Synthesize answers strictly from the provided evidence chunks.\n\n"
                        "CITATION RULES:\n"
                        "1. Every factual claim (number, date, name, ratio) MUST be followed "
                        "immediately by an inline citation marker [n].\n"
                        "2. Only use chunk IDs that appear in the evidence list — "
                        "never invent or guess chunk IDs.\n"
                        "3. If the evidence does not contain enough information, "
                        "say so explicitly rather than speculating.\n"
                        "4. Do not introduce figures, dates, or names not present in the evidence.\n"
                        "5. Keep the answer focused; do not restate the question.\n\n"
                        "UNCERTAINTY LANGUAGE RULES:\n"
                        "6. When a causal link is implied but not directly stated in the evidence, "
                        "use hedged phrasing: 'appears linked to', 'suggests', "
                        "'evidence indicates', 'may have contributed to'.\n"
                        "7. Never assert causation unless the evidence explicitly states it.\n"
                        "8. When only one evidence source supports a claim, note the limitation: "
                        "'based on a single source' or 'according to one report'.\n\n"
                        "CONFIDENCE CALIBRATION RULES:\n"
                        "9. Set confidence HIGH (0.8–1.0) only when: multiple independent "
                        "evidence chunks corroborate the claim AND the chunks are temporally "
                        "consistent AND at least one chunk directly states the fact.\n"
                        "10. Set confidence MEDIUM (0.5–0.79) when: a single source supports "
                        "the claim, or sources are consistent but indirect.\n"
                        "11. Set confidence LOW (0.0–0.49) when: evidence is sparse, "
                        "contradictory, or only weakly related to the question.\n"
                        "12. In the reasoning field, name which chunks support the answer "
                        "and explain any evidence gaps.\n"
                        "13. For root-cause questions, distinguish graph-confirmed causal links "
                        "from textual correlation and use cautious wording for inferred causes."
                    ),
                ),
                (
                    "human",
                    (
                        "Intent type:\n{intent_type}\n\n"
                        "Question:\n{query}\n\n"
                        "High-level section context (RAPTOR):\n{section_context}\n\n"
                        "Causal links from graph traversal:\n{causal_context}\n\n"
                        "Evidence (one chunk per line, format: [n] chunk_id=... text=...):\n"
                        "{evidence_payload}\n\n"
                        "Write a grounded answer. Place [n] citation markers after every "
                        "factual claim. Map each marker to its evidence chunk in the "
                        "citations list. Use hedged language for inferred causal relationships. "
                        "Set confidence and reasoning according to the calibration rules."
                    ),
                ),
        ]
    )

    model = build_strong_model(settings).with_structured_output(SynthesisOutput)
    chain = prompt | model

    try:
        result = chain.invoke(
            {
                "query": query,
                "intent_type": intent_type,
                "section_context": section_context or "(none)",
                "causal_context": causal_context or "(none)",
                "evidence_payload": evidence_payload,
            }
        )
        if isinstance(result, SynthesisOutput):
            output = result
        elif isinstance(result, dict):
            output = SynthesisOutput.model_validate(result)
        else:
            output = SynthesisOutput.model_validate(result.model_dump())
        return _renumber_citations(output)
    except Exception as exc:
        raise RuntimeError(f"Answer synthesis failed: {exc}") from exc


def _build_evidence_payload(ranked_evidence: list[Document]) -> str:
    """Serialize evidence into a compact prompt-friendly text block."""
    lines: list[str] = []
    for idx, doc in enumerate(ranked_evidence, start=1):
        metadata = doc.metadata
        snippet = (doc.page_content or "").strip().replace("\n", " ")

        lines.append(
            (
                f"[{idx}] chunk_id={metadata.get('chunk_id')} "
                f"doc_id={metadata.get('doc_id')} "
                f"position={metadata.get('position')} "
                f"score={metadata.get('final_score', metadata.get('combined_score', 0.0)):.4f} "
                f"text={snippet}"
            )
        )

    return "\n".join(lines)


def _build_section_context(raptor_sections: list[dict[str, str | float]]) -> str:
    lines: list[str] = []
    for section in raptor_sections[:3]:
        title = str(section.get("title") or "Untitled section")
        summary = str(section.get("summary") or "")
        if summary.strip():
            lines.append(f"[Section: {title}] {summary.strip()}")
    return "\n".join(lines)


def _build_causal_context(causal_chain: list[Document]) -> str:
    links: list[str] = []
    for doc in causal_chain[:4]:
        md = doc.metadata
        event = str(md.get("causal_event") or md.get("upstream_event") or "").strip()
        effect = str(md.get("affected_metric") or md.get("downstream_event") or "").strip()
        if event and effect:
            links.append(f"- {event} -> {effect}")
    return "\n".join(links)


def _renumber_citations(output: SynthesisOutput) -> SynthesisOutput:
    """Renumber inline citation markers so they are sequential and match the citations list.

    The LLM may emit arbitrary marker numbers (e.g. [3][7][16]) that exceed
    the length of the returned citations list.  This function:

    1. Collects all unique marker numbers in order of first appearance.
    2. Assigns them new sequential indices: first seen → [1], second → [2], …
    3. Replaces every old [N] in the answer with the new sequential marker.
    4. Trims the citations list to ``min(len(old_markers), len(citations))`` so
       the list length always matches the highest marker number.

    After this call ``max([N] in answer) == len(output.citations)`` is guaranteed
    (or there are no markers at all).
    """
    answer = output.answer
    old_markers: list[int] = list(dict.fromkeys(int(m) for m in re.findall(r"\[(\d+)\]", answer)))

    if not old_markers:
        return output

    # Build old → new mapping (order of first appearance → 1-based sequential)
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(old_markers, start=1)}

    # Replace markers in the answer.  Process in descending order of the old
    # marker number so that e.g. [12] is replaced before [1] to avoid accidental
    # substring collisions (replacing [1] inside [12] before [12] is handled).
    new_answer = answer
    for old in sorted(old_to_new, reverse=True):
        new_answer = re.sub(rf"\[{old}\]", f"[{old_to_new[old]}]", new_answer)

    # Keep only as many citations as there are unique markers (in their original
    # list order — the LLM fills citations[0] for its first reference, etc.)
    new_citations = list(output.citations[: len(old_markers)])

    return SynthesisOutput(
        answer=new_answer,
        citations=new_citations,
        confidence=output.confidence,
        reasoning=output.reasoning,
    )
