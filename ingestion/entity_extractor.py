import json
import random
import time
from typing import Callable, List, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ingestion.chunker import Chunk


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    name: str
    type: str   # Company | Executive | Product | FinancialMetric | RiskFactor | MacroEvent | FinancialEvent
    properties: dict = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relationship: str   # CAUSED | MENTIONS | COMPETES_WITH | DEPENDS_ON | IMPACTED | HAS_EXECUTIVE
    properties: dict = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are a financial knowledge graph builder.
Extract entities and relationships from the provided financial text.

Entity types (use EXACTLY these labels):
  Company | Executive | Product | FinancialMetric | RiskFactor | MacroEvent | FinancialEvent

Relationship types (use EXACTLY these labels):
  CAUSED | MENTIONS | COMPETES_WITH | DEPENDS_ON | IMPACTED | REPORTED_BY | HAS_EXECUTIVE

Return ONLY a single valid JSON object — no prose, no markdown fences:
{{
  "entities": [{{"name": "...", "type": "...", "properties": {{}}}}],
  "relations": [{{"source": "...", "target": "...", "relationship": "...", "properties": {{}}}}]
}}

If nothing is extractable, return: {{"entities": [], "relations": []}}"""

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Extract from this financial text:\n\n{text}"),
])

_ALLOWED_ENTITY_TYPES = {
    "Company",
    "Executive",
    "Product",
    "FinancialMetric",
    "RiskFactor",
    "MacroEvent",
    "FinancialEvent",
}

_ALLOWED_RELATIONSHIPS = {
    "CAUSED",
    "MENTIONS",
    "COMPETES_WITH",
    "DEPENDS_ON",
    "IMPACTED",
    "REPORTED_BY",
    "HAS_EXECUTIVE",
}


def _retry_call(func: Callable[[], T], context: str, max_attempts: int = 3) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            print(
                f"  [entity_extractor] Retry {attempt}/{max_attempts - 1} "
                f"for {context} after error: {exc}"
            )
            time.sleep(delay)
    assert last_exc is not None
    raise RuntimeError(f"Failed {context} after {max_attempts} attempts: {last_exc}")


def _sanitize_result(result: ExtractionResult, chunk_id: str) -> ExtractionResult:
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []

    for entity in result.entities:
        if entity.type not in _ALLOWED_ENTITY_TYPES:
            print(
                f"  [entity_extractor] Skipped invalid entity type "
                f"'{entity.type}' in chunk {chunk_id}"
            )
            continue
        entities.append(entity)

    for relation in result.relations:
        if relation.relationship not in _ALLOWED_RELATIONSHIPS:
            print(
                f"  [entity_extractor] Skipped invalid relationship "
                f"'{relation.relationship}' in chunk {chunk_id}"
            )
            continue
        relations.append(relation)

    return ExtractionResult(entities=entities, relations=relations)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract_entities_and_relations(
    chunks: List[Chunk],
    llm: ChatGoogleGenerativeAI,
) -> List[ExtractionResult]:
    """
    Call Gemini on each chunk to extract entities and relationships.
    Returns one ExtractionResult per chunk (empty result on any error).

    Only called on key sections (MD&A, RISK, EARNINGS) to limit API cost.
    """
    chain = _EXTRACTION_PROMPT | llm
    results: List[ExtractionResult] = []

    for chunk in chunks:
        try:
            response = _retry_call(
                lambda: chain.invoke({"text": chunk.text}),
                context=f"Gemini extraction for chunk {chunk.id}",
            )
            content = response.content.strip()

            # Strip accidental markdown code fences
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            # Find the outermost JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end <= start:
                results.append(ExtractionResult())
                continue

            data = json.loads(content[start:end])
            raw_result = ExtractionResult(**data)
            results.append(_sanitize_result(raw_result, chunk.id))

        except Exception as exc:
            # Log but never crash the ingestion pipeline
            print(f"  [entity_extractor] Skipped chunk {chunk.id}: {exc}")
            results.append(ExtractionResult())

    return results
