from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.config import Settings
from src.ingestion.models import ChunkRecord, ExtractionRecord
from src.llm.gemini import build_fast_model


# ---------------------------------------------------------------------------
# Source-type profiles
# ---------------------------------------------------------------------------
# Different document types need different extractors and different LLM instructions.
# A profile declares which of the three extractors (entity/event/metric) should run
# and what instruction text to pass to each.  Unknown source_type values fall back
# to _DEFAULT_PROFILE which runs entity extraction only with generic instructions —
# the safest option for content whose structure we don't know in advance.

@dataclass(frozen=True)
class SourceProfile:
    run_entities: bool
    run_events: bool
    run_metrics: bool
    entity_instruction: str
    event_instruction: str
    metric_instruction: str


_FIN_ENTITY = (
    "Extract entities from financial text. Return only entities explicitly present in text. "
    "Prefer entity types: Company, Executive, Product, RiskFactor, Regulator, Other."
)
_FIN_EVENT = (
    "Extract financial events from text. Return only events explicitly present in text. "
    "Prefer event types: EarningsCall, ProductLaunch, RegulatoryAction, Acquisition, Merger, "
    "Partnership, Filing, Appointment."
)
_FIN_METRIC = (
    "Extract financial metrics from text. Return only metrics explicitly present in text. "
    "Prefer metric types: Revenue, EPS, Margin, Guidance, CashFlow, EBITDA, NetIncome, "
    "OperatingExpense, GrowthRate."
)
_NEWS_ENTITY = (
    "Extract named entities from news text. Return only entities explicitly present in text. "
    "Prefer entity types: Person, Organization, Location, Topic, Product, Government, Other."
)
_NEWS_EVENT = (
    "Extract notable events mentioned in news text. Return only events explicitly present in text. "
    "Prefer event types: Announcement, Acquisition, Lawsuit, PolicyChange, Appointment, "
    "ProductLaunch, Merger, Other."
)
_GENERIC_ENTITY = (
    "Extract named entities from text. Return only entities explicitly present in text. "
    "Use appropriate entity types such as: Person, Organization, Location, Product, Concept, Other."
)

# Full financial extraction
_FULL_FIN = SourceProfile(True, True, True, _FIN_ENTITY, _FIN_EVENT, _FIN_METRIC)
# Entities + events only (press releases announce things but rarely contain structured metrics)
_ENT_EVT_FIN = SourceProfile(True, True, False, _FIN_ENTITY, _FIN_EVENT, "")
# News: entities + events with news-oriented entity types, no metric extraction
_NEWS = SourceProfile(True, True, False, _NEWS_ENTITY, _NEWS_EVENT, "")
# Generic fallback for unknown source types: entity-only with generic instructions
_DEFAULT_PROFILE = SourceProfile(True, False, False, _GENERIC_ENTITY, "", "")

_SOURCE_PROFILES: dict[str, SourceProfile] = {
    # Financial documents — run all three extractors
    "annual_report": _FULL_FIN,
    "earnings_release": _FULL_FIN,
    "earnings_report": _FULL_FIN,
    "financial_report": _FULL_FIN,
    "research_report": _FULL_FIN,
    "analyst_report": _FULL_FIN,
    "regulatory_filing": _FULL_FIN,
    "sec_filing": _FULL_FIN,
    "10k": _FULL_FIN,
    "10q": _FULL_FIN,
    "8k": _FULL_FIN,
    # Press releases / corporate announcements — entities + events, no metrics
    "press_release": _ENT_EVT_FIN,
    "company_announcement": _ENT_EVT_FIN,
    "investor_update": _ENT_EVT_FIN,
    # News — entities + events, news-oriented instructions
    "news_article": _NEWS,
    "news": _NEWS,
    # Unknown / generic upload — entity only, generic instructions
    "uploaded_document": _DEFAULT_PROFILE,
}


# Keyword fragments checked against the normalised source_type string when an
# exact key lookup fails.  Longer / more-specific phrases are listed first so
# they win over shorter, broader keywords (e.g. "earnings_release" before "earnings").
_PROFILE_KEYWORDS: list[tuple[str, SourceProfile]] = [
    # Financial results / earnings docs → full extraction
    ("annual_report", _FULL_FIN),
    ("10k", _FULL_FIN),
    ("10q", _FULL_FIN),
    ("8k", _FULL_FIN),
    ("earnings_release", _FULL_FIN),
    ("earnings_report", _FULL_FIN),
    ("financial_result", _FULL_FIN),
    ("financial_report", _FULL_FIN),
    ("quarterly_result", _FULL_FIN),
    ("quarterly_report", _FULL_FIN),
    ("fiscal", _FULL_FIN),
    ("revenue", _FULL_FIN),
    ("earnings", _FULL_FIN),
    ("income", _FULL_FIN),
    ("financial", _FULL_FIN),
    ("research_report", _FULL_FIN),
    ("analyst", _FULL_FIN),
    ("regulatory_filing", _FULL_FIN),
    ("sec_filing", _FULL_FIN),
    ("filing", _FULL_FIN),
    # Press releases / announcements → entities + events
    ("press_release", _ENT_EVT_FIN),
    ("announcement", _ENT_EVT_FIN),
    ("investor_update", _ENT_EVT_FIN),
    ("company_released", _ENT_EVT_FIN),
    ("public_statement", _ENT_EVT_FIN),
    # News → entities + events with news-oriented instructions
    ("news_article", _NEWS),
    ("news", _NEWS),
]


def _get_source_profile(source_type: str) -> SourceProfile:
    """Return the extraction profile for a source_type string.

    First tries an exact lookup after normalising the string.  If that misses
    (e.g. the caller passed a free-form title like
    'Financial Results for Second Quarter Fiscal 2024'), falls back to
    substring keyword matching so common phrases still resolve correctly.
    Unknown types that don't match any keyword get _DEFAULT_PROFILE.
    """
    normalised = source_type.strip().lower().replace(" ", "_").replace("-", "_")
    if normalised in _SOURCE_PROFILES:
        return _SOURCE_PROFILES[normalised]
    # Keyword fallback — check each phrase against the full normalised string
    for keyword, profile in _PROFILE_KEYWORDS:
        if keyword in normalised:
            return profile
    return _DEFAULT_PROFILE


class EntityItem(BaseModel):
    type: Literal[
        "Company", "Executive", "Person", "Product", "RiskFactor",
        "Regulator", "Organization", "Location", "Topic", "Government", "Other",
    ] = Field(description="Entity category — choose the closest match")
    canonical_name: str = Field(description="Normalized display name (e.g. 'Apple Inc' not 'AAPL')")
    raw_text: str = Field(description="Verbatim substring from the source text")
    confidence: float = Field(description="1.0=explicitly stated, 0.5=inferred, 0.0=uncertain")


class EntityExtractionResponse(BaseModel):
    entities: list[EntityItem] = Field(default_factory=list)


class EventItem(BaseModel):
    type: Literal[
        "EarningsCall", "ProductLaunch", "RegulatoryAction", "Acquisition", "Merger",
        "Partnership", "Filing", "Appointment", "Announcement", "Lawsuit", "PolicyChange", "Other",
    ] = Field(description="Event category — choose the closest match")
    canonical_name: str = Field(description="Normalized event name")
    raw_text: str = Field(description="Verbatim substring from the source text")
    confidence: float = Field(description="1.0=explicitly stated, 0.5=inferred, 0.0=uncertain")


class EventExtractionResponse(BaseModel):
    events: list[EventItem] = Field(default_factory=list)


class MetricItem(BaseModel):
    type: Literal[
        "Revenue", "EPS", "Margin", "Guidance", "CashFlow",
        "EBITDA", "NetIncome", "OperatingExpense", "GrowthRate", "Other",
    ] = Field(description="Metric category — choose the closest match")
    canonical_name: str = Field(description="Normalized metric identifier (e.g. 'Revenue Q3 FY2024')")
    raw_text: str = Field(description="Verbatim substring from the source text")
    confidence: float = Field(description="1.0=explicitly stated, 0.5=inferred, 0.0=uncertain")
    value: float | None = Field(
        default=None,
        description="Numeric value if explicitly stated (e.g. 35100 for $35.1B; store in millions)",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement (e.g. 'USD millions', '%', 'basis points')",
    )
    yoy_change: float | None = Field(
        default=None,
        description="Year-over-year percentage change if explicitly stated (e.g. 122.4 for +122.4%)",
    )


class MetricExtractionResponse(BaseModel):
    metrics: list[MetricItem] = Field(default_factory=list)


def extract_entities_from_chunks(
    chunks: list[ChunkRecord],
    settings: Settings,
    source_type: str = "uploaded_document",
) -> list[ExtractionRecord]:
    """Extract entity mentions from chunks using model structured output.

    The instruction and whether extraction runs at all is controlled by the
    SourceProfile for the given source_type.  Unknown source types use a
    generic entity-only profile so the pipeline never fails on unexpected
    document types.
    """
    profile = _get_source_profile(source_type)
    if not profile.run_entities:
        return []

    return _extract_with_schema(
        chunks=chunks,
        settings=settings,
        response_schema=EntityExtractionResponse,
        response_field="entities",
        instruction=profile.entity_instruction,
        fallback_type="Entity",
    )


def extract_events_from_chunks(
    chunks: list[ChunkRecord],
    settings: Settings,
    source_type: str = "uploaded_document",
) -> list[ExtractionRecord]:
    profile = _get_source_profile(source_type)
    if not profile.run_events:
        return []

    return _extract_with_schema(
        chunks=chunks,
        settings=settings,
        response_schema=EventExtractionResponse,
        response_field="events",
        instruction=profile.event_instruction,
        fallback_type="Event",
    )


def extract_metrics_from_chunks(
    chunks: list[ChunkRecord],
    settings: Settings,
    source_type: str = "uploaded_document",
) -> list[ExtractionRecord]:
    profile = _get_source_profile(source_type)
    if not profile.run_metrics:
        return []

    return _extract_with_schema(
        chunks=chunks,
        settings=settings,
        response_schema=MetricExtractionResponse,
        response_field="metrics",
        instruction=profile.metric_instruction,
        fallback_type="Metric",
    )


def to_state_dicts(records: list[ExtractionRecord]) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "canonical_name": record.canonical_name,
            "raw_text": record.raw_text,
            "confidence": record.confidence,
            "source_chunk_id": record.source_chunk_id,
            "metadata": record.metadata,
        }
        for record in records
    ]


def _extract_with_schema(
    chunks: list[ChunkRecord],
    settings: Settings,
    response_schema: type[BaseModel],
    response_field: str,
    instruction: str,
    fallback_type: str,
) -> list[ExtractionRecord]:
    """Extract using per-section batching (Fix 5) instead of per-chunk.

    Groups chunks by section_title, builds section context (up to 3000 tokens),
    then makes one LLM call per section instead of per chunk.
    This reduces ~100+ calls on a 57-page document to ~8-12 calls.
    """
    if not chunks:
        return []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You extract structured information from financial and business text.\n"
                    "RULES:\n"
                    "1. Return only items explicitly present in the provided text — never fabricate.\n"
                    "2. If nothing matches the requested category, return an empty list.\n"
                    "3. confidence: 1.0=explicitly stated, 0.5=inferred from context, 0.0=uncertain.\n"
                    "4. canonical_name must be normalised (e.g. 'Apple Inc.' → 'Apple Inc').\n"
                    "5. raw_text must be a verbatim substring from the provided text."
                ),
            ),
            ("human", "{instruction}\n\nText:\n{section_text}"),
        ]
    )

    model = build_fast_model(settings).with_structured_output(response_schema)
    chain = prompt | model

    # Group chunks by section for batch extraction
    sections = _group_chunks_by_section(chunks)

    extraction_inputs = []
    section_chunk_map = {}  # section_text → list[ChunkRecord]
    
    for section_title, section_chunks in sections.items():
        # Build section context: join all chunks up to 3000 tokens
        section_texts = []
        total_tokens = 0
        for chunk in section_chunks:
            chunk_tokens = chunk.token_count or len(chunk.text.split())
            if total_tokens + chunk_tokens <= 3000:
                section_texts.append(chunk.text)
                total_tokens += chunk_tokens
        
        section_text = "\n\n".join(section_texts)
        
        extraction_inputs.append({
            "instruction": instruction,
            "section_text": section_text,
        })
        section_chunk_map[section_text] = section_chunks

    responses: list[Any] = []
    run_config = cast(
        RunnableConfig,
        {"max_concurrency": settings.ingestion_extraction_max_concurrency},
    )
    
    # Batch the section-level extractions
    for batch_inputs in _iter_batches(
        extraction_inputs,
        settings.ingestion_extraction_batch_size,
    ):
        responses.extend(chain.batch(batch_inputs, config=run_config))

    # Convert responses back to per-chunk records for downstream consistency
    records: list[ExtractionRecord] = []
    for input_dict, response in zip(extraction_inputs, responses, strict=False):
        section_text = input_dict["section_text"]
        section_chunks = section_chunk_map.get(section_text, [])
        
        items = getattr(response, response_field, [])
        
        # Distribute extracted items to chunks in this section
        # (assign all items to first chunk of section for simplicity)
        if section_chunks and items:
            primary_chunk = section_chunks[0]
            records.extend(
                _normalize_items(
                    items,
                    primary_chunk,
                    settings,
                    fallback_type=fallback_type,
                )
            )

    return records



def _iter_batches(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# Fix 5: Section-level batching helper
def _group_chunks_by_section(chunks: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    """Group chunks by section_title to enable section-level extraction.

    Returns dict: section_title → list of ChunkRecords in that section
    """
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        section = chunk.section_title or "unknown"
        grouped.setdefault(section, []).append(chunk)
    return grouped


def _build_entity_id(entity_type: str, canonical_name: str) -> str:
    # Key on canonical_name only so the same real-world entity is not duplicated
    # when the LLM assigns different types across chunks (e.g. "Company" vs "Organization").
    key = canonical_name.strip().lower()
    return f"ent_{sha1(key.encode('utf-8')).hexdigest()}"


def _normalize_items(
    items: list[Any],
    chunk: ChunkRecord,
    settings: Settings,
    fallback_type: str,
) -> list[ExtractionRecord]:
    records: list[ExtractionRecord] = []
    for item in items:
        entity_type = (item.type or fallback_type).strip() or fallback_type
        canonical_name = item.canonical_name.strip()
        raw_text = item.raw_text.strip()
        if not canonical_name or not raw_text:
            continue

        safe_confidence = max(0.0, min(1.0, float(item.confidence)))
        entity_id = _build_entity_id(entity_type, canonical_name)
        item_metadata: dict[str, Any] = {
            "source_doc_id": chunk.doc_id,
            "company_id": chunk.company_id,
            "is_low_confidence": safe_confidence < settings.extraction_confidence_threshold,
        }
        # Carry structured numeric fields from MetricItem when present
        if hasattr(item, "value") and item.value is not None:
            item_metadata["value"] = item.value
        if hasattr(item, "unit") and item.unit is not None:
            item_metadata["unit"] = item.unit
        if hasattr(item, "yoy_change") and item.yoy_change is not None:
            item_metadata["yoy_change"] = item.yoy_change
        records.append(
            ExtractionRecord(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                raw_text=raw_text,
                confidence=safe_confidence,
                source_chunk_id=chunk.chunk_id,
                metadata=item_metadata,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Causal link extraction — Event-[:CAUSED]->Metric
# ---------------------------------------------------------------------------


class CausalLinkItem(BaseModel):
    from_canonical_name: str = Field(
        description="Event canonical name (the cause); must match exactly from the EVENT list"
    )
    to_canonical_name: str = Field(
        description="Metric canonical name (the effect); must match exactly from the METRIC list"
    )
    confidence: float = Field(
        description="1.0=explicitly stated causal link, 0.5=implied by context, 0.0=uncertain"
    )


class CausalLinksResponse(BaseModel):
    links: list[CausalLinkItem] = Field(default_factory=list)


def extract_causal_links_from_chunks(
    chunks: list[ChunkRecord],
    event_records: list[ExtractionRecord],
    metric_records: list[ExtractionRecord],
    settings: Settings,
    source_type: str = "uploaded_document",
) -> list[dict[str, Any]]:
    """Extract (Event)-[:CAUSED]->(Metric) links within the same chunk.

    Only runs for source types with both event and metric extraction profiles.
    Returns a deduplicated list of {event_id, metric_id, confidence} dicts.
    """
    profile = _get_source_profile(source_type)
    if not profile.run_events or not profile.run_metrics:
        return []

    # Group records by chunk
    events_by_chunk: dict[str, list[ExtractionRecord]] = {}
    for r in event_records:
        events_by_chunk.setdefault(r.source_chunk_id, []).append(r)

    metrics_by_chunk: dict[str, list[ExtractionRecord]] = {}
    for r in metric_records:
        metrics_by_chunk.setdefault(r.source_chunk_id, []).append(r)

    chunk_map = {c.chunk_id: c for c in chunks}
    causal_inputs: list[dict[str, str]] = []

    for chunk_id, chunk_events in events_by_chunk.items():
        chunk_metrics = metrics_by_chunk.get(chunk_id, [])
        if not chunk_metrics:
            continue
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        event_list = "\n".join(
            f"  EVENT: {r.canonical_name}" for r in chunk_events
        )
        metric_list = "\n".join(
            f"  METRIC: {r.canonical_name}" for r in chunk_metrics
        )
        causal_inputs.append({
            "chunk_text": chunk.text,
            "event_list": event_list,
            "metric_list": metric_list,
        })

    if not causal_inputs:
        return []

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You identify direct causal relationships between events and financial metrics.\n"
                "RULES:\n"
                "1. Only assert CAUSED when the text explicitly states the event caused or drove the metric change.\n"
                "2. Use confidence=0.5 when causation is implied but not directly stated.\n"
                "3. Return an empty list when no causal link is explicit in the text.\n"
                "4. from_canonical_name must match exactly one name from the EVENT list.\n"
                "5. to_canonical_name must match exactly one name from the METRIC list."
            ),
        ),
        (
            "human",
            (
                "Text:\n{chunk_text}\n\n"
                "Events extracted:\n{event_list}\n\n"
                "Metrics extracted:\n{metric_list}\n\n"
                "Which events CAUSED which metrics to change? Return only explicit causal relationships."
            ),
        ),
    ])

    model = build_fast_model(settings).with_structured_output(CausalLinksResponse)
    chain = prompt | model
    run_config = cast(RunnableConfig, {"max_concurrency": settings.ingestion_extraction_max_concurrency})

    # Lookup maps: canonical_name.lower() -> entity_id (deterministic SHA1)
    event_id_map = {r.canonical_name.strip().lower(): r.entity_id for r in event_records}
    metric_id_map = {r.canonical_name.strip().lower(): r.entity_id for r in metric_records}

    raw_results: list[dict[str, Any]] = []
    for batch in _iter_batches(causal_inputs, settings.ingestion_extraction_batch_size):
        for response in chain.batch(batch, config=run_config):
            for link in getattr(response, "links", []):
                from_id = event_id_map.get(link.from_canonical_name.strip().lower())
                to_id = metric_id_map.get(link.to_canonical_name.strip().lower())
                if from_id and to_id:
                    conf = max(0.0, min(1.0, float(link.confidence)))
                    if conf >= settings.extraction_confidence_threshold:
                        raw_results.append({"event_id": from_id, "metric_id": to_id, "confidence": conf})

    # Deduplicate — keep highest confidence per (event_id, metric_id) pair
    best: dict[tuple[str, str], float] = {}
    for r in raw_results:
        key = (r["event_id"], r["metric_id"])
        if key not in best or r["confidence"] > best[key]:
            best[key] = r["confidence"]

    return [{"event_id": eid, "metric_id": mid, "confidence": conf} for (eid, mid), conf in best.items()]


# ---------------------------------------------------------------------------
# Event causal chain extraction — Event-[:LED_TO]->Event
# ---------------------------------------------------------------------------


class EventCausalLinkItem(BaseModel):
    from_canonical_name: str = Field(
        description="Upstream event (the cause); must match exactly from the EVENT list"
    )
    to_canonical_name: str = Field(
        description="Downstream event (the effect); must match exactly from the EVENT list"
    )
    confidence: float = Field(
        description="1.0=explicitly stated, 0.5=implied, 0.0=uncertain"
    )


class EventCausalLinksResponse(BaseModel):
    links: list[EventCausalLinkItem] = Field(default_factory=list)


def extract_event_causal_chains_from_chunks(
    chunks: list[ChunkRecord],
    event_records: list[ExtractionRecord],
    settings: Settings,
    source_type: str = "uploaded_document",
) -> list[dict[str, Any]]:
    """Extract (Event)-[:LED_TO]->(Event) causal chains within the same chunk.

    Requires at least 2 events in a chunk to run. Returns a deduplicated list
    of {from_event_id, to_event_id, confidence} dicts.
    """
    profile = _get_source_profile(source_type)
    if not profile.run_events:
        return []

    events_by_chunk: dict[str, list[ExtractionRecord]] = {}
    for r in event_records:
        events_by_chunk.setdefault(r.source_chunk_id, []).append(r)

    chunk_map = {c.chunk_id: c for c in chunks}
    chain_inputs: list[dict[str, str]] = []

    for chunk_id, chunk_events in events_by_chunk.items():
        if len(chunk_events) < 2:
            continue
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        event_list = "\n".join(f"  EVENT: {r.canonical_name}" for r in chunk_events)
        chain_inputs.append({"chunk_text": chunk.text, "event_list": event_list})

    if not chain_inputs:
        return []

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You identify causal ordering between financial or business events.\n"
                "RULES:\n"
                "1. Only assert LED_TO when the text explicitly states one event led to or caused another.\n"
                "2. from_canonical_name is the upstream (cause) event.\n"
                "3. to_canonical_name is the downstream (effect) event.\n"
                "4. Both names must match exactly from the EVENT list.\n"
                "5. Never create a self-loop (from == to). Return empty list when no chain is stated."
            ),
        ),
        (
            "human",
            (
                "Text:\n{chunk_text}\n\n"
                "Events extracted:\n{event_list}\n\n"
                "Which events LED TO other events? Return explicit causal event chains only."
            ),
        ),
    ])

    model = build_fast_model(settings).with_structured_output(EventCausalLinksResponse)
    chain_runner = prompt | model
    run_config = cast(RunnableConfig, {"max_concurrency": settings.ingestion_extraction_max_concurrency})

    event_id_map = {r.canonical_name.strip().lower(): r.entity_id for r in event_records}

    raw_results: list[dict[str, Any]] = []
    for batch in _iter_batches(chain_inputs, settings.ingestion_extraction_batch_size):
        for response in chain_runner.batch(batch, config=run_config):
            for link in getattr(response, "links", []):
                from_id = event_id_map.get(link.from_canonical_name.strip().lower())
                to_id = event_id_map.get(link.to_canonical_name.strip().lower())
                if from_id and to_id and from_id != to_id:
                    conf = max(0.0, min(1.0, float(link.confidence)))
                    if conf >= settings.extraction_confidence_threshold:
                        raw_results.append({"from_event_id": from_id, "to_event_id": to_id, "confidence": conf})

    # Deduplicate
    best: dict[tuple[str, str], float] = {}
    for r in raw_results:
        key = (r["from_event_id"], r["to_event_id"])
        if key not in best or r["confidence"] > best[key]:
            best[key] = r["confidence"]

    return [{"from_event_id": fid, "to_event_id": tid, "confidence": conf} for (fid, tid), conf in best.items()]
