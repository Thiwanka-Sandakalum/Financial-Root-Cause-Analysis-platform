from __future__ import annotations

import re
from typing import Any

from neo4j import Driver, Query

from src.config import Settings
from src.ingestion.classifier import DetectedMetadata
from src.ingestion.models import ChunkRecord, ExtractionRecord, ParsedDocument

# ---------------------------------------------------------------------------
# Entity deduplication helpers (Fix 6)
# ---------------------------------------------------------------------------

_CORP_SUFFIXES = re.compile(
    r"\s+(inc\.?|corp\.?|corporation|ltd\.?|llc\.?|plc\.?|co\.?|group|holdings?)\s*$",
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Normalize a name by removing corporate suffixes and uppercasing."""
    normalized = _CORP_SUFFIXES.sub("", name.strip()).upper()
    return normalized


def deduplicate_extractions(
    records: list[ExtractionRecord],
) -> list[ExtractionRecord]:
    """Deduplicate extractions BEFORE writing to Neo4j.

    Example: 'NVIDIA CORPORATION' and 'nvidia corp' both map to 'NVIDIA'.
    Uses the first seen canonical_name for each normalized group.

    Previously this deduplication ran post-write via resolve_extracted_node_duplicates,
    meaning duplicate nodes lived in the graph temporarily. This fix dedupes pre-write.
    """
    seen: dict[str, str] = {}  # normalized_name → canonical_name (first seen wins)
    deduped: list[ExtractionRecord] = []

    for record in records:
        norm = _normalize_name(record.canonical_name)
        if norm not in seen:
            # First occurrence of this normalized name — keep as-is
            seen[norm] = record.canonical_name
            deduped.append(record)
        else:
            # Already saw this normalized name — remap to canonical
            canonical = seen[norm]
            if record.canonical_name != canonical:
                # Create new record with updated canonical_name
                deduped.append(
                    ExtractionRecord(
                        entity_id=record.entity_id,
                        entity_type=record.entity_type,
                        canonical_name=canonical,  # use the first seen name
                        raw_text=record.raw_text,
                        confidence=record.confidence,
                        source_chunk_id=record.source_chunk_id,
                        metadata=record.metadata,
                    )
                )
            else:
                # Already using the canonical name
                deduped.append(record)

    return deduped


def write_document_graph(
    driver: Driver,
    settings: Settings,
    document: ParsedDocument,
    chunks: list[ChunkRecord],
    embedding_records: list[dict[str, Any]],
    extraction_records: list[ExtractionRecord],
    detected_metadata: DetectedMetadata | None = None,
    causal_links: list[dict[str, Any]] | None = None,
    event_causal_links: list[dict[str, Any]] | None = None,
    section_summary_records: list[dict[str, Any]] | None = None,
    company_metadata_embedding: list[float] | None = None,
    company_metadata_text: str | None = None,
) -> str:
    """Persist a parsed document, chunks, embeddings, and extracted entities.

    Uses idempotent MERGE plus UNWIND batching to keep write throughput stable.
    Optionally stores detected metadata from LLM classification.
    Pass section_summary_records from raptor.build_section_summaries() to
    persist RAPTOR-style summary + embedding on each Section node.
    """
    # Fix 6: Pre-write entity deduplication (deduplicate before any Neo4j writes)
    extraction_records = deduplicate_extractions(extraction_records)

    with driver.session(database=settings.neo4j_database) as session:
        session.execute_write(
            _upsert_document_tx,
            document,
            detected_metadata,
            company_metadata_embedding,
            company_metadata_text,
        )

        period_key = _build_quarter_period_key(
            document.metadata.get("period_start"),
            document.metadata.get("period_end"),
        )
        if period_key:
            session.execute_write(_upsert_quarter_tx, document, period_key)

        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _chunk_params(chunks),
            (
                "UNWIND $rows AS row "
                "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
                "SET c.doc_id = row.doc_id, "
                "    c.company_id = row.company_id, "
                "    c.text = row.text, "
                "    c.position = row.position, "
                "    c.token_count = row.token_count, "
                "    c.period_start = row.period_start, "
                "    c.period_end = row.period_end, "
                "    c.section_title = row.section_title, "
                "    c.section_id = row.section_id, "
                "    c.page_number = row.page_number, "
                "    c.chunk_type = row.chunk_type, "
                "    c.parent_chunk_id = row.parent_chunk_id "
                "WITH c, row "
                "MATCH (d:Document {doc_id: row.doc_id}) "
                "MERGE (c)-[:PART_OF]->(d)"
            ),
        )

        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            embedding_records,
            (
                "UNWIND $rows AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id}) "
                "SET c.embedding = row.embedding"
            ),
        )

        _create_chunk_sequence_edges(session, chunks)

        # Section nodes — (Document)-[:CONTAINS]->(Section)-[:HAS_CHUNK]->(Chunk)
        # Enables hierarchical graph traversal: Document → Section → Chunk
        # (Yang et al. [21], Gijre & Laddha [16])
        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _section_params(chunks),
            (
                "UNWIND $rows AS row "
                "MERGE (s:Section {section_id: row.section_id}) "
                "SET s.title = row.section_title, "
                "    s.doc_id = row.doc_id "
                "WITH s, row "
                "MATCH (d:Document {doc_id: row.doc_id}) "
                "MERGE (d)-[:CONTAINS]->(s) "
                "WITH s, row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id}) "
                "MERGE (s)-[:HAS_CHUNK]->(c)"
            ),
        )

        # RAPTOR section summaries (Sarthi et al. 2024)
        # Writes s.summary + s.summary_embedding onto the existing Section node.
        # Enables Stage-1 retrieval: vector search on section summaries → drill
        # into Chunk nodes only within the best-matching sections.
        if section_summary_records:
            _batched_unwind_write(
                session,
                settings.ingestion_write_batch_size,
                section_summary_records,
                (
                    "UNWIND $rows AS row "
                    "MATCH (s:Section {section_id: row.section_id}) "
                    "SET s.summary = row.summary, "
                    "    s.summary_embedding = row.summary_embedding"
                ),
            )

        # Parent-child edges: (parent prose chunk)-[:PARENT_OF]->(table chunk)
        # Allows traversal: table chunk → its prose context (Zhu et al. [24])
        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _parent_child_params(chunks),
            (
                "UNWIND $rows AS row "
                "MATCH (parent:Chunk {chunk_id: row.parent_chunk_id}) "
                "MATCH (child:Chunk {chunk_id: row.child_chunk_id}) "
                "MERGE (parent)-[:PARENT_OF]->(child)"
            ),
        )

        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _entity_extraction_params(extraction_records),
            (
                "UNWIND $rows AS row "
                "MERGE (e:Entity {entity_id: row.entity_id}) "
                "ON CREATE SET "
                "    e.entity_type = row.entity_type, "
                "    e.name = row.canonical_name, "
                "    e.raw_text = row.raw_text, "
                "    e.confidence = row.confidence "
                "ON MATCH SET "
                "    e.confidence = CASE WHEN row.confidence > e.confidence "
                "                       THEN row.confidence ELSE e.confidence END "
                "WITH e, row "
                "MATCH (c:Chunk {chunk_id: row.source_chunk_id}) "
                "MERGE (c)-[:HAS_ENTITY]->(e)"
            ),
        )

        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _event_extraction_params(extraction_records),
            (
                "UNWIND $rows AS row "
                "MERGE (e:Event {event_id: row.entity_id}) "
                "ON CREATE SET "
                "    e.event_type = row.entity_type, "
                "    e.name = row.canonical_name, "
                "    e.raw_text = row.raw_text, "
                "    e.confidence = row.confidence "
                "ON MATCH SET "
                "    e.confidence = CASE WHEN row.confidence > e.confidence "
                "                       THEN row.confidence ELSE e.confidence END "
                "WITH e, row "
                "MATCH (c:Chunk {chunk_id: row.source_chunk_id}) "
                "MERGE (c)-[:HAS_EVENT]->(e)"
            ),
        )

        _batched_unwind_write(
            session,
            settings.ingestion_write_batch_size,
            _metric_extraction_params(extraction_records),
            (
                "UNWIND $rows AS row "
                "MERGE (m:Metric {metric_id: row.entity_id}) "
                "ON CREATE SET "
                "    m.metric_type = row.entity_type, "
                "    m.name = row.canonical_name, "
                "    m.raw_text = row.raw_text, "
                "    m.confidence = row.confidence, "
                "    m.value = row.value, "
                "    m.unit = row.unit, "
                "    m.yoy_change = row.yoy_change "
                "ON MATCH SET "
                "    m.confidence = CASE WHEN row.confidence > m.confidence "
                "                       THEN row.confidence ELSE m.confidence END, "
                "    m.value = CASE WHEN row.value IS NOT NULL THEN row.value ELSE m.value END, "
                "    m.unit = CASE WHEN row.unit IS NOT NULL THEN row.unit ELSE m.unit END, "
                "    m.yoy_change = CASE WHEN row.yoy_change IS NOT NULL THEN row.yoy_change ELSE m.yoy_change END "
                "WITH m, row "
                "MATCH (c:Chunk {chunk_id: row.source_chunk_id}) "
                "MERGE (c)-[:HAS_METRIC]->(m)"
            ),
        )

        # Quarter→Metric edges: link each Metric to its reporting Quarter
        if period_key:
            chunk_period_map = _build_chunk_period_map(chunks, period_key)
            _batched_unwind_write(
                session,
                settings.ingestion_write_batch_size,
                _quarter_metric_link_params(extraction_records, chunk_period_map),
                (
                    "UNWIND $rows AS row "
                    "MATCH (m:Metric {metric_id: row.metric_id}) "
                    "MATCH (q:Quarter {period_key: row.period_key}) "
                    "MERGE (q)-[:HAS_METRIC]->(m)"
                ),
            )

        # Causal edges: (Event)-[:CAUSED]->(Metric)
        if causal_links:
            _batched_unwind_write(
                session,
                settings.ingestion_write_batch_size,
                causal_links,
                (
                    "UNWIND $rows AS row "
                    "MATCH (e:Event {event_id: row.event_id}) "
                    "MATCH (m:Metric {metric_id: row.metric_id}) "
                    "MERGE (e)-[r:CAUSED]->(m) "
                    "SET r.confidence = row.confidence"
                ),
            )

        # Causal chain edges: (Event)-[:LED_TO]->(Event)
        if event_causal_links:
            _batched_unwind_write(
                session,
                settings.ingestion_write_batch_size,
                event_causal_links,
                (
                    "UNWIND $rows AS row "
                    "MATCH (e1:Event {event_id: row.from_event_id}) "
                    "MATCH (e2:Event {event_id: row.to_event_id}) "
                    "MERGE (e1)-[r:LED_TO]->(e2) "
                    "SET r.confidence = row.confidence"
                ),
            )

    return document.company_id


def resolve_extracted_node_duplicates(driver: Driver, settings: Settings, doc_id: str) -> None:
    """Merge duplicate extracted nodes linked from chunks in the same document.

    Duplicates are grouped by normalized name plus type and merged by rewiring chunk
    relationships onto a single surviving node per group.
    """

    with driver.session(database=settings.neo4j_database) as session:
        for label, relationship in (
            ("Entity", "HAS_ENTITY"),
            ("Event", "HAS_EVENT"),
            ("Metric", "HAS_METRIC"),
        ):
            query = (
                f"MATCH (:Document {{doc_id: $doc_id}})<-[:PART_OF]-(:Chunk)-[:{relationship}]->(n:{label}) "
                "WITH toLower(trim(coalesce(n.name, ''))) AS normalized_name, "
                "     coalesce(n.entity_type, n.event_type, n.metric_type, '') AS normalized_type, "
                "     collect(DISTINCT n) AS nodes "
                "WHERE normalized_name <> '' AND size(nodes) > 1 "
                "CALL (nodes) { "
                "  WITH head(nodes) AS keep_node, tail(nodes) AS duplicate_nodes "
                "  UNWIND duplicate_nodes AS duplicate_node "
                f"  MATCH (chunk:Chunk)-[r:{relationship}]->(duplicate_node) "
                f"  MERGE (chunk)-[:{relationship}]->(keep_node) "
                "  DELETE r "
                "  DETACH DELETE duplicate_node "
                "  RETURN count(*) AS rewired_count "
                "} "
                "RETURN sum(rewired_count)"
            )
            session.run(
                Query(query),
                doc_id=doc_id,
            ).consume()


def build_similar_chunk_edges(driver: Driver, settings: Settings) -> None:
    """Create optional SIMILAR chunk edges using Neo4j vector index query."""

    query = (
        "MATCH (c:Chunk) "
        "WHERE c.embedding IS NOT NULL "
        "CALL (c) { "
        "  CALL db.index.vector.queryNodes('chunk_embedding_index', 6, c.embedding) "
        "  YIELD node AS nb, score "
        "  RETURN nb, score "
        "} "
        "WITH c, nb, score "
        "WHERE nb <> c AND score >= $threshold AND score < 1.0 "
        "MERGE (c)-[r:SIMILAR]-(nb) "
        "SET r.score = score"
    )

    with driver.session(database=settings.neo4j_database) as session:
        session.run(query, threshold=settings.similarity_threshold).consume()


def _upsert_document_tx(
    tx: Any,
    document: ParsedDocument,
    detected_metadata: DetectedMetadata | None = None,
    company_metadata_embedding: list[float] | None = None,
    company_metadata_text: str | None = None,
) -> None:
    company_name = document.company_id
    if detected_metadata and detected_metadata.company_name:
        company_name = detected_metadata.company_name

    tx.run(
        "MERGE (co:Company {company_id: $company_id}) "
        "SET co.name = CASE "
        "    WHEN $company_name IS NULL OR trim($company_name) = '' "
        "    THEN $company_id "
        "    ELSE $company_name "
        "END, "
        "co.metadata_text = CASE "
        "    WHEN $company_metadata_text IS NULL OR trim($company_metadata_text) = '' "
        "    THEN co.metadata_text "
        "    ELSE $company_metadata_text "
        "END, "
        "co.metadata_embedding = CASE "
        "    WHEN $company_metadata_embedding IS NULL "
        "    THEN co.metadata_embedding "
        "    ELSE $company_metadata_embedding "
        "END "
        "WITH co "
        "MERGE (d:Document {doc_id: $doc_id}) "
        "SET d.company_id = $company_id, "
        "    d.source_type = $source_type, "
        "    d.title = $title, "
        "    d.file_path = $file_path, "
        "    d.period_start = $period_start, "
        "    d.period_end = $period_end "
        "WITH co, d "
        "MERGE (co)-[:PUBLISHED]->(d)",
        doc_id=document.doc_id,
        company_id=document.company_id,
        company_name=company_name,
        company_metadata_embedding=company_metadata_embedding,
        company_metadata_text=company_metadata_text,
        source_type=document.source_type,
        title=document.title,
        file_path=document.file_path,
        period_start=document.metadata.get("period_start"),
        period_end=document.metadata.get("period_end"),
    ).consume()
    
    # Store detection metadata if provided
    if detected_metadata:
        tx.run(
            "MERGE (m:DocumentMetadata {doc_id: $doc_id}) "
            "SET m.company_id_detected = $company_id_detected, "
            "    m.company_name_detected = $company_name_detected, "
            "    m.document_type_detected = $document_type_detected, "
            "    m.confidence = $confidence, "
            "    m.detected_by = $detected_by "
            "WITH m "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (d)-[:HAS_METADATA]->(m)",
            doc_id=document.doc_id,
            company_id_detected=detected_metadata.company_id,
            company_name_detected=detected_metadata.company_name,
            document_type_detected=detected_metadata.document_type,
            confidence=detected_metadata.confidence,
            detected_by=detected_metadata.detected_by,
        ).consume()


def _create_chunk_sequence_edges(session: Any, chunks: list[ChunkRecord]) -> None:
    if not chunks:
        return

    sorted_chunks = sorted(chunks, key=lambda item: item.position)
    first_chunk = sorted_chunks[0]

    session.run(
        "MATCH (d:Document {doc_id: $doc_id}) "
        "MATCH (c:Chunk {chunk_id: $chunk_id}) "
        "MERGE (d)-[:FIRST_CHUNK]->(c)",
        doc_id=first_chunk.doc_id,
        chunk_id=first_chunk.chunk_id,
    ).consume()

    edges = [
        {
            "from_chunk_id": left.chunk_id,
            "to_chunk_id": right.chunk_id,
        }
        for left, right in zip(sorted_chunks, sorted_chunks[1:], strict=False)
    ]

    if not edges:
        return

    session.run(
        "UNWIND $rows AS row "
        "MATCH (from:Chunk {chunk_id: row.from_chunk_id}) "
        "MATCH (to:Chunk {chunk_id: row.to_chunk_id}) "
        "MERGE (from)-[:NEXT_CHUNK]->(to)",
        rows=edges,
    ).consume()


def _batched_unwind_write(session: Any, batch_size: int, rows: list[dict[str, Any]], query: str) -> None:
    if not rows:
        return

    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        session.run(query, rows=batch).consume()


def _chunk_params(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "company_id": chunk.company_id,
            "text": chunk.text,
            "position": chunk.position,
            "token_count": chunk.token_count,
            "period_start": chunk.metadata.get("period_start"),
            "period_end": chunk.metadata.get("period_end"),
            "section_title": chunk.section_title,
            "section_id": chunk.section_id,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "parent_chunk_id": chunk.parent_chunk_id,
        }
        for chunk in chunks
    ]


def _section_params(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    """One row per unique section_id — deduplication happens via MERGE in Cypher."""
    return [
        {
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
        if chunk.section_id
    ]


def _parent_child_params(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    """Rows for (prose)-[:PARENT_OF]->(table) edges."""
    return [
        {
            "parent_chunk_id": chunk.parent_chunk_id,
            "child_chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
        if chunk.parent_chunk_id
    ]


def _entity_extraction_params(records: list[ExtractionRecord]) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "canonical_name": record.canonical_name,
            "raw_text": record.raw_text,
            "confidence": record.confidence,
            "source_chunk_id": record.source_chunk_id,
        }
        for record in records
        if _classify_record_type(record.entity_type) == "entity"
    ]


def _event_extraction_params(records: list[ExtractionRecord]) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "canonical_name": record.canonical_name,
            "raw_text": record.raw_text,
            "confidence": record.confidence,
            "source_chunk_id": record.source_chunk_id,
        }
        for record in records
        if _classify_record_type(record.entity_type) == "event"
    ]


def _metric_extraction_params(records: list[ExtractionRecord]) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "canonical_name": record.canonical_name,
            "raw_text": record.raw_text,
            "confidence": record.confidence,
            "source_chunk_id": record.source_chunk_id,
            "value": record.metadata.get("value"),
            "unit": record.metadata.get("unit"),
            "yoy_change": record.metadata.get("yoy_change"),
        }
        for record in records
        if _classify_record_type(record.entity_type) == "metric"
    ]


def _classify_record_type(entity_type: str) -> str:
    normalized = entity_type.strip().lower()

    # Use substring matching so compound types like "Cash Flow" or
    # "FinancialMetric" are routed correctly instead of falling through to entity.
    metric_substrings = (
        "metric", "revenue", "eps", "guidance", "margin", "kpi",
        "income", "cash flow", "cashflow", "earnings", "growth",
        "profit", "loss", "expense", "cost", "ebitda", "operating",
        "financial result", "financialresult",
    )
    event_substrings = (
        "event", "regulatory", "productlaunch", "product launch",
        "earningscall", "earnings call", "announcement", "incident",
        "launch", "filing", "acquisition", "merger", "partnership",
        "financialevent", "financial event",
    )

    # Check events first — they are more specific (e.g. "earningscall" contains
    # "earnings" which is also a metric keyword; event must win).
    if any(kw in normalized for kw in event_substrings):
        return "event"
    if any(kw in normalized for kw in metric_substrings):
        return "metric"
    return "entity"


# ---------------------------------------------------------------------------
# Company identity helpers
# ---------------------------------------------------------------------------

_CORPORATE_SUFFIXES = (
    " corporation",
    " corp",
    " incorporated",
    " inc",
    " limited",
    " ltd",
    " llc",
    " plc",
    " holdings",
    " group",
    " co",
)


def _normalize_company_identity(name: str) -> str:
    """Return a lowercase, suffix-stripped version of a company name.

    Used to detect duplicate Company nodes caused by slightly different
    name representations (e.g. "NVIDIA" vs "NVIDIA CORPORATION").
    """
    stripped = name.strip().lower()
    if not stripped:
        return stripped
    for suffix in _CORPORATE_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].rstrip(" ,.")
    return stripped


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _build_quarter_period_key(period_start: str | None, period_end: str | None) -> str | None:
    """Build a stable period key for a Quarter node from document period bounds."""
    if period_start and period_end:
        return f"{period_start}/{period_end}"
    if period_start:
        return period_start
    if period_end:
        return period_end
    return None


def _upsert_quarter_tx(tx: Any, document: ParsedDocument, period_key: str) -> None:
    """Idempotently create a Quarter node and link Company + Document to it."""
    tx.run(
        "MERGE (q:Quarter {period_key: $period_key}) "
        "SET q.period_start = $period_start, "
        "    q.period_end = $period_end "
        "WITH q "
        "MATCH (co:Company {company_id: $company_id}) "
        "MERGE (co)-[:REPORTED_IN]->(q) "
        "WITH q "
        "MATCH (d:Document {doc_id: $doc_id}) "
        "MERGE (d)-[:BELONGS_TO]->(q)",
        period_key=period_key,
        period_start=document.metadata.get("period_start"),
        period_end=document.metadata.get("period_end"),
        company_id=document.company_id,
        doc_id=document.doc_id,
    ).consume()


def _build_chunk_period_map(chunks: list[ChunkRecord], period_key: str) -> dict[str, str]:
    """Map every chunk_id to the document period_key for Quarter->Metric linking."""
    return {chunk.chunk_id: period_key for chunk in chunks}


def _quarter_metric_link_params(
    records: list[ExtractionRecord],
    chunk_period_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Params for (Quarter)-[:HAS_METRIC]->(Metric) edge creation."""
    return [
        {
            "metric_id": record.entity_id,
            "period_key": chunk_period_map[record.source_chunk_id],
        }
        for record in records
        if _classify_record_type(record.entity_type) == "metric"
        and record.source_chunk_id in chunk_period_map
    ]
