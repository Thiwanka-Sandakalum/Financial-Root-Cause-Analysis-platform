"""Graph expansion for multi-hop context retrieval from Neo4j."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document
from neo4j import Driver

from src.config import Settings


def expand_chunk_graph(
    driver: Driver,
    chunk_ids: list[str],
    settings: Settings,
    intent_type: str | None = None,
) -> list[Document]:
    """Expand chunks by traversing graph edges for contextual enrichment.
    
    Expansion strategy:
    1. For each seed chunk, get NEXT_CHUNK neighbors (±1 hop for document flow)
    2. Traverse HAS_ENTITY/HAS_EVENT/HAS_METRIC to linked domain nodes
    3. Include SIMILAR chunks (score > threshold) for semantic context
    
    Args:
        driver: Neo4j driver instance
        chunk_ids: List of seed chunk IDs to expand from
        settings: Application settings with expansion thresholds
        
    Returns:
        List of Document objects representing expanded context.
        For root-cause intent, causal chain evidence is appended.
    """
    if not chunk_ids:
        return []

    # Query for multi-hop expansion with deduplication
    cypher = _build_expansion_cypher(settings)
    
    similarity_threshold = settings.query_similarity_threshold
    results: list[Document] = []
    try:
        with driver.session(database=settings.neo4j_database) as session:
            records = session.run(
                cast(Any, cypher),
                seed_chunk_ids=chunk_ids,
                similarity_threshold=similarity_threshold,
            ).fetch(1000)
            
            for record in records:
                doc = Document(
                    page_content=record["text"] or "",
                    metadata={
                        "chunk_id": record["chunk_id"],
                        "doc_id": record["doc_id"],
                        "company_id": record["company_id"],
                        "position": record["position"],
                        "source_type": record["source_type"],
                        "is_seed": record["is_seed"],
                        "expansion_reason": record["expansion_reason"],
                        "combined_score": record["combined_score"],
                    },
                )
                results.append(doc)
    except Exception as exc:
        raise RuntimeError(f"Graph expansion failed: {exc}") from exc

    if intent_type == "root_cause_analysis":
        causal_docs = causal_chain_expansion(driver, chunk_ids, settings)
        merged: dict[str, Document] = {
            str(doc.metadata.get("chunk_id")): doc
            for doc in results
            if doc.metadata.get("chunk_id")
        }
        for doc in causal_docs:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = doc
                continue
            if float(doc.metadata.get("combined_score", 0.0)) > float(
                existing.metadata.get("combined_score", 0.0)
            ):
                merged[chunk_id] = doc
        return list(merged.values())

    return results


def causal_chain_expansion(
    driver: Driver,
    seed_chunk_ids: list[str],
    settings: Settings,
) -> list[Document]:
    """Expand through Event->Metric and Event->Event causal edges."""
    if not seed_chunk_ids:
        return []

    docs: list[Document] = []
    try:
        with driver.session(database=settings.neo4j_database) as session:
            caused_rows = session.run(
                (
                    "MATCH (seed:Chunk) "
                    "WHERE seed.chunk_id IN $seed_ids "
                    "MATCH (seed)-[:HAS_EVENT]->(e:Event)-[:CAUSED]->(m:Metric) "
                    "MATCH (target:Chunk)-[:HAS_METRIC]->(m) "
                    "RETURN DISTINCT "
                    "  target.chunk_id AS chunk_id, "
                    "  target.doc_id AS doc_id, "
                    "  target.company_id AS company_id, "
                    "  target.text AS text, "
                    "  target.position AS position, "
                    "  target.chunk_type AS chunk_type, "
                    "  target.section_title AS section_title, "
                    "  target.page_number AS page_number, "
                    "  e.description AS causal_event, "
                    "  m.name AS affected_metric "
                    "LIMIT 8"
                ),
                seed_ids=seed_chunk_ids,
            ).data()

            for row in caused_rows:
                docs.append(
                    Document(
                        page_content=row.get("text") or "",
                        metadata={
                            "chunk_id": row.get("chunk_id"),
                            "doc_id": row.get("doc_id"),
                            "company_id": row.get("company_id"),
                            "position": row.get("position"),
                            "chunk_type": row.get("chunk_type") or "prose",
                            "section_title": row.get("section_title"),
                            "page_number": row.get("page_number"),
                            "causal_event": row.get("causal_event"),
                            "affected_metric": row.get("affected_metric"),
                            "combined_score": 0.85,
                            "expansion_reason": "causal_caused",
                        },
                    )
                )

            led_rows = session.run(
                (
                    "MATCH (seed:Chunk) "
                    "WHERE seed.chunk_id IN $seed_ids "
                    "MATCH (seed)-[:HAS_EVENT]->(e1:Event)-[:LED_TO]->(e2:Event) "
                    "MATCH (target:Chunk)-[:HAS_EVENT]->(e2) "
                    "RETURN DISTINCT "
                    "  target.chunk_id AS chunk_id, "
                    "  target.doc_id AS doc_id, "
                    "  target.company_id AS company_id, "
                    "  target.text AS text, "
                    "  target.position AS position, "
                    "  target.chunk_type AS chunk_type, "
                    "  target.section_title AS section_title, "
                    "  target.page_number AS page_number, "
                    "  e1.description AS upstream_event, "
                    "  e2.description AS downstream_event "
                    "LIMIT 6"
                ),
                seed_ids=seed_chunk_ids,
            ).data()

            for row in led_rows:
                docs.append(
                    Document(
                        page_content=row.get("text") or "",
                        metadata={
                            "chunk_id": row.get("chunk_id"),
                            "doc_id": row.get("doc_id"),
                            "company_id": row.get("company_id"),
                            "position": row.get("position"),
                            "chunk_type": row.get("chunk_type") or "prose",
                            "section_title": row.get("section_title"),
                            "page_number": row.get("page_number"),
                            "upstream_event": row.get("upstream_event"),
                            "downstream_event": row.get("downstream_event"),
                            "combined_score": 0.80,
                            "expansion_reason": "causal_led_to",
                        },
                    )
                )
    except Exception as exc:
        raise RuntimeError(f"Causal graph expansion failed: {exc}") from exc

    # Keep the best scoring instance per chunk.
    by_chunk_id: dict[str, Document] = {}
    for doc in docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if not chunk_id:
            continue
        existing = by_chunk_id.get(chunk_id)
        if existing is None or float(doc.metadata.get("combined_score", 0.0)) > float(
            existing.metadata.get("combined_score", 0.0)
        ):
            by_chunk_id[chunk_id] = doc
    return list(by_chunk_id.values())


def _build_expansion_cypher(settings: Settings) -> str:
    """Build Cypher query for graph expansion with deduplication and scoring.

    Returns chunks in this order:
    1. Seed chunks (original retrieved set)
    2. Adjacent chunks via NEXT_CHUNK (up to query_graph_expansion_hops hops)
    3. Chunks connected through shared entities/events/metrics
    4. Semantically similar chunks via SIMILAR (score > threshold)

    All deduplicated by chunk_id.

    Notes:
        ``hops`` is interpolated via f-string because Neo4j does not support
        parameterized relationship hop counts (``*1..$hops`` is not valid Cypher).
        ``similarity_threshold`` is passed as a named parameter ``$similarity_threshold``
        to keep it out of the query text and allow proper query-plan caching.
    """
    hops = settings.query_graph_expansion_hops

    cypher = f"""
        CALL () {{
            // Seed chunks
            MATCH (seed:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
            RETURN seed AS chunk, 1.0 AS score, "seed" AS reason

            UNION ALL

            // NEXT_CHUNK: forward context (up to {hops} hops, score decays with distance)
            MATCH (seed:Chunk)-[r:NEXT_CHUNK*1..{hops}]->(adjacent:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
            RETURN adjacent AS chunk, (0.9 - (size(r) - 1) * 0.1) AS score, "next_chunk_forward" AS reason

            UNION ALL

            // NEXT_CHUNK: backward context (up to {hops} hops, score decays with distance)
            MATCH (seed:Chunk)<-[r:NEXT_CHUNK*1..{hops}]-(adjacent:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
            RETURN adjacent AS chunk, (0.85 - (size(r) - 1) * 0.05) AS score, "next_chunk_backward" AS reason

            UNION ALL

            // Shared entity context: seed -> entity <- related chunk
            MATCH (seed:Chunk)-[:HAS_ENTITY]->(:Entity)<-[:HAS_ENTITY]-(related:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
                AND related.chunk_id <> seed.chunk_id
            RETURN related AS chunk, 0.8 AS score, "shared_entity" AS reason

            UNION ALL

            // Shared event context: seed -> event <- related chunk
            MATCH (seed:Chunk)-[:HAS_EVENT]->(:Event)<-[:HAS_EVENT]-(related:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
                AND related.chunk_id <> seed.chunk_id
            RETURN related AS chunk, 0.78 AS score, "shared_event" AS reason

            UNION ALL

            // Shared metric context: seed -> metric <- related chunk
            MATCH (seed:Chunk)-[:HAS_METRIC]->(:Metric)<-[:HAS_METRIC]-(related:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
                AND related.chunk_id <> seed.chunk_id
            RETURN related AS chunk, 0.76 AS score, "shared_metric" AS reason

            UNION ALL

            // SIMILAR: semantic neighbors
            MATCH (seed:Chunk)-[sim:SIMILAR]-(similar:Chunk)
            WHERE seed.chunk_id IN $seed_chunk_ids
                AND sim.score > $similarity_threshold
            RETURN similar AS chunk, sim.score AS score, "similar_chunk" AS reason
        }}

        WITH
            chunk.chunk_id AS chunk_id,
            chunk.doc_id AS doc_id,
            chunk.company_id AS company_id,
            chunk.text AS text,
            chunk.position AS position,
            MAX(score) AS combined_score,
            HEAD(COLLECT(DISTINCT reason)) AS expansion_reason,
            CASE WHEN chunk.chunk_id IN $seed_chunk_ids THEN true ELSE false END AS is_seed
        MATCH (d:Document {{doc_id: doc_id}})
        RETURN
            chunk_id,
            doc_id,
            company_id,
            text,
            position,
            d.source_type AS source_type,
            is_seed,
            expansion_reason,
            combined_score
        ORDER BY is_seed DESC, combined_score DESC
        """
    
    return cypher
