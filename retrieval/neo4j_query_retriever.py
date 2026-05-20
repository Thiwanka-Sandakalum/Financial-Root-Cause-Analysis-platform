from __future__ import annotations

from typing import Iterable

from langsmith import traceable
from neo4j import Driver
from neo4j.exceptions import Neo4jError


_SEARCH_VECTOR_SUPPORTED: bool | None = None


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _retriever_doc(source_type: str, source_id: str, text: str, metadata: dict) -> dict:
    payload = dict(metadata)
    payload["source_type"] = source_type
    payload["source_id"] = source_id
    return {
        "page_content": text,
        "type": "Document",
        "metadata": payload,
    }


def _vector_search(
    driver: Driver,
    index_name: str,
    embedding: list[float],
    top_k: int,
) -> list[dict]:
    global _SEARCH_VECTOR_SUPPORTED

    search_query: str
    legacy_query: str
    if index_name == "chunk_embeddings":
        search_query = """
            SEARCH INDEX chunk_embeddings
            FOR (node:Chunk) ON (node.embedding)
            WHERE node.embedding ANN OF $embedding TOP $top_k
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
        """
        legacy_query = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
        """
    else:
        search_query = """
            SEARCH INDEX table_embeddings
            FOR (node:Table) ON (node.embedding)
            WHERE node.embedding ANN OF $embedding TOP $top_k
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
        """
        legacy_query = """
            CALL db.index.vector.queryNodes('table_embeddings', $top_k, $embedding)
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
        """

    # Current DB can emit deprecation notifications for queryNodes.
    # Silence only DEPRECATION notifications to keep logs readable.
    with driver.session(
        notifications_disabled_classifications=["DEPRECATION"]
    ) as session:
        if _SEARCH_VECTOR_SUPPORTED is not False:
            try:
                rows = session.run(
                    search_query,
                    top_k=top_k,
                    embedding=embedding,
                ).data()
                _SEARCH_VECTOR_SUPPORTED = True
                return rows
            except Neo4jError as exc:
                # Older servers may not support SEARCH syntax yet.
                if _SEARCH_VECTOR_SUPPORTED is True:
                    raise
                msg = str(exc)
                if "Invalid input 'SEARCH'" in msg or "SyntaxError" in msg:
                    _SEARCH_VECTOR_SUPPORTED = False
                else:
                    raise

        rows = session.run(
            legacy_query,
            top_k=top_k,
            embedding=embedding,
        ).data()
    return rows


@traceable(run_type="retriever", name="retrieve_chunk_documents")
def retrieve_chunk_documents(
    driver: Driver,
    embedding: list[float],
    top_k: int = 6,
    company_ticker: str | None = None,
    time_filter: str | None = None,
) -> list[dict]:
    rows = _vector_search(driver, "chunk_embeddings", embedding, top_k)
    docs: list[dict] = []

    for row in rows:
        chunk = row["node"]
        score = float(row["score"])
        chunk_id = chunk.get("id")

        with driver.session() as session:
            detail = session.run(
                """
                MATCH (ch:Chunk {id: $chunk_id})
                OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(ch)
                OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
                OPTIONAL MATCH (c:Company)-[:FILED]->(d)
                WITH ch, s, d, c
                WHERE $ticker IS NULL OR c.ticker = $ticker
                RETURN
                    ch.id AS chunk_id,
                    ch.text AS text,
                    ch.page AS page,
                    ch.section_type AS section_type,
                    s.id AS section_id,
                    s.title AS section_title,
                    d.id AS doc_id,
                    d.type AS doc_type,
                    d.period AS period,
                    c.ticker AS ticker,
                    c.name AS company
                LIMIT 1
                """,
                chunk_id=chunk_id,
                ticker=_normalize_text(company_ticker) or None,
            ).single()

        if not detail:
            continue

        page_content = detail["text"] or ""
        docs.append(
            _retriever_doc(
                "chunk",
                detail["chunk_id"],
                page_content,
                {
                    "score": score,
                    "page": detail["page"],
                    "section_type": detail["section_type"],
                    "section_id": detail["section_id"],
                    "section_title": detail["section_title"],
                    "doc_id": detail["doc_id"],
                    "doc_type": detail["doc_type"],
                    "period": detail["period"],
                    "ticker": detail["ticker"],
                    "company": detail["company"],
                    "time_filter": time_filter,
                },
            )
        )

    return docs


@traceable(run_type="retriever", name="retrieve_table_documents")
def retrieve_table_documents(
    driver: Driver,
    embedding: list[float],
    top_k: int = 4,
    company_ticker: str | None = None,
    time_filter: str | None = None,
) -> list[dict]:
    rows = _vector_search(driver, "table_embeddings", embedding, top_k)
    docs: list[dict] = []

    for row in rows:
        table = row["node"]
        score = float(row["score"])
        table_id = table.get("id")

        with driver.session() as session:
            detail = session.run(
                """
                MATCH (t:Table {id: $table_id})
                OPTIONAL MATCH (s:Section)-[:HAS_TABLE]->(t)
                OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
                OPTIONAL MATCH (c:Company)-[:FILED]->(d)
                WITH t, s, d, c
                WHERE $ticker IS NULL OR c.ticker = $ticker
                RETURN
                    t.id AS table_id,
                    t.markdown AS markdown,
                    t.sequence AS sequence,
                    s.id AS section_id,
                    s.title AS section_title,
                    s.section_type AS section_type,
                    d.id AS doc_id,
                    d.type AS doc_type,
                    d.period AS period,
                    c.ticker AS ticker,
                    c.name AS company
                LIMIT 1
                """,
                table_id=table_id,
                ticker=_normalize_text(company_ticker) or None,
            ).single()

        if not detail:
            continue

        docs.append(
            _retriever_doc(
                "table",
                detail["table_id"],
                detail["markdown"] or "",
                {
                    "score": score,
                    "sequence": detail["sequence"],
                    "section_type": detail["section_type"],
                    "section_id": detail["section_id"],
                    "section_title": detail["section_title"],
                    "doc_id": detail["doc_id"],
                    "doc_type": detail["doc_type"],
                    "period": detail["period"],
                    "ticker": detail["ticker"],
                    "company": detail["company"],
                    "time_filter": time_filter,
                },
            )
        )

    return docs


def expand_graph_context(
    driver: Driver,
    source_documents: Iterable[dict],
    hop_depth: int = 2,
    use_causal_edges: bool = True,
) -> list[dict]:
    paths: list[dict] = []

    with driver.session() as session:
        for doc in source_documents:
            meta = doc.get("metadata", {})
            source_type = meta.get("source_type")
            source_id = meta.get("source_id")

            if source_type == "chunk":
                record = session.run(
                    """
                    MATCH (ch:Chunk {id: $source_id})
                    OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(ch)
                    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
                    OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(ch)
                    OPTIONAL MATCH (ch)-[:NEXT_CHUNK]->(next:Chunk)
                    OPTIONAL MATCH (ch)-[:MENTIONS]->(entity)
                    OPTIONAL MATCH (entity)-[rel:CAUSED|IMPACTED|DEPENDS_ON|REPORTED_BY|HAS_EXECUTIVE]->(other)
                    RETURN
                        ch.id AS chunk_id,
                        s.id AS section_id,
                        s.title AS section_title,
                        d.id AS doc_id,
                        d.type AS doc_type,
                        d.period AS period,
                        collect(DISTINCT prev.id) AS prev_chunks,
                        collect(DISTINCT next.id) AS next_chunks,
                        collect(DISTINCT entity.name) AS entities,
                        collect(DISTINCT CASE WHEN $use_causal THEN type(rel) + ':' + coalesce(other.name, other.id, '') ELSE NULL END) AS causal_links
                    LIMIT 1
                    """,
                    source_id=source_id,
                    use_causal=use_causal_edges,
                ).single()
            else:
                record = session.run(
                    """
                    MATCH (t:Table {id: $source_id})
                    OPTIONAL MATCH (s:Section)-[:HAS_TABLE]->(t)
                    OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
                    RETURN
                        t.id AS table_id,
                        s.id AS section_id,
                        s.title AS section_title,
                        d.id AS doc_id,
                        d.type AS doc_type,
                        d.period AS period
                    LIMIT 1
                    """,
                    source_id=source_id,
                ).single()

            if record:
                payload = dict(record)
                payload["hop_depth"] = hop_depth
                paths.append(payload)

    return paths
