import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import Driver

from ingestion.pipeline.chunker import chunk_document
from ingestion.pipeline.entity_extractor import extract_entities_and_relations
from ingestion.pipeline.parser import parse_financial_pdf, parse_text_file

# Sections worth running entity extraction on (limits Gemini API cost)
_KEY_SECTIONS = {"MD&A", "RISK", "EARNINGS"}
_MAX_ENTITY_CHUNKS = 20
_ALLOWED_ENTITY_LABELS = {
    "Company",
    "Executive",
    "Product",
    "FinancialMetric",
    "RiskFactor",
    "MacroEvent",
    "FinancialEvent",
}
_ALLOWED_REL_TYPES = {
    "CAUSED",
    "MENTIONS",
    "COMPETES_WITH",
    "DEPENDS_ON",
    "IMPACTED",
    "REPORTED_BY",
    "HAS_EXECUTIVE",
}

T = TypeVar("T")


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
                f"  [graph_writer] Retry {attempt}/{max_attempts - 1} for {context}: {exc}"
            )
            time.sleep(delay)
    assert last_exc is not None
    raise RuntimeError(f"Failed {context} after {max_attempts} attempts: {last_exc}")


def _normalize_name(name: str) -> str:
    return name.strip().title()


def _write_document_company(
    tx, company_ticker, company_name, doc_id, doc_type, fiscal_period, filename
):
    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        ON CREATE SET c.name = $name
        ON MATCH SET c.name = coalesce(c.name, $name)
        MERGE (d:Document {id: $doc_id})
        SET d.type = $doc_type,
            d.period = $period,
            d.filename = $filename
        MERGE (c)-[:FILED]->(d)
        """,
        ticker=company_ticker,
        name=company_name,
        doc_id=doc_id,
        doc_type=doc_type,
        period=fiscal_period,
        filename=filename,
    )


def _write_sections(tx, doc_id, section_rows):
    if not section_rows:
        return
    tx.run(
        """
        MATCH (d:Document {id: $doc_id})
        UNWIND $rows AS row
        MERGE (s:Section {id: row.id})
        SET s.title = row.title,
            s.section_type = row.section_type,
            s.page_start = row.page_start,
            s.page_end = row.page_end
        MERGE (d)-[:CONTAINS]->(s)
        """,
        doc_id=doc_id,
        rows=section_rows,
    )


def _write_chunks(tx, chunk_rows):
    if not chunk_rows:
        return
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Section {id: row.section_id})
        MERGE (ch:Chunk {id: row.id})
        SET ch.text = row.text,
            ch.section_type = row.section_type,
            ch.page = row.page,
            ch.sequence = row.sequence,
            ch.embedding = row.embedding
        MERGE (s)-[rel:HAS_CHUNK]->(ch)
        SET rel.sequence = row.sequence
        """,
        rows=chunk_rows,
    )


def _write_tables(tx, table_rows):
    if not table_rows:
        return
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Section {id: row.section_id})
        MERGE (t:Table {id: row.id})
        SET t.markdown = row.markdown,
            t.sequence = row.sequence,
            t.embedding = row.embedding
        MERGE (s)-[rel:HAS_TABLE]->(t)
        SET rel.sequence = row.sequence
        """,
        rows=table_rows,
    )


def _write_next_chunk_links(tx, doc_id):
    tx.run(
        """
        MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(:Section)-[:HAS_CHUNK]->(ch:Chunk)
        WITH ch ORDER BY ch.sequence
        WITH collect(ch) AS chunks
        UNWIND range(0, size(chunks) - 2) AS i
        WITH chunks[i] AS curr, chunks[i+1] AS nxt
        MERGE (curr)-[:NEXT_CHUNK]->(nxt)
        """,
        doc_id=doc_id,
    )


def _write_mentions(tx, entity_label: str, mention_rows):
    if not mention_rows:
        return
    tx.run(
        f"""
        UNWIND $rows AS row
        MERGE (e:{entity_label} {{name: row.name}})
        WITH e, row
        MATCH (ch:Chunk {{id: row.chunk_id}})
        MERGE (ch)-[:MENTIONS]->(e)
        """,
        rows=mention_rows,
    )


def _write_relationships(tx, rel_type: str, rel_rows):
    if not rel_rows:
        return
    tx.run(
        f"""
        UNWIND $rows AS row
        MATCH (src {{name: row.source}})
        MATCH (tgt {{name: row.target}})
        MERGE (src)-[:{rel_type}]->(tgt)
        """,
        rows=rel_rows,
    )


def ingest_document(
    file_path: str,
    company_ticker: str,
    company_name: str,
    doc_type: str,
    fiscal_period: str,
    driver: Driver,
    embedder: GoogleGenerativeAIEmbeddings,
    llm: ChatGoogleGenerativeAI,
    doc_id: Optional[str] = None,
) -> str:
    """
    Full ingestion pipeline for a single document (PDF or .txt).

    Steps:
    1. Parse into structure-aware sections (pdfplumber / plain text)
    2. Chunk sections with 512-char sentence windows
    3. Embed all chunks via text-embedding-004 (batch call)
    4. Extract entities + relations from key sections via Gemini
    5. Write everything to Neo4j

    Returns the document UUID used as the graph node id.
    """
    doc_id = doc_id or str(uuid.uuid4())
    path = Path(file_path)

    print(f"\n[graph_writer] Ingesting '{path.name}' as {doc_type} {fiscal_period} ...")

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    if path.suffix.lower() == ".pdf":
        sections = parse_financial_pdf(file_path, llm)
    else:
        sections = parse_text_file(file_path)

    print(f"  Parsed {len(sections)} section(s): {[s.section_type for s in sections]}")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    all_chunks = chunk_document(sections, doc_id)
    print(f"  Produced {len(all_chunks)} chunk(s)")

    if not all_chunks:
        print("  [graph_writer] No chunks produced — skipping document.")
        return doc_id

    # ── Step 3: Embed (single batch call — chunks + tables together) ─────────
    # Build table records: (section_id, seq, raw_markdown, context_text)
    # Context prefix = "[SECTION_TYPE] Title" so table embeddings carry their
    # heading even when retrieved in isolation (Zhu et al. 2021 alignment).
    table_records = []
    for idx, section in enumerate(sections):
        section_id = f"{doc_id}_sec_{idx}"
        for tbl_idx, table_md in enumerate(section.tables):
            context_text = f"[{section.section_type}] {section.title}\n\n{table_md}"
            table_records.append((section_id, tbl_idx, table_md, context_text))

    chunk_texts = [c.text for c in all_chunks]
    table_texts = [r[3] for r in table_records]
    all_texts = chunk_texts + table_texts

    print(f"  Embedding {len(chunk_texts)} chunk(s) + {len(table_texts)} table(s) ...")
    all_embeddings = _retry_call(
        lambda: embedder.embed_documents(all_texts),
        context=f"embedding {len(all_texts)} text blocks",
    )
    if len(all_embeddings) != len(all_texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(all_texts)}, got {len(all_embeddings)}"
        )
    if not all_embeddings:
        raise RuntimeError("Embedder returned zero vectors")

    embedding_dims = {len(vec) for vec in all_embeddings}
    if len(embedding_dims) != 1:
        raise RuntimeError(
            f"Inconsistent embedding dimensions: {sorted(embedding_dims)}"
        )

    chunk_embeddings = all_embeddings[: len(chunk_texts)]
    table_embeddings = all_embeddings[len(chunk_texts) :]
    print(f"  Embedding done (dim={len(all_embeddings[0])})")

    # ── Step 4: Entity extraction (key sections only) ─────────────────────────
    key_chunks = [c for c in all_chunks if c.section_type in _KEY_SECTIONS]
    key_chunks = key_chunks[:_MAX_ENTITY_CHUNKS]
    print(f"  Extracting entities from {len(key_chunks)} key chunk(s) ...")
    extraction_results = extract_entities_and_relations(key_chunks, llm)

    # ── Step 5: Write to Neo4j ────────────────────────────────────────────────
    print("  Writing to Neo4j ...")

    section_rows = [
        {
            "id": f"{doc_id}_sec_{idx}",
            "title": section.title,
            "section_type": section.section_type,
            "page_start": section.page_start,
            "page_end": section.page_end,
        }
        for idx, section in enumerate(sections)
    ]

    chunk_rows = [
        {
            "id": chunk.id,
            "section_id": chunk.section_id,
            "text": chunk.text,
            "section_type": chunk.section_type,
            "page": chunk.page,
            "sequence": chunk.sequence,
            "embedding": embedding,
        }
        for chunk, embedding in zip(all_chunks, chunk_embeddings)
    ]

    table_rows = [
        {
            "id": f"{section_id}_tbl_{tbl_seq}",
            "section_id": section_id,
            "markdown": table_md,
            "sequence": tbl_seq,
            "embedding": table_emb,
        }
        for (section_id, tbl_seq, table_md, _), table_emb in zip(
            table_records, table_embeddings
        )
    ]

    mentions_by_type = defaultdict(list)
    rels_by_type = defaultdict(list)

    for chunk, extraction in zip(key_chunks, extraction_results):
        for entity in extraction.entities:
            if entity.type not in _ALLOWED_ENTITY_LABELS:
                print(f"  [graph_writer] Skipped invalid entity label '{entity.type}'")
                continue
            normalized_name = _normalize_name(entity.name)
            if not normalized_name:
                continue
            mentions_by_type[entity.type].append(
                {"name": normalized_name, "chunk_id": chunk.id}
            )

        for rel in extraction.relations:
            if rel.relationship not in _ALLOWED_REL_TYPES:
                print(
                    f"  [graph_writer] Skipped invalid relationship type '{rel.relationship}'"
                )
                continue
            source = _normalize_name(rel.source)
            target = _normalize_name(rel.target)
            if not source or not target:
                continue
            rels_by_type[rel.relationship].append({"source": source, "target": target})

    with driver.session() as session:
        session.execute_write(
            _write_document_company,
            company_ticker,
            company_name,
            doc_id,
            doc_type,
            fiscal_period,
            path.name,
        )
        session.execute_write(_write_sections, doc_id, section_rows)
        session.execute_write(_write_chunks, chunk_rows)
        session.execute_write(_write_tables, table_rows)
        session.execute_write(_write_next_chunk_links, doc_id)

        for entity_label, rows in mentions_by_type.items():
            session.execute_write(_write_mentions, entity_label, rows)

        for rel_type, rows in rels_by_type.items():
            session.execute_write(_write_relationships, rel_type, rows)

    entity_count = sum(len(r.entities) for r in extraction_results)
    print(
        f"  [graph_writer] Done — {len(all_chunks)} chunks, "
        f"{entity_count} entities written. doc_id={doc_id}"
    )
    return doc_id
