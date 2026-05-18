import json
import logging

from neo4j import Driver

from src.config import Settings

logger = logging.getLogger(__name__)

_CHUNK_VECTOR_INDEX_NAME = "chunk_embedding_index"
_CHUNK_VECTOR_REQUIRED_PROPERTIES = {"embedding", "company_id", "period_start", "period_end"}
_COMPANY_VECTOR_INDEX_NAME = "company_metadata_embedding_index"


def _run_write(driver: Driver, database: str, query: str) -> None:
    with driver.session(database=database) as session:
        session.run(query).consume()


def _chunk_vector_index_query(settings: Settings) -> str:
    return (
        f"CREATE VECTOR INDEX {_CHUNK_VECTOR_INDEX_NAME} IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "WITH [c.company_id, c.period_start, c.period_end] "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        + str(settings.embedding_dimensions)
        + ", `vector.similarity_function`: 'cosine'}}"
    )


def _ensure_section_summary_vector_index(driver: Driver, settings: Settings) -> None:
    """Create the vector index on Section.summary_embedding (RAPTOR Stage-1 search)."""
    query = (
        "CREATE VECTOR INDEX section_summary_embedding_index IF NOT EXISTS "
        "FOR (s:Section) ON (s.summary_embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        + str(settings.embedding_dimensions)
        + ", `vector.similarity_function`: 'cosine'}}"
    )
    with driver.session(database=settings.neo4j_database) as session:
        session.run(query).consume()


def _ensure_company_metadata_vector_index(driver: Driver, settings: Settings) -> None:
    """Create vector index on Company.metadata_embedding for canonical matching."""
    query = (
        f"CREATE VECTOR INDEX {_COMPANY_VECTOR_INDEX_NAME} IF NOT EXISTS "
        "FOR (co:Company) ON (co.metadata_embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        + str(settings.embedding_dimensions)
        + ", `vector.similarity_function`: 'cosine'}}"
    )
    with driver.session(database=settings.neo4j_database) as session:
        session.run(query).consume()


def _ensure_chunk_vector_index(driver: Driver, settings: Settings) -> None:
    with driver.session(database=settings.neo4j_database) as session:
        row = session.run(
            "SHOW VECTOR INDEXES YIELD name, properties "
            "WHERE name = $name "
            "RETURN properties",
            name=_CHUNK_VECTOR_INDEX_NAME,
        ).single()

        if row is None:
            session.run(_chunk_vector_index_query(settings)).consume()
            return

        properties = set(row.get("properties") or [])
        if _CHUNK_VECTOR_REQUIRED_PROPERTIES.issubset(properties):
            return

        logger.info(
            "Recreating %s to enable filterable vector properties",
            _CHUNK_VECTOR_INDEX_NAME,
        )
        session.run(f"DROP INDEX {_CHUNK_VECTOR_INDEX_NAME} IF EXISTS").consume()
        session.run(_chunk_vector_index_query(settings)).consume()


def initialize_neo4j_schema(driver: Driver, settings: Settings) -> None:
    constraints = [
        "CREATE CONSTRAINT company_company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.company_id IS UNIQUE",
        "CREATE CONSTRAINT document_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        # Uniqueness constraints on extracted nodes prevent duplicate creation
        # when the same entity appears across multiple chunks or re-ingestion runs.
        # (IS NOT NULL existence constraints require Enterprise Edition; uniqueness
        # here combined with the application-level non-empty check in _normalize_items
        # gives equivalent protection on Community Edition.)
        "CREATE CONSTRAINT entity_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE CONSTRAINT event_event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
        "CREATE CONSTRAINT metric_metric_id IF NOT EXISTS FOR (m:Metric) REQUIRE m.metric_id IS UNIQUE",
        "CREATE CONSTRAINT quarter_period_key IF NOT EXISTS FOR (q:Quarter) REQUIRE q.period_key IS UNIQUE",
        # Section nodes — one node per (doc, section_title) pair (P2: explicit Section node)
        "CREATE CONSTRAINT section_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
    ]

    indexes = [
        (
            "chunk_fulltext_index",
            "CREATE FULLTEXT INDEX chunk_fulltext_index IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text] "
            "OPTIONS {indexConfig: {`fulltext.eventually_consistent`: true}}",
        ),
        (
            "entity_fulltext_index",
            "CREATE FULLTEXT INDEX entity_fulltext_index IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.raw_text] "
            "OPTIONS {indexConfig: {`fulltext.eventually_consistent`: true}}",
        ),
        (
            "event_fulltext_index",
            "CREATE FULLTEXT INDEX event_fulltext_index IF NOT EXISTS "
            "FOR (e:Event) ON EACH [e.name, e.raw_text] "
            "OPTIONS {indexConfig: {`fulltext.eventually_consistent`: true}}",
        ),
        (
            "metric_fulltext_index",
            "CREATE FULLTEXT INDEX metric_fulltext_index IF NOT EXISTS "
            "FOR (m:Metric) ON EACH [m.name, m.raw_text] "
            "OPTIONS {indexConfig: {`fulltext.eventually_consistent`: true}}",
        ),
        (
            "document_company_index",
            "CREATE INDEX document_company_index IF NOT EXISTS FOR (d:Document) ON (d.company_id)",
        ),
        (
            "document_period_start_index",
            "CREATE INDEX document_period_start_index IF NOT EXISTS FOR (d:Document) ON (d.period_start)",
        ),
        (
            "document_period_end_index",
            "CREATE INDEX document_period_end_index IF NOT EXISTS FOR (d:Document) ON (d.period_end)",
        ),
        (
            "quarter_period_start_index",
            "CREATE INDEX quarter_period_start_index IF NOT EXISTS FOR (q:Quarter) ON (q.period_start)",
        ),
        (
            "quarter_period_end_index",
            "CREATE INDEX quarter_period_end_index IF NOT EXISTS FOR (q:Quarter) ON (q.period_end)",
        ),
        (
            "section_doc_index",
            "CREATE INDEX section_doc_index IF NOT EXISTS FOR (s:Section) ON (s.doc_id)",
        ),
        (
            "chunk_type_index",
            "CREATE INDEX chunk_type_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
        ),
        (
            "chunk_section_id_index",
            "CREATE INDEX chunk_section_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.section_id)",
        ),
    ]

    for query in constraints:
        _run_write(driver, settings.neo4j_database, query)

    _ensure_chunk_vector_index(driver, settings)
    _ensure_section_summary_vector_index(driver, settings)
    _ensure_company_metadata_vector_index(driver, settings)

    for _, query in indexes:
        _run_write(driver, settings.neo4j_database, query)

    logger.info("Neo4j schema initialization finished")


def main() -> None:
    import sys

    from src.config import get_settings
    from src.db.neo4j_client import Neo4jClient

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    client = Neo4jClient(settings)

    try:
        client.verify_connectivity()
        initialize_neo4j_schema(client.driver, settings)
        status = verify_neo4j_schema(client.driver, settings)
        print(json.dumps(status, indent=2))
        if not all(status.values()):
            sys.exit(1)
    finally:
        client.close()


def verify_neo4j_schema(driver: Driver, settings: Settings) -> dict[str, bool]:
    required_constraints = {
        ("Company", "company_id"),
        ("Document", "doc_id"),
        ("Chunk", "chunk_id"),
        ("Entity", "entity_id"),
        ("Event", "event_id"),
        ("Metric", "metric_id"),
        ("Quarter", "period_key"),
        # Section node added for RAPTOR Stage-1 (P2)
        ("Section", "section_id"),
    }

    with driver.session(database=settings.neo4j_database) as session:
        constraint_rows = list(
            session.run(
                "SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties "
                "RETURN type, labelsOrTypes, properties"
            )
        )
        index_rows = list(
            session.run(
                "SHOW INDEXES YIELD type, state, labelsOrTypes, properties "
                "RETURN type, state, labelsOrTypes, properties"
            )
        )

    found_constraints: set[tuple[str, str]] = set()
    for row in constraint_rows:
        if row["type"] not in {"UNIQUENESS", "NODE_KEY", "NODE_PROPERTY_UNIQUENESS"}:
            continue
        labels = row["labelsOrTypes"] or []
        props = row["properties"] or []
        if labels and props:
            found_constraints.add((labels[0], props[0]))

    has_chunk_vector = False
    has_chunk_fulltext = False
    has_entity_fulltext = False
    has_event_fulltext = False
    has_metric_fulltext = False
    has_document_company_index = False
    has_document_period_start_index = False
    has_document_period_end_index = False
    has_quarter_period_start_index = False
    has_quarter_period_end_index = False
    has_section_raptor_vector = False
    has_company_metadata_vector = False

    online_checks: list[bool] = []

    for row in index_rows:
        idx_type = row["type"]
        state = row["state"]
        labels = row["labelsOrTypes"] or []
        props = row["properties"] or []

        if idx_type == "VECTOR" and "Chunk" in labels and "embedding" in props:
            has_chunk_vector = _CHUNK_VECTOR_REQUIRED_PROPERTIES.issubset(set(props))
            online_checks.append(state == "ONLINE")

        if idx_type == "FULLTEXT" and "Chunk" in labels and "text" in props:
            has_chunk_fulltext = True
            online_checks.append(state == "ONLINE")

        if idx_type == "FULLTEXT" and "Entity" in labels and "name" in props:
            has_entity_fulltext = True
            online_checks.append(state == "ONLINE")

        if idx_type == "FULLTEXT" and "Event" in labels and "name" in props:
            has_event_fulltext = True
            online_checks.append(state == "ONLINE")

        if idx_type == "FULLTEXT" and "Metric" in labels and "name" in props:
            has_metric_fulltext = True
            online_checks.append(state == "ONLINE")

        if idx_type == "RANGE" and "Document" in labels and "company_id" in props:
            has_document_company_index = True
            online_checks.append(state == "ONLINE")

        if idx_type == "RANGE" and "Document" in labels and "period_start" in props:
            has_document_period_start_index = True
            online_checks.append(state == "ONLINE")

        if idx_type == "RANGE" and "Document" in labels and "period_end" in props:
            has_document_period_end_index = True
            online_checks.append(state == "ONLINE")

        if idx_type == "RANGE" and "Quarter" in labels and "period_start" in props:
            has_quarter_period_start_index = True
            online_checks.append(state == "ONLINE")

        if idx_type == "RANGE" and "Quarter" in labels and "period_end" in props:
            has_quarter_period_end_index = True
            online_checks.append(state == "ONLINE")

        if idx_type == "VECTOR" and "Section" in labels and "summary_embedding" in props:
            has_section_raptor_vector = True
            online_checks.append(state == "ONLINE")

        if idx_type == "VECTOR" and "Company" in labels and "metadata_embedding" in props:
            has_company_metadata_vector = True
            online_checks.append(state == "ONLINE")

    indexes_ok = all(
        [
            has_chunk_vector,
            has_chunk_fulltext,
            has_entity_fulltext,
            has_event_fulltext,
            has_metric_fulltext,
            has_document_company_index,
            has_document_period_start_index,
            has_document_period_end_index,
            has_quarter_period_start_index,
            has_quarter_period_end_index,
            has_section_raptor_vector,
            has_company_metadata_vector,
        ]
    )

    status = {
        "constraints_ok": required_constraints.issubset(found_constraints),
        "indexes_ok": indexes_ok,
        "indexes_online": indexes_ok and all(online_checks),
    }
    return status


if __name__ == "__main__":
    main()
