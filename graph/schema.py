from neo4j import Driver


def create_schema(driver: Driver, embedding_dim: int = 3072) -> None:
    """
    Create Neo4j constraints and vector index for RootAlpha.
    Safe to call multiple times — all statements use IF NOT EXISTS.
    """
    with driver.session() as session:
        # --- Uniqueness constraints ---
        session.run(
            "CREATE CONSTRAINT company_ticker IF NOT EXISTS "
            "FOR (c:Company) REQUIRE c.ticker IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT company_name IF NOT EXISTS "
            "FOR (c:Company) REQUIRE c.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT document_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT section_id IF NOT EXISTS "
            "FOR (s:Section) REQUIRE s.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT table_id IF NOT EXISTS "
            "FOR (t:Table) REQUIRE t.id IS UNIQUE"
        )

        # --- Vector index on Chunk.embedding ---
        # Dimensions must match the model configured in GEMINI_EMBEDDING_MODEL.
        session.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (ch:Chunk) ON (ch.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $embedding_dim,
                `vector.similarity_function`: 'cosine'
            }}
        """, embedding_dim=embedding_dim)

        # --- Vector index on Table.embedding ---
        # Tables are embedded with their section heading as context prefix
        # so they are independently retrievable by semantic similarity.
        session.run("""
            CREATE VECTOR INDEX table_embeddings IF NOT EXISTS
            FOR (t:Table) ON (t.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $embedding_dim,
                `vector.similarity_function`: 'cosine'
            }}
        """, embedding_dim=embedding_dim)

    print(f"[schema] Constraints and vector indexes created (dim={embedding_dim}).")
