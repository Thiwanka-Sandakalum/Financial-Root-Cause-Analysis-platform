"""Neo4j schema initialization and migration helpers."""
import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


async def init_schema(driver: AsyncDriver) -> None:
    """
    Initialize Neo4j schema with constraints and indexes.
    
    Args:
        driver: Neo4j async driver
    """
    queries = [
        # Job constraints and indexes
        """
        CREATE CONSTRAINT job_id IF NOT EXISTS
        FOR (j:Job) REQUIRE j.id IS UNIQUE
        """,
        """
        CREATE INDEX job_status IF NOT EXISTS
        FOR (j:Job) ON (j.status)
        """,
        """
        CREATE INDEX job_created_at IF NOT EXISTS
        FOR (j:Job) ON (j.created_at)
        """,
        
        # Document constraints and indexes
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
        """,
        """
        CREATE INDEX document_ticker IF NOT EXISTS
        FOR (d:Document) ON (d.ticker)
        """,
        """
        CREATE INDEX document_status IF NOT EXISTS
        FOR (d:Document) ON (d.status)
        """,
        
        # Section constraints
        """
        CREATE CONSTRAINT section_id IF NOT EXISTS
        FOR (s:Section) REQUIRE s.id IS UNIQUE
        """,
        
        # Chunk constraints
        """
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """,
        
        # Entity constraints
        """
        CREATE CONSTRAINT entity_id IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """,
    ]

    async with driver.session() as session:
        for query in queries:
            try:
                await session.run(query)
                logger.info(f"Executed: {query.strip()[:50]}...")
            except Exception as e:
                logger.warning(f"Schema operation failed (may already exist): {e}")


async def verify_connection(driver: AsyncDriver) -> bool:
    """
    Verify Neo4j connection.
    
    Args:
        driver: Neo4j async driver
        
    Returns:
        True if connection is successful
    """
    try:
        await driver.verify_connectivity()
        logger.info("Neo4j connection verified")
        return True
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return False
