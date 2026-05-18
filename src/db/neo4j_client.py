import logging

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError

from src.config import Settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._driver: Driver | None = None
        self._resolved_database: str | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_username, self._settings.neo4j_password),
            )
        return self._driver

    def verify_connectivity(self) -> None:
        self.resolve_database()

    def resolve_database(self) -> str:
        if self._resolved_database is not None:
            return self._resolved_database

        configured_database = self._settings.neo4j_database
        fallback_database = "neo4j"

        self.driver.verify_connectivity()

        if self._database_exists(configured_database):
            self._resolved_database = configured_database
            return configured_database

        if configured_database != fallback_database and self._database_exists(fallback_database):
            logger.warning(
                "Configured Neo4j database '%s' was not found; falling back to '%s'.",
                configured_database,
                fallback_database,
            )
            self._settings.neo4j_database = fallback_database
            self._resolved_database = fallback_database
            return fallback_database

        raise RuntimeError(
            "Configured Neo4j database "
            f"'{configured_database}' was not found and fallback database '{fallback_database}' "
            "is unavailable. Update NEO4J_DATABASE in your environment."
        )

    def _database_exists(self, database: str) -> bool:
        try:
            with self.driver.session(database=database) as session:
                session.run("RETURN 1 AS ok").consume()
            return True
        except ClientError as exc:
            if exc.code == "Neo.ClientError.Database.DatabaseNotFound":
                return False
            raise

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
