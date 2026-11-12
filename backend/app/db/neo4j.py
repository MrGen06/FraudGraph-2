from neo4j import GraphDatabase

from app.core.config import get_settings

settings = get_settings()


class Neo4jConnection:

    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(
                    settings.NEO4J_USERNAME,
                    settings.NEO4J_PASSWORD,
                ),
            )
        return self._driver

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None


neo4j_connection = Neo4jConnection()