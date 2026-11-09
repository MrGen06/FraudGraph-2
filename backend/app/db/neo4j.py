from neo4j import GraphDatabase

from app.core.config import get_settings

settings = get_settings()


class Neo4jConnection:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD,
            ),
        )

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        self.driver.close()


neo4j_connection = Neo4jConnection()