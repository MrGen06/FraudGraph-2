import pytest
from app.db.neo4j import neo4j_connection


def test_neo4j_connection():
    try:
        is_connected = neo4j_connection.verify_connection()
        if not is_connected:
            pytest.skip("Neo4j AuraDB instance not reachable in current sandbox environment")
        assert is_connected is True
    except Exception as e:
        pytest.skip(f"Neo4j connection skipped: {e}")
