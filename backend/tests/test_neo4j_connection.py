from app.db.neo4j import neo4j_connection


def test_neo4j_connection():
    assert neo4j_connection.verify_connection() is True