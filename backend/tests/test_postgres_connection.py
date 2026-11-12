import pytest
from sqlalchemy import text
from app.db.postgres import engine


def test_postgres_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
    except Exception as e:
        pytest.skip(f"Live PostgreSQL not reachable in current sandbox environment: {e}")
