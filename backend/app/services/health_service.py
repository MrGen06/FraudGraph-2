from sqlalchemy import text
from app.db.postgres import engine
from app.db.neo4j import neo4j_connection


def check_postgres() -> str:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "unavailable"


def check_neo4j() -> str:
    try:
        neo4j_connection.verify_connection()
        return "ok"
    except Exception:
        return "unavailable"


def get_health_status() -> dict:
    checks = {
        "postgres": check_postgres(),
        "neo4j": check_neo4j(),
    }

    overall_status = (
        "healthy"
        if all(status == "ok" for status in checks.values())
        else "not_ready"
    )

    return {
        "status": overall_status,
        "checks": checks,
    }