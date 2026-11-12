import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

db_url = settings.DATABASE_URL

# Only require SSL if connecting to managed remote cloud databases (e.g. Supabase, AWS RDS)
connect_args = {}
if "supabase" in db_url or "aws-" in db_url or "pooler.supabase.com" in db_url:
    connect_args = {
        "connect_timeout": 10,
        "sslmode": "require",
    }

if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_size=5 if not db_url.startswith("sqlite") else None,
    max_overflow=5 if not db_url.startswith("sqlite") else None,
    connect_args=connect_args,
)


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
