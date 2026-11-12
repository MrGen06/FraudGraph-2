from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.neo4j import neo4j_connection
from app.db.postgres import Base, engine
from app.services.model_service import FraudModelService

from app.api.v1.endpoints.alerts import router as alerts_router
from app.api.v1.endpoints.graph import router as graph_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.models import router as models_router
from app.api.v1.endpoints.transactions import router as transactions_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and preload models
    logger.info("Initializing FraudGraph backend services...")
    import sys
    if "pytest" not in sys.modules:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database schemas verified.")
        except Exception as e:
            logger.warning(f"Database schema auto-creation skipped: {e}")

    try:
        model_svc = FraudModelService.get_instance()
        logger.info(f"Model service initialized (Healthy: {model_svc.is_healthy()}).")
    except Exception as e:
        logger.error(f"Error pre-loading models: {e}")

    yield

    # Shutdown: clean up connection pools
    logger.info("Shutting down FraudGraph services...")
    try:
        neo4j_connection.close()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.warning(f"Error closing Neo4j driver: {e}")

    if "pytest" not in sys.modules:
        try:
            engine.dispose()
            logger.info("PostgreSQL engine connection pool disposed.")
        except Exception as e:
            logger.warning(f"Error disposing database engine: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Graph-powered financial fraud detection and investigation platform.",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(health_router)
app.include_router(transactions_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(alerts_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to FraudGraph API",
        "docs": "/docs",
        "endpoints": {
            "transactions": "/api/v1/transactions",
            "models": "/api/v1/models",
            "alerts": "/api/v1/alerts",
            "graph": "/api/v1/graph",
            "health": "/health",
        },
    }
