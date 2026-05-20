"""Main FastAPI application factory."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ingestion.api.config import get_settings
from ingestion.api.exceptions import IngestError, ingest_error_handler
from ingestion.api.routes.documents import router as documents_router
from ingestion.api.routes.ingestion import router as ingestion_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager (startup/shutdown)."""
    logger.info("RootAlpha Ingestion API starting up...")
    yield
    logger.info("RootAlpha Ingestion API shutting down...")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handler
    app.add_exception_handler(IngestError, ingest_error_handler)

    # Include routers
    app.include_router(documents_router)
    app.include_router(ingestion_router)

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/")
    async def root() -> dict:
        """Root endpoint."""
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/api/docs",
        }

    return app


# Create application instance
app = create_app()
