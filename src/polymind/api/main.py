"""FastAPI application factory."""

from __future__ import annotations

import structlog
from fastapi import FastAPI

from polymind import __version__
from polymind.api.routes.health import router as health_router

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PolyMind",
        description="Self-evaluating, multimodal, multi-agent knowledge assistant",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(health_router)

    @app.on_event("startup")
    async def startup() -> None:
        logger.info("polymind.startup", version=__version__)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("polymind.shutdown")

    return app


app = create_app()
