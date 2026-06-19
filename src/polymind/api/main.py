"""FastAPI application factory — wires all routes and middleware."""

from __future__ import annotations

import structlog
from fastapi import FastAPI

from polymind import __version__
from polymind.api.middleware.logging import RequestLoggingMiddleware
from polymind.api.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from polymind.api.routes.eval import router as eval_router
from polymind.api.routes.health import router as health_router
from polymind.api.routes.ingest import router as ingest_router
from polymind.api.routes.query import router as query_router

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PolyMind",
        description=(
            "Self-evaluating, multimodal, multi-agent knowledge assistant. "
            "Routes queries across 7+ HuggingFace task types with a Critic agent "
            "that self-evaluates outputs using RAGAS before delivery."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Routes ───────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(query_router, prefix="/query", tags=["Query"])
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
    app.include_router(eval_router, prefix="/eval", tags=["Eval"])

    # ── Metrics endpoint ─────────────────────────────────
    @app.get("/metrics")
    async def metrics():
        return metrics_endpoint()

    # ── Lifecycle ────────────────────────────────────────
    @app.on_event("startup")
    async def startup() -> None:
        logger.info("polymind.startup", version=__version__)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("polymind.shutdown")

    return app


app = create_app()
