"""FastAPI application factory — wires all routes and middleware."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from polymind import __version__
from polymind.api.middleware.auth import APIKeyAuthMiddleware
from polymind.api.middleware.logging import RequestLoggingMiddleware
from polymind.api.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from polymind.api.middleware.rate_limit import RateLimitMiddleware
from polymind.api.routes.eval import router as eval_router
from polymind.api.routes.health import router as health_router
from polymind.api.routes.ingest import router as ingest_router
from polymind.api.routes.query import router as query_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    # ── Startup ──────────────────────────────────────────
    logger.info("polymind.startup", version=__version__)

    # Initialize tracing
    try:
        from polymind.infrastructure.tracing import _setup_tracer

        _setup_tracer()
    except Exception as e:
        logger.warning("tracing.init_failed", error=str(e))

    # Pre-warm Qdrant connection
    try:
        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )

        client = get_qdrant_client()
        collections = client.get_collections()
        logger.info(
            "polymind.qdrant.connected",
            collections=len(collections.collections),
        )
    except Exception as e:
        logger.warning("polymind.qdrant.unavailable", error=str(e))

    yield

    # ── Shutdown ─────────────────────────────────────────
    # Flush traces
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass

    logger.info("polymind.shutdown")


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
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Security ─────────────────────────────────────────
    app.add_middleware(APIKeyAuthMiddleware)

    # ── Rate Limiting ────────────────────────────────────
    app.add_middleware(RateLimitMiddleware)

    # ── Observability ────────────────────────────────────
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

    return app


app = create_app()
