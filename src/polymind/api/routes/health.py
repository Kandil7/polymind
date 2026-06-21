"""Health check endpoint — verifies service and dependency health."""

from __future__ import annotations

import os

import structlog
from fastapi import APIRouter

from polymind import __version__
from polymind.api.schemas.health import HealthResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status with dependency checks.

    Checks Qdrant connectivity, LLM API key, and circuit breaker status.
    """
    checks: dict[str, object] = {}
    overall_status = "ok"

    # Check Qdrant
    try:
        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )

        client = get_qdrant_client()
        collections = client.get_collections()
        checks["qdrant"] = {
            "status": "ok",
            "collections": len(collections.collections),
        }
    except Exception as e:
        checks["qdrant"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"

    # Check LLM API key
    groq_key = os.getenv("GROQ_API_KEY", "")
    checks["llm"] = {
        "status": "ok" if groq_key else "no_api_key",
        "provider": "groq",
    }
    if not groq_key:
        overall_status = "degraded"

    # Check embedding model availability (lightweight — no model load)
    checks["embedder"] = {
        "status": "configured",
        "model": "BAAI/bge-m3",
    }

    # Check circuit breaker status
    try:
        from polymind.infrastructure.degradation import degradation

        degradation_status = degradation.get_status()
        checks["degradation"] = degradation_status

        # Update overall status based on degradation
        if degradation_status["overall"] == "degraded":
            overall_status = "degraded"
    except Exception as e:
        logger.debug("health.degradation_check.failed", error=str(e))

    return HealthResponse(
        status=overall_status,
        version=__version__,
        service="polymind",
        checks=checks,
    )
