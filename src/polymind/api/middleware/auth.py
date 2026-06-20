"""API Key authentication middleware — validates Bearer tokens.

Checks the Authorization header for a valid API key.
Skips authentication for health check and docs endpoints.
"""

from __future__ import annotations

import os

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger()

# ── Endpoints that skip authentication ──────────────────
PUBLIC_PATHS = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/feedback",  # Feedback can be submitted without auth
})


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validates API key from Authorization header.

    Usage:
        Set POLYMIND_API_KEY env var to enable authentication.
        If not set, all requests are allowed (development mode).

    Headers:
        Authorization: Bearer <api_key>
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._api_key = os.getenv("POLYMIND_API_KEY", "")

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip auth for public endpoints
        if path in PUBLIC_PATHS or path.startswith("/docs"):
            return await call_next(request)

        # Skip auth if no API key configured (dev mode)
        if not self._api_key:
            return await call_next(request)

        # Validate API key
        auth_header = request.headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            logger.warning("auth.missing_header", path=path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing Authorization header",
                    "hint": "Include 'Authorization: Bearer <api_key>' header",
                },
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        if token != self._api_key:
            logger.warning("auth.invalid_key", path=path)
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key"},
            )

        return await call_next(request)
