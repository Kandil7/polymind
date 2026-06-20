"""Rate limiting middleware — per-IP request throttling.

Uses a sliding window counter to limit requests per IP address.
Protects against abuse without requiring external dependencies.
"""

from __future__ import annotations

import time
from collections import defaultdict

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger()

# ── Default limits ──────────────────────────────────────
DEFAULT_RATE_LIMIT = 30  # requests per window
DEFAULT_WINDOW_SECONDS = 60  # 1 minute window
QUERY_RATE_LIMIT = 10  # stricter for expensive endpoints
INGEST_RATE_LIMIT = 5  # very strict for resource-heavy ingestion


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter.

    Tracks request counts per IP using an in-memory dict.
    Old entries are pruned on each request to prevent memory leaks.
    """

    def __init__(
        self,
        app,
        default_limit: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        super().__init__(app)
        self._default_limit = default_limit
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = self._get_client_ip(request)
        path = request.url.path

        # Select rate limit based on endpoint
        limit = self._get_limit(path)

        # Check rate limit
        now = time.time()
        window_start = now - self._window_seconds

        # Prune old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= limit:
            logger.warning(
                "rate_limit.exceeded",
                ip=client_ip,
                path=path,
                count=len(self._requests[client_ip]),
                limit=limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": self._window_seconds,
                    "retry_after": self._window_seconds,
                },
                headers={"Retry-After": str(self._window_seconds)},
            )

        # Record request
        self._requests[client_ip].append(now)

        response = await call_next(request)
        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_limit(self, path: str) -> int:
        """Select rate limit based on endpoint path."""
        if path.startswith("/query"):
            return QUERY_RATE_LIMIT
        elif path.startswith("/ingest"):
            return INGEST_RATE_LIMIT
        return self._default_limit
