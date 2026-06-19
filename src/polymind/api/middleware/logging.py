"""Request logging middleware — structlog-based request/response logging."""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: TCH002
from starlette.requests import Request  # noqa: TCH002
from starlette.responses import Response  # noqa: TCH002

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing, status, and path."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start) * 1000

        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response
