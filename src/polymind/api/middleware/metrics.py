"""Prometheus metrics middleware — exposes /metrics endpoint."""

from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: TCH002
from starlette.requests import Request  # noqa: TCH002
from starlette.responses import Response  # noqa: TCH002

# ── Metrics ─────────────────────────────────────────────
QUERY_TOTAL = Counter(
    "polymind_queries_total",
    "Total queries processed",
    ["modality", "passed_critic"],
)

QUERY_LATENCY = Histogram(
    "polymind_query_latency_seconds",
    "Query processing latency",
    ["modality"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

ACTIVE_REQUESTS = Gauge(
    "polymind_active_requests",
    "Number of active requests",
)

FAITHFULNESS_SCORE = Histogram(
    "polymind_faithfulness_score",
    "Critic faithfulness score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collects Prometheus metrics for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        ACTIVE_REQUESTS.inc()
        start = time.time()

        response = await call_next(request)

        duration = time.time() - start
        ACTIVE_REQUESTS.dec()

        # Record latency
        QUERY_LATENCY.labels(
            modality="unknown"
        ).observe(duration)

        return response


def record_query(modality: str, passed: bool, faithfulness: float) -> None:
    """Record query metrics after processing."""
    QUERY_TOTAL.labels(
        modality=modality,
        passed_critic=str(passed),
    ).inc()

    FAITHFULNESS_SCORE.observe(faithfulness)


def metrics_endpoint() -> Response:
    """Return Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )
