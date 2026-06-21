"""Graceful degradation manager — provides fallbacks when services fail.

Tracks service health via circuit breakers and provides degraded-mode
alternatives for each component in the pipeline.

Degradation modes:
- Qdrant down → Use LLM parametric knowledge only
- Embedding down → Use keyword search fallback
- LLM down → Use heuristic classification and extractive generation
- Memory down → Skip memory recall (operate without context)
"""

from __future__ import annotations

import structlog

from polymind.infrastructure.circuit_breaker import (
    CircuitBreaker,
    all_breakers,
    get_breaker,
)

logger = structlog.get_logger()


class DegradationManager:
    """Manages graceful degradation across all services.

    Provides a unified interface to check service health and
    get degraded-mode behavior for each component.
    """

    def __init__(self) -> None:
        """Initialize the degradation manager."""
        self._breakers: dict[str, CircuitBreaker] = {
            "qdrant": get_breaker("qdrant"),
            "llm": get_breaker("llm"),
            "embedder": get_breaker("embedder"),
            "memory": get_breaker("memory"),
        }

    def is_service_healthy(self, service: str) -> bool:
        """Check if a service is available.

        Args:
            service: Service name (qdrant, llm, embedder, memory).

        Returns:
            True if service is healthy (circuit closed or half-open).
        """
        breaker = self._breakers.get(service)
        if breaker is None:
            return True
        return breaker.allow_request()

    def record_service_success(self, service: str) -> None:
        """Record successful service call."""
        breaker = self._breakers.get(service)
        if breaker:
            breaker.record_success()

    def record_service_failure(self, service: str) -> None:
        """Record failed service call."""
        breaker = self._breakers.get(service)
        if breaker:
            breaker.record_failure()

    def get_degradation_mode(self) -> dict[str, str]:
        """Get current degradation mode for all services.

        Returns:
            Dict mapping service names to their status.
        """
        modes = {}
        for name, breaker in self._breakers.items():
            if breaker.state.value == "closed":
                modes[name] = "healthy"
            elif breaker.state.value == "half_open":
                modes[name] = "recovering"
            else:
                modes[name] = "degraded"
        return modes

    def should_skip_retrieval(self) -> bool:
        """Check if retrieval should be skipped due to service degradation.

        Returns:
            True if both Qdrant and embedder are down.
        """
        qdrant_ok = self.is_service_healthy("qdrant")
        embedder_ok = self.is_service_healthy("embedder")
        # Only skip if BOTH are down (either one working allows degraded retrieval)
        return not qdrant_ok and not embedder_ok

    def should_use_heuristic_classification(self) -> bool:
        """Check if LLM classification should fall back to keywords.

        Returns:
            True if LLM is unavailable.
        """
        return not self.is_service_healthy("llm")

    def should_use_heuristic_generation(self) -> bool:
        """Check if LLM generation should fall back to extractive.

        Returns:
            True if LLM is unavailable.
        """
        return not self.is_service_healthy("llm")

    def should_skip_memory(self) -> bool:
        """Check if memory recall should be skipped.

        Returns:
            True if memory service is unavailable.
        """
        return not self.is_service_healthy("memory")

    def get_status(self) -> dict:
        """Get full degradation status for health endpoint."""
        modes = self.get_degradation_mode()
        healthy_count = sum(1 for v in modes.values() if v == "healthy")
        total = len(modes)

        return {
            "services": modes,
            "healthy_count": healthy_count,
            "total_services": total,
            "overall": "healthy" if healthy_count == total else "degraded",
        }


# ── Global instance ──────────────────────────────────────
degradation = DegradationManager()
