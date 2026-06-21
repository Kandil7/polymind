"""Circuit Breaker pattern — prevents cascading failures.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail fast
- HALF_OPEN: Testing if service recovered

Usage:
    breaker = CircuitBreaker("qdrant", failure_threshold=3, recovery_timeout=30)

    if breaker.allow_request():
        try:
            result = call_service()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            # Use fallback
"""

from __future__ import annotations

import time
from enum import Enum

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external service calls.

    Prevents cascading failures by tracking success/failure rates
    and temporarily blocking requests to failing services.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        half_open_max: int = 1,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Service name for logging.
            failure_threshold: Failures before opening circuit.
            recovery_timeout: Seconds before trying again.
            half_open_max: Requests to test in half-open state.
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_count = 0
        self._just_failed_in_half_open = False

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            # Don't transition if we just failed in half-open
            if self._just_failed_in_half_open:
                return CircuitState.OPEN

            # Check if recovery timeout has elapsed
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_count = 0
                logger.info("circuit.half_open", service=self._name)
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_count < self._half_open_max:
                self._half_open_count += 1
                return True
            return False

        # OPEN state
        return False

    def record_success(self) -> None:
        """Record a successful request."""
        self._just_failed_in_half_open = False

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            # Recovery successful
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info("circuit.recovered", service=self._name)
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery — back to open
            self._state = CircuitState.OPEN
            self._just_failed_in_half_open = True
            logger.warning("circuit.recovery_failed", service=self._name)
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit.opened",
                service=self._name,
                failures=self._failure_count,
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_count = 0
        self._just_failed_in_half_open = False
        logger.info("circuit.reset", service=self._name)

    def stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self.state.value,
            "failures": self._failure_count,
            "threshold": self._failure_threshold,
        }


# ── Global circuit breakers ──────────────────────────────
_qdrant_breaker = CircuitBreaker("qdrant", failure_threshold=3, recovery_timeout=30)
_llm_breaker = CircuitBreaker("llm", failure_threshold=5, recovery_timeout=60)
_embedder_breaker = CircuitBreaker("embedder", failure_threshold=3, recovery_timeout=30)
_memory_breaker = CircuitBreaker("memory", failure_threshold=3, recovery_timeout=30)


def get_breaker(service: str) -> CircuitBreaker:
    """Get circuit breaker for a service."""
    breakers = {
        "qdrant": _qdrant_breaker,
        "llm": _llm_breaker,
        "embedder": _embedder_breaker,
        "memory": _memory_breaker,
    }
    return breakers.get(service, CircuitBreaker(service))


def all_breakers() -> list[CircuitBreaker]:
    """Get all circuit breakers."""
    return [_qdrant_breaker, _llm_breaker, _embedder_breaker, _memory_breaker]
