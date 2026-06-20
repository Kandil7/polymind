"""Tests for Circuit Breaker and Graceful Degradation."""

from __future__ import annotations

import time

from polymind.infrastructure.circuit_breaker import CircuitBreaker, CircuitState
from polymind.infrastructure.degradation import DegradationManager


class TestCircuitBreaker:
    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_when_closed(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_blocks_when_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # With 0 timeout, should transition to half-open immediately
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovers_on_success(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.allow_request()  # Transition to half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_during_half_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.allow_request()  # Transition to half-open
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should have decremented failure count
        assert cb._failure_count == 1

    def test_reset(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_stats(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=5)
        stats = cb.stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failures"] == 0
        assert stats["threshold"] == 5


class TestDegradationManager:
    def test_initial_all_healthy(self) -> None:
        dm = DegradationManager()
        modes = dm.get_degradation_mode()
        assert all(v == "healthy" for v in modes.values())

    def test_service_failure_marks_degraded(self) -> None:
        dm = DegradationManager()
        for _ in range(3):
            dm.record_service_failure("qdrant")
        modes = dm.get_degradation_mode()
        assert modes["qdrant"] == "degraded"

    def test_should_skip_retrieval_when_qdrant_down(self) -> None:
        dm = DegradationManager()
        for _ in range(3):
            dm.record_service_failure("qdrant")
        assert dm.should_skip_retrieval() is True

    def test_should_not_skip_when_only_one_down(self) -> None:
        dm = DegradationManager()
        for _ in range(3):
            dm.record_service_failure("qdrant")
        # Embedder is still healthy
        assert dm.should_skip_retrieval() is False

    def test_should_use_heuristic_when_llm_down(self) -> None:
        dm = DegradationManager()
        for _ in range(5):
            dm.record_service_failure("llm")
        assert dm.should_use_heuristic_classification() is True
        assert dm.should_use_heuristic_generation() is True

    def test_get_status(self) -> None:
        dm = DegradationManager()
        status = dm.get_status()
        assert "services" in status
        assert "healthy_count" in status
        assert "overall" in status
        assert status["overall"] == "healthy"

    def test_degraded_status(self) -> None:
        dm = DegradationManager()
        for _ in range(3):
            dm.record_service_failure("qdrant")
        status = dm.get_status()
        assert status["overall"] == "degraded"
