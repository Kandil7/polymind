"""End-to-end tests — infrastructure component validation.

Tests caching, degradation, feedback, and other infrastructure components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestE2ECache:
    """Test the caching layer end-to-end."""

    def test_cache_set_get_cycle(self) -> None:
        """Should store and retrieve values."""
        from polymind.infrastructure.cache import TTLCache

        cache = TTLCache(max_size=100, default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_expiration(self) -> None:
        """Should expire entries after TTL."""
        import time
        from polymind.infrastructure.cache import TTLCache

        cache = TTLCache(max_size=100, default_ttl=0)
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self) -> None:
        """Should evict oldest entries when full."""
        from polymind.infrastructure.cache import TTLCache

        cache = TTLCache(max_size=2, default_ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("c") == 3


@pytest.mark.e2e
class TestE2ECircuitBreaker:
    """Test circuit breaker end-to-end."""

    def test_circuit_breaker_lifecycle(self) -> None:
        """Test full circuit breaker lifecycle."""
        from polymind.infrastructure.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=0)

        # Closed state
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

        # Failures open the circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

        # Recovery transitions to half-open
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Success closes the circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


@pytest.mark.e2e
class TestE2EFeedback:
    """Test feedback system end-to-end."""

    def test_feedback_lifecycle(self) -> None:
        """Test full feedback lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from polymind.infrastructure.feedback import FeedbackStore

            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            # Record feedback
            store.record(
                query_id="q1",
                query="What is RAG?",
                rating=5,
                feedback="thumbs_up",
                metadata={"intent": "factual_qa"},
            )

            # Check stats
            stats = store.get_stats()
            assert stats["total"] == 1
            assert stats["average_rating"] == 5.0

            # Check by intent
            by_intent = store.get_satisfaction_by_intent()
            assert by_intent["factual_qa"] == 5.0

            # Persistence
            store2 = FeedbackStore(store_path=f"{tmpdir}/feedback.json")
            assert store2.count() == 1


@pytest.mark.e2e
class TestE2EConsolidation:
    """Test memory consolidation end-to-end."""

    def test_consolidation_pipeline(self) -> None:
        """Test consolidation pipeline lifecycle."""
        from polymind.infrastructure.memory.consolidation import (
            ConsolidationPipeline,
        )

        pipeline = ConsolidationPipeline(user_id="e2e_test")

        # Buffer should be empty initially
        assert pipeline.get_stats()["buffer_size"] == 0

        # Add episodes
        for i in range(3):
            pipeline._episode_buffer.append({
                "query": f"Query {i}",
                "answer": f"Answer {i}",
                "scores": {"faithfulness": 0.9},
                "modality": "text",
            })

        # Extract procedures
        procedures = pipeline._extract_procedures()
        assert len(procedures) == 3
        assert all(p["task_type"] == "general" for p in procedures)


@pytest.mark.e2e
class TestE2EChunking:
    """Test chunking strategies end-to-end."""

    def test_chunking_strategy_selection(self) -> None:
        """Should select correct chunker for file type."""
        from polymind.infrastructure.rag.chunk_strategy import select_chunker
        from polymind.infrastructure.rag.chunker import (
            RecursiveChunker,
            TableChunker,
        )
        from polymind.infrastructure.rag.chunk_strategy import CodeChunker

        assert isinstance(select_chunker("txt"), RecursiveChunker)
        assert isinstance(select_chunker("csv"), TableChunker)
        assert isinstance(select_chunker("py"), CodeChunker)

    def test_code_chunker_splits_functions(self) -> None:
        """Code chunker should split at function boundaries."""
        from polymind.infrastructure.rag.chunk_strategy import CodeChunker

        code = """
def hello():
    print("hello")

def world():
    print("world")
"""
        chunker = CodeChunker()
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1
        assert all(c["text"].strip() for c in chunks)


@pytest.mark.e2e
class TestE2EMoA:
    """Test Mixture-of-Agents end-to-end."""

    def test_moa_configs(self) -> None:
        """MoA should have multiple agent configurations."""
        from polymind.infrastructure.moa import AGENT_CONFIGS, get_agent_configs

        assert len(AGENT_CONFIGS) == 3
        temps = [c["temperature"] for c in AGENT_CONFIGS]
        assert len(set(temps)) > 1  # Different temperatures for diversity

        configs = get_agent_configs(2)
        assert len(configs) == 2


@pytest.mark.e2e
class TestE2EHyDE:
    """Test HyDE query expansion end-to-end."""

    def test_hyde_expansion(self) -> None:
        """HyDE should expand complex queries."""
        from polymind.infrastructure.hyde import expand_query_hyde

        result = expand_query_hyde("What is RAG?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_should_use_hyde(self) -> None:
        """Should determine when to use HyDE."""
        from polymind.application.agents.rag_node import _should_use_hyde

        assert _should_use_hyde("What is RAG?") is False
        assert _should_use_hyde("Compare RAG and fine-tuning") is True
        assert _should_use_hyde("How does RAG work?") is True


@pytest.mark.e2e
class TestE2EDegradation:
    """Test graceful degradation end-to-end."""

    def test_degradation_manager(self) -> None:
        """Degradation manager should track service health."""
        from polymind.infrastructure.degradation import DegradationManager

        dm = DegradationManager()
        status = dm.get_status()
        assert "services" in status
        assert "overall" in status
