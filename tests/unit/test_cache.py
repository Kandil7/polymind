"""Tests for In-Memory Cache."""

from __future__ import annotations

import time

from polymind.infrastructure.cache import (
    TTLCache,
    cache_key,
    cache,
    cached_classification,
    store_classification,
)


class TestTTLCache:
    def test_set_and_get(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        c.set("key1", "value1")
        assert c.get("key1") == "value1"

    def test_get_missing_key(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        assert c.get("nonexistent") is None

    def test_expiry(self) -> None:
        c = TTLCache(max_size=100, default_ttl=1)
        c.set("key1", "value1", ttl=0)  # Immediate expiry
        time.sleep(0.01)
        assert c.get("key1") is None

    def test_lru_eviction(self) -> None:
        c = TTLCache(max_size=3, default_ttl=60)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)  # Should evict "a"
        assert c.get("a") is None
        assert c.get("d") == 4

    def test_delete(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        c.set("key1", "value1")
        assert c.delete("key1") is True
        assert c.get("key1") is None
        assert c.delete("key1") is False

    def test_clear(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_stats(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        c.set("key1", "value1")
        c.get("key1")  # hit
        c.get("key2")  # miss
        stats = c.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_update_existing_key(self) -> None:
        c = TTLCache(max_size=100, default_ttl=60)
        c.set("key1", "old")
        c.set("key1", "new")
        assert c.get("key1") == "new"
        assert c.stats()["size"] == 1


class TestCacheKey:
    def test_deterministic(self) -> None:
        key1 = cache_key("a", "b", "c")
        key2 = cache_key("a", "b", "c")
        assert key1 == key2

    def test_different_inputs_different_keys(self) -> None:
        key1 = cache_key("a", "b")
        key2 = cache_key("a", "c")
        assert key1 != key2

    def test_returns_hex_string(self) -> None:
        key = cache_key("test")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


class TestCachedClassification:
    def test_store_and_retrieve(self) -> None:
        store_classification("What is RAG?", "factual_qa")
        result = cached_classification("What is RAG?")
        assert result == "factual_qa"

    def test_cache_miss(self) -> None:
        result = cached_classification("unique query that should not be cached")
        assert result is None

    def test_different_queries_different_results(self) -> None:
        store_classification("query1", "summarization")
        store_classification("query2", "comparison")
        assert cached_classification("query1") == "summarization"
        assert cached_classification("query2") == "comparison"


class TestGlobalCache:
    def test_global_cache_exists(self) -> None:
        assert cache is not None
        assert isinstance(cache, TTLCache)

    def test_global_cache_stats(self) -> None:
        stats = cache.stats()
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
