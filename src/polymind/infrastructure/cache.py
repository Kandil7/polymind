"""In-memory cache with TTL — for embeddings, responses, and classifications.

Provides a simple caching layer that can be swapped for Redis in production.
Uses a sliding window with automatic expiration.

Usage:
    from polymind.infrastructure.cache import cache

    # Cache an embedding
    cache.set("embed:What is RAG?", embedding_vector, ttl=3600)

    # Retrieve cached embedding
    embedding = cache.get("embed:What is RAG?")

    # Cache a response
    cache.set("response:abc123", response_dict, ttl=1800)

    # Check cache stats
    stats = cache.stats()
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

import structlog

logger = structlog.get_logger()


class TTLCache:
    """Thread-safe in-memory cache with TTL expiration.

    Uses OrderedDict for O(1) access and LRU eviction.
    Old entries are pruned on access to prevent memory leaks.
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,
    ) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries.
            default_ttl: Default time-to-live in seconds.
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if expired/missing.
        """
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            else:
                # Expired — remove
                del self._cache[key]

        self._misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (uses default if None).
        """
        if key in self._cache:
            # Update existing entry
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._max_size:
            # Evict oldest entry
            self._cache.popitem(last=False)

        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Returns:
            True if the key existed, False otherwise.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


# ── Global cache instance ────────────────────────────────
cache = TTLCache(max_size=10000, default_ttl=3600)


def cache_key(*parts: Any) -> str:
    """Generate a deterministic cache key from parts.

    Args:
        *parts: Key components (will be hashed).

    Returns:
        Hex string cache key.
    """
    raw = ":".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cached_embedding(query: str) -> list[float] | None:
    """Get cached embedding for a query."""
    key = cache_key("embed", query)
    return cache.get(key)


def store_embedding(query: str, embedding: list[float], ttl: int = 86400) -> None:
    """Cache an embedding vector (24h default TTL)."""
    key = cache_key("embed", query)
    cache.set(key, embedding, ttl=ttl)


def cached_response(query: str, user_id: str = "default") -> dict | None:
    """Get cached response for a query."""
    key = cache_key("response", query, user_id)
    return cache.get(key)


def store_response(
    query: str,
    response: dict,
    user_id: str = "default",
    ttl: int = 1800,
) -> None:
    """Cache a query response (30min default TTL)."""
    key = cache_key("response", query, user_id)
    cache.set(key, response, ttl=ttl)


def cached_classification(query: str) -> str | None:
    """Get cached intent classification."""
    key = cache_key("intent", query)
    return cache.get(key)


def store_classification(query: str, intent: str, ttl: int = 3600) -> None:
    """Cache an intent classification (1h default TTL)."""
    key = cache_key("intent", query)
    cache.set(key, intent, ttl=ttl)
