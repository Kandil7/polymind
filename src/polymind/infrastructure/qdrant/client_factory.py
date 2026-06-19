"""Qdrant client factory — singleton client management."""

from __future__ import annotations

from functools import lru_cache

import structlog
from qdrant_client import QdrantClient

logger = structlog.get_logger()

DEFAULT_URL = "http://localhost:6333"


@lru_cache(maxsize=1)
def get_qdrant_client(url: str = DEFAULT_URL) -> QdrantClient:
    """Get or create a cached Qdrant client.

    Args:
        url: Qdrant server URL.

    Returns:
        Cached QdrantClient instance.
    """
    logger.info("qdrant.client.create", url=url)
    return QdrantClient(url=url)


def reset_qdrant_client() -> None:
    """Reset the cached client (for testing)."""
    get_qdrant_client.cache_clear()
