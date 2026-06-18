"""IMemoryStore interface — contract for memory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IMemoryStore(ABC):
    """Contract for agent memory stores (episodic, semantic, procedural)."""

    @abstractmethod
    async def store(self, key: str, value: Any, **kwargs: object) -> None:
        """Store a memory entry."""
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5) -> list[Any]:
        """Recall relevant memories for a query."""
        ...

    @abstractmethod
    async def consolidate(self) -> None:
        """Consolidate episodic memories into semantic facts."""
        ...
