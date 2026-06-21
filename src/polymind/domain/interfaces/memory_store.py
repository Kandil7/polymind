"""IMemoryStore interface — contract for memory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IMemoryStore(ABC):
    """Contract for agent memory stores (episodic, semantic, procedural).

    Subclasses implement domain-specific store/recall methods.
    The generic methods below provide a common interface for
    orchestration code that doesn't know the concrete store type.

    Note: Concrete stores may have different store() signatures for
    domain-specific parameters. The generic store() method provides
    a minimal contract for orchestration code.
    """

    @abstractmethod
    def store(self, *args: Any, **kwargs: Any) -> None:
        """Store a memory entry.

        Concrete implementations accept domain-specific arguments:
        - EpisodicStore: store(query, answer, faithfulness, modality)
        - SemanticStore: store(fact, source_query)
        - ProceduralStore: store(task_type, steps, success_score)
        """
        ...

    @abstractmethod
    def recall(self, query: str, top_k: int = 5) -> list[Any]:
        """Recall relevant memories for a query."""
        ...

    @abstractmethod
    def consolidate(self) -> None:
        """Consolidate episodic memories into semantic facts."""
        ...
