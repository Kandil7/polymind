"""IEvaluator interface — contract for answer evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymind.domain.entities.query import ScoreResult


class IEvaluator(ABC):
    """Contract for evaluating answer quality (Critic agent)."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        **kwargs: object,
    ) -> dict[str, ScoreResult]:
        """Evaluate answer quality and return named scores."""
        ...
