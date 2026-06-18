"""IGenerator interface — contract for answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymind.domain.entities.answer import Answer
    from polymind.domain.entities.chunk import DocumentChunk


class IGenerator(ABC):
    """Contract for generating answers from retrieved context."""

    @abstractmethod
    async def generate(
        self, query: str, context: list[DocumentChunk], **kwargs: object
    ) -> Answer:
        """Generate an answer grounded in the provided context."""
        ...
