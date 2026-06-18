"""IRetriever interface — contract for all retrieval implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymind.domain.entities.chunk import DocumentChunk


class IRetriever(ABC):
    """Contract for retrieving relevant document chunks."""

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5, **kwargs: object
    ) -> list[DocumentChunk]:
        """Retrieve the top-k most relevant chunks for a query."""
        ...

    @abstractmethod
    async def index(self, chunks: list[DocumentChunk]) -> None:
        """Index document chunks into the retrieval store."""
        ...
