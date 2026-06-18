"""Adaptive Retriever — selects retrieval strategy based on query complexity."""

from __future__ import annotations

from enum import Enum

import structlog

from polymind.domain.entities.chunk import DocumentChunk  # noqa: TCH001
from polymind.domain.interfaces.retriever import IRetriever

logger = structlog.get_logger()


class RetrievalStrategy(str, Enum):
    """Strategy for retrieving context from the knowledge base."""

    SKIP = "skip"
    STANDARD = "standard"
    HIPPORAG = "hipporag"
    SPECULATIVE = "speculative"
    SPARC = "sparc"


class AdaptiveRetriever(IRetriever):
    """Selects the best retrieval strategy based on query complexity.

    Routes to different retrievers:
    - SKIP: Simple factual questions (LLM can answer from params)
    - STANDARD: Single-hop document lookup (Qdrant vector search)
    - HIPPORAG: Multi-hop reasoning (Knowledge Graph + PPR)
    - SPECULATIVE: Draft first, verify after
    """

    def __init__(self, retrievers: dict[str, IRetriever]) -> None:
        """Initialize adaptive retriever.

        Args:
            retrievers: Map of strategy name to IRetriever implementation.
        """
        self._retrievers = retrievers

    @property
    def name(self) -> str:
        """Return the retriever name."""
        return "adaptive"

    async def retrieve(
        self, query: str, top_k: int = 5, **kwargs: object
    ) -> list[DocumentChunk]:
        """Retrieve using the best strategy for the query.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Retrieved DocumentChunks.
        """
        strategy = self._classify_query(query)

        logger.info("adaptive.classify", strategy=strategy.value, query_length=len(query))

        if strategy == RetrievalStrategy.SKIP:
            return []

        retriever = self._retrievers.get(strategy.value)
        if retriever is None:
            # Fallback to standard
            retriever = self._retrievers.get("standard")
            if retriever is None:
                return []

        return await retriever.retrieve(query, top_k=top_k, **kwargs)

    async def index(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks using the standard retriever.

        Args:
            chunks: List of DocumentChunks to index.
        """
        standard = self._retrievers.get("standard")
        if standard:
            await standard.index(chunks)

    def _classify_query(self, query: str) -> RetrievalStrategy:
        """Classify query complexity to select retrieval strategy."""
        q = query.lower()

        # Multi-hop signals
        multi_hop_patterns = [
            "who founded", "which company", "what led to",
            "connection between", "relationship between",
            "compare", "difference between", "how does",
            "why did", "what caused",
        ]
        if any(pattern in q for pattern in multi_hop_patterns):
            return RetrievalStrategy.HIPPORAG

        # Skip simple factual questions
        simple_patterns = [
            "what is the capital", "who is", "when was",
            "how old", "what year",
        ]
        if any(pattern in q for pattern in simple_patterns):
            return RetrievalStrategy.SKIP

        # Speculative: time-sensitive queries
        speculative_patterns = [
            "latest", "current", "today", "recent", "now",
        ]
        if any(pattern in q for pattern in speculative_patterns):
            return RetrievalStrategy.SPECULATIVE

        # Default: standard retrieval
        return RetrievalStrategy.STANDARD
