"""Tests for Adaptive Retriever."""

from __future__ import annotations

import pytest

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.domain.interfaces.retriever import IRetriever
from polymind.infrastructure.qdrant.adaptive_retriever import (
    AdaptiveRetriever,
    RetrievalStrategy,
)


class TestAdaptiveRetrieverStructure:
    def test_implements_retriever_interface(self) -> None:
        assert issubclass(AdaptiveRetriever, IRetriever)

    def test_name_property(self) -> None:
        retriever = AdaptiveRetriever(retrievers={})
        assert retriever.name == "adaptive"


class TestQueryClassification:
    def _make_retriever(self) -> AdaptiveRetriever:
        return AdaptiveRetriever(retrievers={})

    def test_multi_hop_returns_hipporag(self) -> None:
        r = self._make_retriever()
        assert r._classify_query("What is the connection between Qdrant and RAG?") == RetrievalStrategy.HIPPORAG

    def test_simple_returns_skip(self) -> None:
        r = self._make_retriever()
        assert r._classify_query("What is the capital of France?") == RetrievalStrategy.SKIP

    def test_time_sensitive_returns_speculative(self) -> None:
        r = self._make_retriever()
        assert r._classify_query("What is the latest news about AI?") == RetrievalStrategy.SPECULATIVE

    def test_default_returns_standard(self) -> None:
        r = self._make_retriever()
        assert r._classify_query("Tell me about document processing") == RetrievalStrategy.STANDARD


class TestAdaptiveRetrieval:
    @pytest.mark.asyncio
    async def test_skip_returns_empty(self) -> None:
        r = AdaptiveRetriever(retrievers={})
        result = await r.retrieve("What is the capital of France?")
        assert result == []

    @pytest.mark.asyncio
    async def test_fallback_to_standard(self) -> None:
        class MockRetriever(IRetriever):
            async def retrieve(self, query: str, top_k: int = 5, **kwargs: object) -> list[DocumentChunk]:
                return [DocumentChunk(
                    text="mock",
                    metadata=ChunkMetadata(source="mock", file_type="text"),
                    score=1.0,
                )]
            async def index(self, chunks: list[DocumentChunk]) -> None:
                pass

        mock = MockRetriever()
        r = AdaptiveRetriever(retrievers={"standard": mock})
        result = await r.retrieve("Tell me about something")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_index_delegates_to_standard(self) -> None:
        class MockRetriever(IRetriever):
            def __init__(self) -> None:
                self.indexed: list[DocumentChunk] = []

            async def retrieve(self, query: str, top_k: int = 5, **kwargs: object) -> list[DocumentChunk]:
                return []

            async def index(self, chunks: list[DocumentChunk]) -> None:
                self.indexed.extend(chunks)

        mock = MockRetriever()
        r = AdaptiveRetriever(retrievers={"standard": mock})
        chunk = DocumentChunk(
            text="test",
            metadata=ChunkMetadata(source="test", file_type="text"),
        )
        await r.index([chunk])
        assert len(mock.indexed) == 1
