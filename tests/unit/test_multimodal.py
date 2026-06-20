"""Tests for Multi-Modal Retrieval with CLIP."""

from __future__ import annotations

import pytest

from polymind.infrastructure.rag.clip_embedder import CLIPEmbedder, DEFAULT_DIMENSION
from polymind.infrastructure.rag.multimodal_retriever import MultiModalRetriever
from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk


class TestCLIPEmbedderStructure:
    def test_default_model(self) -> None:
        from polymind.infrastructure.rag.clip_embedder import DEFAULT_MODEL
        assert DEFAULT_MODEL == "openai/clip-vit-base-patch32"

    def test_default_dimension(self) -> None:
        assert DEFAULT_DIMENSION == 512

    def test_is_available_when_not_loaded(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        clip._model = None
        assert clip.is_available is False

    def test_is_available_when_loaded(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        clip._model = "mock_model"
        assert clip.is_available is True

    def test_dimension_without_model(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        clip._model = None
        assert clip.dimension == DEFAULT_DIMENSION

    def test_embed_text_raises_without_model(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        clip._model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            clip.embed_text("test")

    def test_embed_image_raises_without_model(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        clip._model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            clip.embed_image("test.jpg")


class TestCLIPEmbedderWithMock:
    def test_compute_similarity(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        sim = clip.compute_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 1e-6

    def test_compute_similarity_orthogonal(self) -> None:
        clip = CLIPEmbedder.__new__(CLIPEmbedder)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = clip.compute_similarity(vec1, vec2)
        assert abs(sim) < 1e-6


class TestMultiModalRetriever:
    def test_init(self) -> None:
        retriever = MultiModalRetriever()
        assert retriever.count() == 0

    def test_index_document(self) -> None:
        retriever = MultiModalRetriever.__new__(MultiModalRetriever)
        retriever._clip = None
        retriever._documents = {}
        retriever._embeddings = {}

        # Mock CLIP
        class MockCLIP:
            is_available = True
            def embed_text(self, text):
                return [0.1] * 512

        retriever._clip = MockCLIP()

        import asyncio
        chunk = DocumentChunk(
            text="Test document",
            metadata=ChunkMetadata(source="test", file_type="txt"),
        )
        asyncio.run(retriever.index([chunk]))

        assert retriever.count() == 1

    def test_empty_retrieve(self) -> None:
        retriever = MultiModalRetriever.__new__(MultiModalRetriever)
        retriever._documents = {}
        retriever._clip = None

        import asyncio
        results = asyncio.run(retriever.retrieve("test"))
        assert results == []

    def test_search_images_empty(self) -> None:
        retriever = MultiModalRetriever.__new__(MultiModalRetriever)
        retriever._documents = {}
        retriever._clip = None

        results = retriever.search_images("test")
        assert results == []
