"""Integration tests — Qdrant, Embedder, and Ingestion pipeline.

These tests require a running Qdrant instance.
Mark with: pytest -m integration
"""

from __future__ import annotations

import pytest

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk


@pytest.fixture
def qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )

        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture
def embedder():
    """Create an embedder instance."""
    from polymind.infrastructure.rag.embedder import Embedder

    return Embedder()


class TestQdrantIntegration:
    """Tests requiring a running Qdrant instance."""

    @pytest.mark.integration
    def test_qdrant_client_connects(self, qdrant_available: bool) -> None:
        """Qdrant client should connect to localhost:6333."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )

        client = get_qdrant_client()
        collections = client.get_collections()
        assert collections is not None

    @pytest.mark.integration
    def test_collection_creation(self, qdrant_available: bool) -> None:
        """Should create a test collection if it doesn't exist."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )
        from polymind.infrastructure.rag.embedder import Embedder
        from polymind.infrastructure.qdrant.chunk_repository import (
            QdrantChunkRepository,
        )

        client = get_qdrant_client()
        embedder = Embedder()
        repo = QdrantChunkRepository(
            client, embedder, collection="test_integration"
        )
        assert repo is not None

        # Cleanup
        try:
            client.delete_collection("test_integration")
        except Exception:
            pass

    @pytest.mark.integration
    async def test_index_and_retrieve(
        self, qdrant_available: bool, embedder
    ) -> None:
        """Should index chunks and retrieve them."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from polymind.infrastructure.qdrant.client_factory import (
            get_qdrant_client,
        )
        from polymind.infrastructure.qdrant.chunk_repository import (
            QdrantChunkRepository,
        )

        collection = "test_integration_idx"
        client = get_qdrant_client()
        repo = QdrantChunkRepository(client, embedder, collection=collection)

        chunks = [
            DocumentChunk(
                text="Python is a programming language.",
                metadata=ChunkMetadata(
                    source="test", file_type="txt", modality="text"
                ),
            ),
            DocumentChunk(
                text="JavaScript is used for web development.",
                metadata=ChunkMetadata(
                    source="test", file_type="txt", modality="text"
                ),
            ),
        ]

        # Index
        await repo.index(chunks)

        # Retrieve
        results = await repo.retrieve("programming language", top_k=2)
        assert len(results) > 0
        assert results[0].text is not None

        # Cleanup
        try:
            client.delete_collection(collection)
        except Exception:
            pass


class TestEmbedderIntegration:
    """Tests requiring embedding model download."""

    @pytest.mark.integration
    def test_embedder_dimensions(self, embedder) -> None:
        """Embedder should return correct dimensions for bge-m3."""
        assert embedder.dimension == 1024

    @pytest.mark.integration
    def test_embed_single_text(self, embedder) -> None:
        """Should embed a single text into a vector."""
        result = embedder.embed_single("Hello world")
        assert isinstance(result, list)
        assert len(result) == 1024
        # Normalize check: L2 norm should be ~1.0
        norm = sum(x**2 for x in result) ** 0.5
        assert 0.99 <= norm <= 1.01

    @pytest.mark.integration
    def test_embed_multiple_texts(self, embedder) -> None:
        """Should embed multiple texts."""
        texts = ["Hello", "World", "Test"]
        results = embedder.embed(texts)
        assert len(results) == 3
        assert all(len(v) == 1024 for v in results)

    @pytest.mark.integration
    def test_similar_texts_have_higher_similarity(self, embedder) -> None:
        """Similar texts should have higher cosine similarity."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        emb1 = embedder.embed_single("What is RAG?")
        emb2 = embedder.embed_single("Explain Retrieval Augmented Generation")
        emb3 = embedder.embed_single("What is the capital of France?")

        sim_related = cosine_similarity([emb1], [emb2])[0][0]
        sim_unrelated = cosine_similarity([emb1], [emb3])[0][0]

        assert sim_related > sim_unrelated


class TestIngestionPipeline:
    """Tests for the full ingestion pipeline."""

    @pytest.mark.integration
    async def test_ingest_text_file(
        self, qdrant_available: bool, sample_text_file
    ) -> None:
        """Should ingest a text file into chunks."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from polymind.infrastructure.rag.embedder import Embedder
        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        embedder = Embedder()
        pipeline = IngestionPipeline(
            embedder, collection="test_ingestion"
        )

        chunks = await pipeline.ingest_file(str(sample_text_file))
        assert len(chunks) > 0
        assert all(c.text for c in chunks)
        assert all(c.metadata.source for c in chunks)

    @pytest.mark.integration
    async def test_ingest_csv_file(
        self, qdrant_available: bool, sample_csv_file
    ) -> None:
        """Should ingest a CSV file."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from polymind.infrastructure.rag.embedder import Embedder
        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        embedder = Embedder()
        pipeline = IngestionPipeline(
            embedder, collection="test_ingestion"
        )

        chunks = await pipeline.ingest_file(str(sample_csv_file))
        assert len(chunks) > 0

    @pytest.mark.integration
    def test_ingest_nonexistent_file(self, embedder) -> None:
        """Should raise FileNotFoundError for missing file."""
        import asyncio

        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(embedder)

        with pytest.raises(FileNotFoundError):
            asyncio.run(pipeline.ingest_file("/nonexistent/file.txt"))
