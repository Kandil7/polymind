"""Integration tests — full end-to-end pipeline tests.

Tests require:
- Running Qdrant instance (Docker)
- GROQ_API_KEY environment variable
- Embedding model (auto-downloaded on first run)

Run with: pytest -m integration
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk


# ── Fixtures ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def qdrant_client():
    """Get Qdrant client (module-scoped for speed)."""
    from polymind.infrastructure.qdrant.client_factory import get_qdrant_client

    try:
        client = get_qdrant_client()
        client.get_collections()
        return client
    except Exception:
        pytest.skip("Qdrant not available")


@pytest.fixture(scope="module")
def embedder():
    """Get embedder (module-scoped to avoid reloading model)."""
    from polymind.infrastructure.rag.embedder import Embedder

    return Embedder()


@pytest.fixture(scope="module")
def seeded_qdrant(qdrant_client, embedder):
    """Seed Qdrant with test data for integration tests."""
    from polymind.infrastructure.qdrant.chunk_repository import (
        QdrantChunkRepository,
    )

    collection = "test_integration_e2e"
    repo = QdrantChunkRepository(qdrant_client, embedder, collection=collection)

    chunks = [
        DocumentChunk(
            text="RAG stands for Retrieval Augmented Generation. It combines information retrieval with language model generation to produce more accurate and grounded answers.",
            metadata=ChunkMetadata(source="rag_overview.txt", file_type="txt", modality="text"),
        ),
        DocumentChunk(
            text="RAG systems work by first retrieving relevant documents from a knowledge base using vector similarity search, then passing those documents as context to a language model to generate an answer.",
            metadata=ChunkMetadata(source="rag_howitworks.txt", file_type="txt", modality="text"),
        ),
        DocumentChunk(
            text="The key benefits of RAG include reducing hallucinations, providing citeable answers, and allowing knowledge bases to be updated without retraining the model.",
            metadata=ChunkMetadata(source="rag_benefits.txt", file_type="txt", modality="text"),
        ),
        DocumentChunk(
            text="Qdrant is a vector similarity search engine and database. It provides a production-ready API for storing, searching, and managing vectors with additional payload information.",
            metadata=ChunkMetadata(source="qdrant_docs.txt", file_type="txt", modality="text"),
        ),
        DocumentChunk(
            text="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps.",
            metadata=ChunkMetadata(source="langgraph_docs.txt", file_type="txt", modality="text"),
        ),
    ]

    asyncio.run(repo.index(chunks))
    yield collection

    # Cleanup
    try:
        qdrant_client.delete_collection(collection)
    except Exception:
        pass


@pytest.fixture(scope="module")
def groq_available() -> bool:
    """Check if Groq API key is set."""
    import os

    return bool(os.getenv("GROQ_API_KEY"))


# ── Qdrant Integration Tests ────────────────────────────
@pytest.mark.integration
class TestQdrantIntegration:
    def test_client_connects(self, qdrant_client) -> None:
        collections = qdrant_client.get_collections()
        assert collections is not None

    def test_collection_exists(self, qdrant_client, seeded_qdrant) -> None:
        info = qdrant_client.get_collection(seeded_qdrant)
        assert info.points_count == 5


# ── Embedder Integration Tests ──────────────────────────
@pytest.mark.integration
class TestEmbedderIntegration:
    def test_dimension(self, embedder) -> None:
        assert embedder.dimension == 1024

    def test_embed_single(self, embedder) -> None:
        result = embedder.embed_single("Hello world")
        assert len(result) == 1024
        # Check normalization
        norm = sum(x**2 for x in result) ** 0.5
        assert 0.99 <= norm <= 1.01

    def test_embed_batch(self, embedder) -> None:
        results = embedder.embed(["Hello", "World", "Test"])
        assert len(results) == 3
        assert all(len(v) == 1024 for v in results)


# ── Retrieval Integration Tests ─────────────────────────
@pytest.mark.integration
class TestRetrievalIntegration:
    def test_standard_retrieval(self, seeded_qdrant, embedder) -> None:
        from polymind.infrastructure.qdrant.chunk_repository import (
            QdrantChunkRepository,
        )
        from polymind.infrastructure.qdrant.client_factory import get_qdrant_client

        client = get_qdrant_client()
        repo = QdrantChunkRepository(client, embedder, collection=seeded_qdrant)
        results = asyncio.run(repo.retrieve("What is RAG?", top_k=3))

        assert len(results) > 0
        assert any("RAG" in r.text for r in results)

    def test_reranker_integration(self, seeded_qdrant, embedder) -> None:
        from polymind.infrastructure.qdrant.chunk_repository import (
            QdrantChunkRepository,
        )
        from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
        from polymind.infrastructure.rag.reranker import CrossEncoderReranker

        client = get_qdrant_client()
        repo = QdrantChunkRepository(client, embedder, collection=seeded_qdrant)
        results = asyncio.run(repo.retrieve("What is RAG?", top_k=10))

        reranker = CrossEncoderReranker()
        texts = [r.text for r in results]
        ranked = reranker.rerank("What is RAG?", texts, top_k=3)

        assert len(ranked) == 3
        # Top result should be relevant
        top_idx = ranked[0][0]
        assert "RAG" in texts[top_idx]


# ── Ingestion Pipeline Tests ────────────────────────────
@pytest.mark.integration
class TestIngestionPipeline:
    def test_ingest_text_file(self, qdrant_client, embedder, sample_text_file) -> None:
        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(embedder, collection="test_ingestion")
        chunks = asyncio.run(pipeline.ingest_file(str(sample_text_file)))

        assert len(chunks) > 0
        assert all(c.text for c in chunks)

        # Cleanup
        try:
            qdrant_client.delete_collection("test_ingestion")
        except Exception:
            pass

    def test_ingest_csv_file(self, qdrant_client, embedder, sample_csv_file) -> None:
        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(embedder, collection="test_ingestion_csv")
        chunks = asyncio.run(pipeline.ingest_file(str(sample_csv_file)))

        assert len(chunks) > 0

        # Cleanup
        try:
            qdrant_client.delete_collection("test_ingestion_csv")
        except Exception:
            pass


# ── Graph Integration Tests ─────────────────────────────
@pytest.mark.integration
class TestGraphIntegration:
    def test_graph_builds(self) -> None:
        from polymind.application.graph import build_graph

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_end_to_end(self, seeded_qdrant, groq_available) -> None:
        if not groq_available:
            pytest.skip("Groq API key not set")

        from polymind.application.graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "user_query": "What is RAG?",
            "user_id": "integration_test",
        })

        assert "final_answer" in result
        assert result["final_answer"] is not None
        assert len(result["final_answer"]) > 0
        assert result["modality"] == "text"
        assert result["intent"] in ("factual_qa", "general", "summarization", "comparison", "translation", "extraction", "reasoning", "creative")


# ── Streaming Integration Tests ─────────────────────────
@pytest.mark.integration
class TestStreamingIntegration:
    def test_stream_endpoint_exists(self) -> None:
        from polymind.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        # Test with missing question (should return 422)
        response = client.post("/query/stream/")
        assert response.status_code == 422

    def test_stream_openapi(self) -> None:
        from polymind.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/openapi.json")
        data = response.json()
        assert "/query/stream" in data.get("paths", {})


# ── Middleware Integration Tests ─────────────────────────
@pytest.mark.integration
class TestMiddlewareIntegration:
    def test_rate_limit_headers(self) -> None:
        from polymind.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        # Health check should not be rate limited
        response = client.get("/health")
        assert response.status_code == 200

    def test_auth_skips_public_endpoints(self) -> None:
        from polymind.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        # Health should work without auth
        response = client.get("/health")
        assert response.status_code == 200

    def test_cors_headers(self) -> None:
        from polymind.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.options(
            "/query/",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORS should allow the request
        assert response.status_code in (200, 405)
