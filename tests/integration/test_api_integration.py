"""Integration tests — API endpoints with real services.

Tests require:
- Running Qdrant instance
- GROQ_API_KEY environment variable

Run with: pytest -m integration
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client (module-scoped)."""
    from fastapi.testclient import TestClient
    from polymind.api.main import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["service"] == "polymind"

    def test_health_has_all_checks(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "qdrant" in data["checks"]
        assert "llm" in data["checks"]
        assert "embedder" in data["checks"]


@pytest.mark.integration
class TestQueryEndpoint:
    def test_query_requires_question(self, client) -> None:
        response = client.post("/query/")
        assert response.status_code == 422

    def test_query_with_text(self, client) -> None:
        response = client.post(
            "/query/",
            data={"question": "What is RAG?", "user_id": "integration_test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "modality" in data
        assert "confidence" in data
        assert data["modality"] == "text"


@pytest.mark.integration
class TestIngestEndpoint:
    def test_ingest_requires_file(self, client) -> None:
        response = client.post("/ingest/")
        assert response.status_code == 422

    def test_ingest_with_file(self, client, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for ingestion.")

        with open(test_file, "rb") as f:
            response = client.post(
                "/ingest/",
                files={"file": ("test.txt", f, "text/plain")},
                data={"source_name": "integration_test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("success", "error")
        assert "chunks_created" in data


@pytest.mark.integration
class TestEvalEndpoint:
    def test_eval_endpoint_exists(self, client) -> None:
        response = client.post("/eval/", json={"limit": 1})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.integration
class TestMetricsEndpoint:
    def test_metrics_returns_prometheus(self, client) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "polymind_" in response.text


@pytest.mark.integration
class TestStreamingEndpoint:
    def test_stream_requires_question(self, client) -> None:
        response = client.post("/query/stream/")
        assert response.status_code == 422

    def test_stream_openapi_registered(self, client) -> None:
        response = client.get("/openapi.json")
        data = response.json()
        assert "/query/stream" in data.get("paths", {})
