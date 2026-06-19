"""Tests for API endpoints — health, query, ingest, eval.

Heavy tests (invoking the full agent graph) are marked with `pytest.mark.heavy`
and skipped in environments where ML dependencies crash (e.g., pyarrow DLL issues).
"""

from __future__ import annotations

import pytest

from polymind.api.main import create_app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ── Marker for tests requiring full ML stack ─────────────
heavy = pytest.mark.heavy


class TestHealthEndpoint:
    """Tests for GET /health — lightweight, no ML models loaded."""

    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["service"] == "polymind"
        assert "version" in data

    def test_health_has_checks(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_health_includes_qdrant_check(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "qdrant" in data["checks"]

    def test_health_includes_llm_check(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "llm" in data["checks"]


class TestQueryEndpoint:
    """Tests for POST /query/ — validation and structure."""

    def test_query_requires_question(self, client) -> None:
        response = client.post("/query/")
        assert response.status_code == 422  # Validation error

    @heavy
    def test_query_with_text(self, client) -> None:
        response = client.post(
            "/query/",
            data={"question": "What is RAG?", "user_id": "test_user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "modality" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert data["modality"] == "text"

    @heavy
    def test_query_response_has_critic_scores(self, client) -> None:
        response = client.post(
            "/query/",
            data={"question": "Hello", "user_id": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "critic_scores" in data
        assert "retry_count" in data

    @heavy
    def test_query_with_user_id(self, client) -> None:
        response = client.post(
            "/query/",
            data={
                "question": "Test query",
                "user_id": "specific_user",
            },
        )
        assert response.status_code == 200


class TestIngestEndpoint:
    """Tests for POST /ingest/ — validation and structure."""

    def test_ingest_requires_file(self, client) -> None:
        response = client.post("/ingest/")
        assert response.status_code == 422

    @heavy
    def test_ingest_with_file(self, client, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for ingestion.")

        with open(test_file, "rb") as f:
            response = client.post(
                "/ingest/",
                files={"file": ("test.txt", f, "text/plain")},
                data={"source_name": "test_source"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("success", "error")
        assert "chunks_created" in data
        assert "processing_time_ms" in data


class TestEvalEndpoint:
    """Tests for POST /eval/ — structure."""

    @heavy
    def test_eval_returns_response(self, client) -> None:
        response = client.post("/eval/", json={"limit": 1})
        # May succeed or fail depending on Qdrant availability
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "total" in data
        assert "pass_rate" in data


class TestMetricsEndpoint:
    """Tests for GET /metrics — lightweight."""

    def test_metrics_returns_prometheus_format(self, client) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        content = response.text
        assert "polymind_" in content


class TestAppStructure:
    """Tests for the FastAPI application structure — lightweight."""

    def test_app_has_docs(self, client) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_app_has_redoc(self, client) -> None:
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_app_title(self, client) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "PolyMind"
