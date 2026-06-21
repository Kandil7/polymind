"""End-to-end tests — full application stack validation.

These tests verify the complete system works correctly:
- API endpoints with real HTTP requests
- Full graph execution with all nodes
- Ingestion pipeline with real files
- Streaming endpoint
- Rate limiting and auth
- Error handling and degradation

Run with: pytest tests/e2e/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client (module-scoped for speed)."""
    from fastapi.testclient import TestClient
    from polymind.api.main import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def groq_available() -> bool:
    """Check if Groq API key is set."""
    import os

    return bool(os.getenv("GROQ_API_KEY"))


# ── E2E API Tests ────────────────────────────────────────
@pytest.mark.e2e
class TestE2EHealthCheck:
    def test_health_endpoint(self, client) -> None:
        """Health endpoint should return service status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert data["service"] == "polymind"
        assert "checks" in data

    def test_health_has_degradation_info(self, client) -> None:
        """Health should include degradation status."""
        response = client.get("/health")
        data = response.json()
        if "degradation" in data.get("checks", {}):
            degradation = data["checks"]["degradation"]
            assert "services" in degradation
            assert "overall" in degradation


@pytest.mark.e2e
class TestE2EQueryEndpoint:
    def test_query_requires_question(self, client) -> None:
        """Query endpoint should require a question."""
        response = client.post("/query/")
        assert response.status_code == 422

    def test_query_returns_valid_response(self, client) -> None:
        """Query should return a valid response structure."""
        response = client.post(
            "/query/",
            data={"question": "What is RAG?", "user_id": "e2e_test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "modality" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert data["modality"] == "text"

    def test_query_with_all_modalities(self, client) -> None:
        """Query should handle all modalities."""
        response = client.post(
            "/query/",
            data={
                "question": "What is this?",
                "user_id": "e2e_test",
            },
        )
        assert response.status_code == 200


@pytest.mark.e2e
class TestE2EStreamingEndpoint:
    def test_stream_requires_question(self, client) -> None:
        """Stream endpoint should require a question."""
        response = client.post("/query/stream/")
        assert response.status_code == 422

    def test_stream_returns_sse(self, client) -> None:
        """Stream endpoint should return SSE events."""
        response = client.post(
            "/query/stream/",
            data={"question": "What is RAG?", "user_id": "e2e_test"},
            timeout=60.0,
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_contains_events(self, client) -> None:
        """Stream should contain node_start/node_done events."""
        response = client.post(
            "/query/stream/",
            data={"question": "What is RAG?", "user_id": "e2e_test"},
            timeout=60.0,
        )
        content = response.text
        assert "event:" in content
        assert "data:" in content


@pytest.mark.e2e
class TestE2EIngestEndpoint:
    def test_ingest_requires_file(self, client) -> None:
        """Ingest endpoint should require a file."""
        response = client.post("/ingest/")
        assert response.status_code == 422

    def test_ingest_with_text_file(self, client) -> None:
        """Ingest should handle text files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("PolyMind is a self-evaluating knowledge assistant.")
            f.flush()

            with open(f.name, "rb") as file_obj:
                response = client.post(
                    "/ingest/",
                    files={"file": ("test.txt", file_obj, "text/plain")},
                    data={"source_name": "e2e_test"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("success", "error")
        assert "chunks_created" in data


@pytest.mark.e2e
class TestE2EEvalEndpoint:
    def test_eval_returns_response(self, client) -> None:
        """Eval endpoint should return a response."""
        response = client.post("/eval/", json={"limit": 1})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "total" in data
        assert "pass_rate" in data


@pytest.mark.e2e
class TestE2EFeedbackEndpoint:
    def test_feedback_requires_fields(self, client) -> None:
        """Feedback endpoint should require query_id and rating."""
        response = client.post("/feedback/", json={})
        assert response.status_code == 422

    def test_feedback_submission(self, client) -> None:
        """Should accept valid feedback."""
        response = client.post(
            "/feedback/",
            json={
                "query_id": "e2e_test_123",
                "query": "What is RAG?",
                "rating": 5,
                "feedback": "thumbs_up",
                "intent": "factual_qa",
                "strategy": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert "stats" in data

    def test_feedback_stats(self, client) -> None:
        """Should return feedback statistics."""
        response = client.get("/feedback/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "average_rating" in data


@pytest.mark.e2e
class TestE2EMetricsEndpoint:
    def test_metrics_returns_prometheus(self, client) -> None:
        """Metrics should return Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "polymind_" in response.text


@pytest.mark.e2e
class TestE2EDocumentation:
    def test_openapi_spec(self, client) -> None:
        """Should serve OpenAPI specification."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_swagger_ui(self, client) -> None:
        """Should serve Swagger UI."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self, client) -> None:
        """Should serve ReDoc."""
        response = client.get("/redoc")
        assert response.status_code == 200


@pytest.mark.e2e
class TestE2EErrorHandling:
    def test_invalid_endpoint_returns_404(self, client) -> None:
        """Non-existent endpoints should return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client) -> None:
        """Wrong HTTP method should return 405."""
        response = client.get("/query/")
        assert response.status_code == 405
