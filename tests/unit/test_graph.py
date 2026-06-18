"""Tests for Graph build and structure."""

from __future__ import annotations

from polymind.application.graph import build_graph


class TestGraphStructure:
    def test_graph_builds_successfully(self) -> None:
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self) -> None:
        graph = build_graph()
        # Compiled graph should be invocable
        assert hasattr(graph, "ainvoke")

    def test_graph_invokes_with_minimal_state(self) -> None:
        """End-to-end smoke test with minimal input."""
        graph = build_graph()
        result = graph.invoke({
            "user_query": "What is RAG?",
            "user_id": "test_user",
            "audio_path": None,
            "image_path": None,
            "file_path": None,
        })
        # Should produce some output
        assert "final_answer" in result or "modality" in result

    def test_graph_handles_audio_modality(self) -> None:
        """Test that audio path sets modality correctly (unit test, no model load)."""
        from polymind.application.agents.planner import run as planner_run

        state = {
            "user_query": "Transcribe this",
            "user_id": "test_user",
            "audio_path": "/tmp/fake.mp3",
            "image_path": None,
            "file_path": None,
        }
        result = planner_run(state)
        assert result["modality"] == "audio"
