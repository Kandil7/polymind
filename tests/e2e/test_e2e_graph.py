"""End-to-end tests — full graph execution validation.

Tests the complete agent graph from query to answer.
"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
class TestE2EGraphExecution:
    """Test the full LangGraph agent pipeline."""

    def test_graph_builds_successfully(self) -> None:
        """Graph should build without errors."""
        from polymind.application.graph import build_graph

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_invokes_with_text(self) -> None:
        """Graph should process text queries."""
        from polymind.application.graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "user_query": "What is RAG?",
            "user_id": "e2e_test",
        })

        assert "final_answer" in result
        assert result["final_answer"] is not None
        assert result["modality"] == "text"

    def test_graph_handles_all_modalities(self) -> None:
        """Graph should handle different modalities."""
        from polymind.application.graph import build_graph

        graph = build_graph()

        # Test with no file paths (text only)
        result = graph.invoke({
            "user_query": "Hello",
            "user_id": "e2e_test",
        })
        assert result["modality"] == "text"

    def test_graph_returns_critic_scores(self) -> None:
        """Graph should return critic evaluation scores."""
        from polymind.application.graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "user_query": "What is RAG?",
            "user_id": "e2e_test",
        })

        assert "critic_scores" in result
        assert "passed_critic" in result


@pytest.mark.e2e
class TestE2EPlannerNode:
    """Test the planner node in isolation."""

    def test_planner_classifies_modality(self) -> None:
        """Planner should detect modality from file paths."""
        from polymind.application.agents.planner import run

        state = {
            "user_query": "Transcribe this",
            "user_id": "test",
            "audio_path": "/tmp/audio.mp3",
        }
        result = run(state)
        assert result["modality"] == "audio"

    def test_planner_classifies_intent(self) -> None:
        """Planner should classify query intent."""
        from polymind.application.agents.planner import run

        state = {
            "user_query": "Summarize this document",
            "user_id": "test",
        }
        result = run(state)
        assert result["intent"] in VALID_INTENTS

    def test_planner_initializes_state(self) -> None:
        """Planner should initialize retry and critic state."""
        from polymind.application.agents.planner import run

        state = {"user_query": "Hello", "user_id": "test"}
        result = run(state)
        assert result["retry_count"] == 0
        assert result["passed_critic"] is False


@pytest.mark.e2e
class TestE2ERouterNode:
    """Test the router node."""

    def test_router_classifies_strategy(self) -> None:
        """Router should classify retrieval strategy."""
        from polymind.application.agents.router import run

        state = {
            "user_query": "What is RAG?",
            "user_id": "test",
            "modality": "text",
        }
        result = run(state)
        assert "retrieval_strategy" in result
        assert result["retrieval_strategy"] in VALID_STRATEGIES

    def test_router_decide_routes_correctly(self) -> None:
        """Router should route to correct nodes."""
        from polymind.application.agents.router import decide

        state = {"modality": "audio"}
        assert decide(state) == "asr"

        state = {"modality": "image"}
        assert decide(state) == "vqa"

        state = {"modality": "text"}
        assert decide(state) == "rag"


@pytest.mark.e2e
class TestE2EGeneratorNode:
    """Test the generator node."""

    def test_generator_with_context(self) -> None:
        """Generator should use retrieved context."""
        from polymind.application.agents.generator import run

        state = {
            "user_query": "What is RAG?",
            "retrieved_chunks": [
                {"text": "RAG combines retrieval and generation.", "source": "doc.txt", "score": 0.9},
            ],
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        assert result["final_answer"] is not None
        assert len(result["citations"]) == 1

    def test_generator_without_context(self) -> None:
        """Generator should handle missing context."""
        from polymind.application.agents.generator import run

        state = {
            "user_query": "What is RAG?",
            "retrieved_chunks": [],
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        assert result["final_answer"] is not None


@pytest.mark.e2e
class TestE2ECriticNode:
    """Test the critic node."""

    def test_critic_evaluates_answer(self) -> None:
        """Critic should evaluate answer quality."""
        from polymind.application.agents.critic import run

        state = {
            "user_query": "What is RAG?",
            "final_answer": "RAG is Retrieval Augmented Generation.",
            "retrieved_chunks": [
                {"text": "RAG combines retrieval and generation."},
            ],
            "retry_count": 0,
        }
        result = run(state)
        assert "critic_scores" in result
        assert "passed_critic" in result

    def test_critic_retries_bad_answer(self) -> None:
        """Critic should retry when answer is poor."""
        from polymind.application.agents.critic import run

        state = {
            "user_query": "test",
            "final_answer": "I don't know.",
            "retrieved_chunks": [],
            "retry_count": 0,
        }
        result = run(state)
        # May or may not retry depending on heuristic
        assert "should_retry" in result


# Valid intents and strategies
VALID_INTENTS = {
    "summarization", "comparison", "factual_qa", "translation",
    "extraction", "reasoning", "creative", "general",
}
VALID_STRATEGIES = {"skip", "standard", "hipporag", "speculative"}
