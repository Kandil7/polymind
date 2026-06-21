"""Tests for Generator node."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from polymind.application.agents.generator import (
    _generate_fallback,
    run,
)
from polymind.application.state import PolyMindState


class TestGenerateFallback:
    def test_no_context_returns_helpful_message(self) -> None:
        result = _generate_fallback("question", "No context available.")
        assert "enough context" in result.lower()

    def test_with_context_returns_extractive(self) -> None:
        context = "RAG combines retrieval and generation for better answers."
        result = _generate_fallback("What is RAG?", context)
        assert "Based on the available context" in result
        assert "RAG" in result

    def test_truncates_long_context(self) -> None:
        long_context = "word " * 500
        result = _generate_fallback("test", long_context)
        assert len(result) < 2000  # Should be truncated


class TestGeneratorRun:
    def test_run_with_no_chunks(self) -> None:
        state: PolyMindState = {
            "user_query": "What is RAG?",
            "retrieved_chunks": [],
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        assert result["final_answer"] is not None
        assert len(result["citations"]) == 0

    @patch("polymind.application.agents.generator._generate_with_llm")
    def test_run_with_chunks(self, mock_llm) -> None:
        mock_llm.return_value = "RAG is Retrieval Augmented Generation."
        state: PolyMindState = {
            "user_query": "What is RAG?",
            "retrieved_chunks": [
                {
                    "text": "RAG combines retrieval and generation.",
                    "source": "doc.txt",
                    "score": 0.9,
                },
            ],
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        # LLM may or may not be called depending on degradation status
        # But the answer should be present
        assert result["final_answer"] is not None
        assert len(result["citations"]) == 1
        assert result["citations"][0]["source"] == "doc.txt"

    def test_run_includes_specialist_outputs(self) -> None:
        state: PolyMindState = {
            "user_query": "What is this?",
            "retrieved_chunks": [],
            "asr_transcript": "This is a test recording",
            "vqa_result": {"answer": "A photo of a cat"},
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        assert result["final_answer"] is not None

    @patch("polymind.application.agents.generator._generate_with_llm")
    def test_citations_filter_by_score(self, mock_llm) -> None:
        mock_llm.return_value = "Test answer."
        state: PolyMindState = {
            "user_query": "test",
            "retrieved_chunks": [
                {"text": "high", "source": "a.txt", "score": 0.9},
                {"text": "mid", "source": "b.txt", "score": 0.35},
                {"text": "low", "source": "c.txt", "score": 0.1},
            ],
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = run(state)
        # Only scores > 0.3 should be citations
        assert len(result["citations"]) == 2
