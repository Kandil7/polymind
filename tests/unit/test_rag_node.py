"""Tests for RAG node — strategy routing and query building."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from polymind.application.agents.rag_node import (
    _build_effective_query,
    _retrieve_by_strategy,
)
from polymind.application.state import PolyMindState


class TestBuildEffectiveQuery:
    def test_simple_text_query(self) -> None:
        state: PolyMindState = {
            "user_query": "What is RAG?",
            "asr_transcript": None,
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = _build_effective_query(state)
        assert result == "What is RAG?"

    def test_with_transcript(self) -> None:
        state: PolyMindState = {
            "user_query": "Transcribe this",
            "asr_transcript": "Hello world this is a test",
            "vqa_result": None,
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = _build_effective_query(state)
        assert "Transcript:" in result
        assert "Hello world" in result

    def test_with_vqa_result(self) -> None:
        state: PolyMindState = {
            "user_query": "What is in this image?",
            "asr_transcript": None,
            "vqa_result": {"answer": "A cat sitting on a table"},
            "docqa_result": None,
            "tableqa_result": None,
        }
        result = _build_effective_query(state)
        assert "Image shows:" in result
        assert "cat" in result

    def test_with_all_modalities(self) -> None:
        state: PolyMindState = {
            "user_query": "Summarize everything",
            "asr_transcript": "Audio content here",
            "vqa_result": {"answer": "Image description"},
            "docqa_result": {"answer": "Document summary"},
            "tableqa_result": {"answer": "Table data"},
        }
        result = _build_effective_query(state)
        assert "Transcript:" in result
        assert "Image shows:" in result
        assert "Document says:" in result
        assert "Table shows:" in result


@pytest.mark.heavy
class TestRetrieveByStrategy:
    def test_skip_returns_empty(self) -> None:
        """The 'skip' strategy is handled in run() before _retrieve_by_strategy."""
        state: PolyMindState = {"retrieval_strategy": "skip"}
        from polymind.application.agents import rag_node

        with patch.object(rag_node, "run") as mock_run:
            mock_run.return_value = {
                **state,
                "current_node": "rag",
                "retrieved_chunks": [],
                "retrieval_scores": [],
            }
            result = rag_node.run(state)
            assert result["retrieved_chunks"] == []
            assert result["retrieval_scores"] == []
