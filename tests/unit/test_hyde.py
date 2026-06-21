"""Tests for HyDE Query Expansion."""

from __future__ import annotations

from unittest.mock import patch

from polymind.infrastructure.hyde import (
    HYDE_PROMPT,
    expand_query_hyde,
    expand_query_multi,
)
from polymind.application.agents.rag_node import _should_use_hyde


class TestHyDEPrompt:
    def test_prompt_formatting(self) -> None:
        prompt = HYDE_PROMPT.format(query="What is RAG?")
        assert "What is RAG?" in prompt
        assert "Passage:" in prompt

    def test_prompt_requests_factual_content(self) -> None:
        prompt = HYDE_PROMPT.format(query="test")
        assert "factual" in prompt.lower()


class TestShouldUseHyDE:
    def test_short_query_no_hyde(self) -> None:
        assert _should_use_hyde("What is RAG?") is False

    def test_long_query_uses_hyde(self) -> None:
        query = "What is the difference between RAG and fine-tuning for language models?"
        assert _should_use_hyde(query) is True

    def test_comparison_query_uses_hyde(self) -> None:
        assert _should_use_hyde("Compare RAG and fine-tuning") is True

    def test_how_query_uses_hyde(self) -> None:
        assert _should_use_hyde("How does RAG work?") is True

    def test_why_query_uses_hyde(self) -> None:
        assert _should_use_hyde("Why is RAG important?") is True

    def test_simple_factual_no_hyde(self) -> None:
        assert _should_use_hyde("What is the capital of France?") is False


class TestHyDEExpansion:
    def test_expand_returns_string(self) -> None:
        result = expand_query_hyde("What is RAG?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_expand_multi_returns_list(self) -> None:
        result = expand_query_multi("What is RAG?", num_variants=2)
        assert isinstance(result, list)
        assert len(result) >= 1  # At least original query

    def test_expand_preserves_original_format(self) -> None:
        """Expand function should return a string."""
        result = expand_query_hyde("How does RAG work?")
        assert isinstance(result, str)
        # The function should return something (either expanded or original)
        assert len(result) > 0
