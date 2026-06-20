"""Tests for Planner node."""

from __future__ import annotations

from polymind.application.agents.planner import (
    VALID_INTENTS,
    _classify_intent,
    _classify_intent_keywords,
    run,
)


class TestPlannerModality:
    def test_text_modality(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["modality"] == "text"

    def test_audio_modality(self) -> None:
        state = {"user_query": "hello", "audio_path": "/tmp/audio.mp3"}
        result = run(state)
        assert result["modality"] == "audio"

    def test_image_modality(self) -> None:
        state = {"user_query": "hello", "image_path": "/tmp/img.jpg"}
        result = run(state)
        assert result["modality"] == "image"

    def test_document_modality(self) -> None:
        state = {"user_query": "hello", "file_path": "/tmp/doc.pdf"}
        result = run(state)
        assert result["modality"] == "document"

    def test_table_modality(self) -> None:
        state = {"user_query": "hello", "file_path": "/tmp/data.csv"}
        result = run(state)
        assert result["modality"] == "table"

    def test_xlsx_is_table(self) -> None:
        state = {"user_query": "hello", "file_path": "/tmp/data.xlsx"}
        result = run(state)
        assert result["modality"] == "table"


class TestPlannerIntentKeywords:
    """Test keyword-based intent classification (fallback path)."""

    def test_summarization_intent(self) -> None:
        assert _classify_intent_keywords("Summarize this document") == "summarization"

    def test_comparison_intent(self) -> None:
        assert _classify_intent_keywords("Compare these two approaches") == "comparison"

    def test_factual_qa_intent(self) -> None:
        assert _classify_intent_keywords("What is RAG?") == "factual_qa"

    def test_translation_intent(self) -> None:
        assert _classify_intent_keywords("Translate this to Arabic") == "translation"

    def test_general_intent(self) -> None:
        assert _classify_intent_keywords("Tell me about something") == "general"

    def test_tldr_is_summarization(self) -> None:
        assert _classify_intent_keywords("tldr this article") == "summarization"

    def test_extraction_intent(self) -> None:
        assert _classify_intent_keywords("Extract all names from this text") == "extraction"

    def test_reasoning_intent(self) -> None:
        assert _classify_intent_keywords("Explain why this happened") == "reasoning"

    def test_creative_intent(self) -> None:
        assert _classify_intent_keywords("Write a story about AI") == "creative"

    def test_valid_intents_defined(self) -> None:
        assert len(VALID_INTENTS) == 8
        assert "summarization" in VALID_INTENTS
        assert "general" in VALID_INTENTS


class TestPlannerIntent:
    """Test the main classify_intent function (LLM + keyword fallback)."""

    def test_returns_valid_intent(self) -> None:
        result = _classify_intent("What is RAG?")
        assert result in VALID_INTENTS

    def test_falls_back_to_keywords(self) -> None:
        # Without Groq API key, LLM fails and keywords are used
        result = _classify_intent("Summarize this document")
        assert result == "summarization"


class TestPlannerState:
    def test_initializes_retry_count(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["retry_count"] == 0

    def test_initializes_passed_critic(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["passed_critic"] is False

    def test_intent_is_valid(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["intent"] in VALID_INTENTS
