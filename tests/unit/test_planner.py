"""Tests for Planner node."""

from __future__ import annotations

from polymind.application.agents.planner import _classify_intent, run


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


class TestPlannerIntent:
    def test_summarization_intent(self) -> None:
        assert _classify_intent("Summarize this document") == "summarization"

    def test_comparison_intent(self) -> None:
        assert _classify_intent("Compare these two approaches") == "comparison"

    def test_factual_qa_intent(self) -> None:
        assert _classify_intent("What is RAG?") == "factual_qa"

    def test_translation_intent(self) -> None:
        assert _classify_intent("Translate this to Arabic") == "translation"

    def test_general_intent(self) -> None:
        assert _classify_intent("Tell me about something") == "general"

    def test_tldr_is_summarization(self) -> None:
        assert _classify_intent("tldr this article") == "summarization"


class TestPlannerState:
    def test_initializes_retry_count(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["retry_count"] == 0

    def test_initializes_passed_critic(self) -> None:
        state = {"user_query": "hello"}
        result = run(state)
        assert result["passed_critic"] is False
