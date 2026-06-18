"""Tests for Router node."""

from __future__ import annotations

from polymind.application.agents.router import _classify_retrieval, decide, run


class TestRouterDecide:
    def test_audio_routes_to_asr(self) -> None:
        state = {"modality": "audio"}
        assert decide(state) == "asr"

    def test_image_routes_to_vqa(self) -> None:
        state = {"modality": "image"}
        assert decide(state) == "vqa"

    def test_document_routes_to_docqa(self) -> None:
        state = {"modality": "document"}
        assert decide(state) == "docqa"

    def test_table_routes_to_tableqa(self) -> None:
        state = {"modality": "table"}
        assert decide(state) == "tableqa"

    def test_text_routes_to_rag(self) -> None:
        state = {"modality": "text"}
        assert decide(state) == "rag"

    def test_unknown_defaults_to_rag(self) -> None:
        state = {"modality": "unknown"}
        assert decide(state) == "rag"


class TestRouterClassification:
    def test_multi_hop_returns_hipporag(self) -> None:
        assert _classify_retrieval("What is the connection between X and Y?") == "hipporag"

    def test_simple_returns_skip(self) -> None:
        assert _classify_retrieval("What is the capital of France?") == "skip"

    def test_time_sensitive_returns_speculative(self) -> None:
        assert _classify_retrieval("What is the latest news?") == "speculative"

    def test_default_returns_standard(self) -> None:
        assert _classify_retrieval("Tell me about document processing") == "standard"


class TestRouterRun:
    def test_sets_retrieval_strategy(self) -> None:
        state = {"user_query": "Tell me about RAG", "modality": "text"}
        result = run(state)
        assert "retrieval_strategy" in result
