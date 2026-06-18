"""Tests for LLM Factory."""

from __future__ import annotations

import pytest

from polymind.infrastructure.llm.llm_factory import (
    GROQ_BASE_URL,
    GROQ_MODELS,
    LLMFactory,
)


class TestLLMFactoryStructure:
    def test_groq_models_defined(self) -> None:
        assert "reasoning" in GROQ_MODELS
        assert "fast" in GROQ_MODELS
        assert "qwen" in GROQ_MODELS
        assert "code" in GROQ_MODELS

    def test_groq_base_url(self) -> None:
        assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"

    def test_factory_init_default_provider(self) -> None:
        factory = LLMFactory.__new__(LLMFactory)
        factory._provider = "groq"
        factory._api_key = "test"
        factory._clients = {}
        assert factory.provider == "groq"

    def test_available_tiers(self) -> None:
        factory = LLMFactory.__new__(LLMFactory)
        factory._provider = "groq"
        factory._api_key = "test"
        factory._clients = {}
        tiers = factory.available_tiers
        assert "reasoning" in tiers
        assert "fast" in tiers


class TestLLMFactoryModels:
    def test_reasoning_model_is_llama70b(self) -> None:
        assert GROQ_MODELS["reasoning"] == "llama-3.3-70b-versatile"

    def test_fast_model_is_llama8b(self) -> None:
        assert GROQ_MODELS["fast"] == "llama-3.1-8b-instant"

    def test_qwen_model(self) -> None:
        assert GROQ_MODELS["qwen"] == "qwen/qwen3-32b"


class TestGroqASR:
    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist
        from polymind.infrastructure.llm.groq_asr import GroqASRWrapper
        assert issubclass(GroqASRWrapper, ISpecialist)

    def test_name_property(self) -> None:
        from polymind.infrastructure.llm.groq_asr import GroqASRWrapper
        wrapper = GroqASRWrapper.__new__(GroqASRWrapper)
        wrapper._model_id = "test"
        wrapper._client = None
        assert wrapper.name == "groq_asr"

    @pytest.mark.asyncio
    async def test_process_raises_when_no_client(self) -> None:
        from polymind.infrastructure.llm.groq_asr import GroqASRWrapper
        wrapper = GroqASRWrapper.__new__(GroqASRWrapper)
        wrapper._model_id = "test"
        wrapper._client = None
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("dummy.wav")

    @pytest.mark.asyncio
    async def test_process_raises_when_file_not_found(self) -> None:
        from polymind.infrastructure.llm.groq_asr import GroqASRWrapper
        wrapper = GroqASRWrapper.__new__(GroqASRWrapper)
        wrapper._model_id = "test"
        wrapper._client = object()  # Fake client
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await wrapper.process("/nonexistent/audio.wav")
