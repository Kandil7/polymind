"""Tests for Mixture-of-Agents."""

from __future__ import annotations

import pytest

from polymind.infrastructure.moa import AGENT_CONFIGS, get_agent_configs


class TestMoAConfigs:
    def test_default_configs_count(self) -> None:
        assert len(AGENT_CONFIGS) == 3

    def test_config_has_required_fields(self) -> None:
        for config in AGENT_CONFIGS:
            assert "name" in config
            assert "temperature" in config
            assert "system_prompt" in config

    def test_temperature_varies(self) -> None:
        temps = [c["temperature"] for c in AGENT_CONFIGS]
        # Should have different temperatures for diversity
        assert len(set(temps)) > 1

    def test_get_agent_configs(self) -> None:
        configs = get_agent_configs(2)
        assert len(configs) == 2

    def test_get_agent_configs_limit(self) -> None:
        configs = get_agent_configs(10)
        assert len(configs) == 3  # Capped at available configs


class TestMoAPrompts:
    def test_merger_prompt_formatting(self) -> None:
        from polymind.infrastructure.moa import MERGER_PROMPT

        formatted = MERGER_PROMPT.format(
            query="What is RAG?",
            candidates="### Candidate 1\nRAG is Retrieval Augmented Generation.",
            context="RAG combines retrieval and generation.",
        )
        assert "What is RAG?" in formatted
        assert "Candidate 1" in formatted


class TestMoAGenerator:
    def test_moa_generation_function_exists(self) -> None:
        from polymind.application.agents.generator import _generate_with_moa
        assert callable(_generate_with_moa)

    def test_generator_runs_with_moa_flag(self, sample_query_with_chunks) -> None:
        """Test that generator can be called with MoA flag."""
        from polymind.application.agents.generator import run

        state = {**sample_query_with_chunks, "use_moa": False}
        # This should work without actually calling MoA
        result = run(state)
        assert "final_answer" in result
        assert result["final_answer"] is not None
