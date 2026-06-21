"""Tests for Memory Consolidation Pipeline."""

from __future__ import annotations

import pytest

from polymind.infrastructure.memory.consolidation import (
    MIN_EPISODES_FOR_CONSOLIDATION,
    ConsolidationPipeline,
)


class TestConsolidationPipeline:
    def test_init(self) -> None:
        pipeline = ConsolidationPipeline(user_id="test_user")
        assert pipeline._user_id == "test_user"
        assert pipeline._episode_buffer == []

    def test_buffer_accumulates(self) -> None:
        pipeline = ConsolidationPipeline(user_id="test")

        # Add episodes to buffer
        for i in range(3):
            pipeline._episode_buffer.append({
                "query": f"Query {i}",
                "answer": f"Answer {i}",
                "scores": {"faithfulness": 0.9},
                "modality": "text",
                "timestamp": "2024-01-01T00:00:00Z",
            })

        assert len(pipeline._episode_buffer) == 3

    def test_classify_task_type(self) -> None:
        pipeline = ConsolidationPipeline(user_id="test")

        assert pipeline._classify_task_type("Summarize this document") == "summarization"
        assert pipeline._classify_task_type("Compare RAG and fine-tuning") == "comparison"
        assert pipeline._classify_task_type("What is RAG?") == "factual_qa"
        assert pipeline._classify_task_type("Translate this to Arabic") == "translation"
        assert pipeline._classify_task_type("Tell me something") == "general"

    def test_get_stats(self) -> None:
        pipeline = ConsolidationPipeline(user_id="test")
        stats = pipeline.get_stats()
        assert "buffer_size" in stats
        assert "last_consolidation" in stats
        assert stats["buffer_size"] == 0


class TestConsolidationConfig:
    def test_min_episodes(self) -> None:
        assert MIN_EPISODES_FOR_CONSOLIDATION == 3


class TestConsolidationIntegration:
    @pytest.mark.asyncio
    async def test_consolidate_deferred_when_buffer_small(self) -> None:
        """Should defer consolidation when buffer is too small."""
        pipeline = ConsolidationPipeline(user_id="test")

        result = await pipeline.consolidate(
            query="What is RAG?",
            answer="RAG is Retrieval Augmented Generation.",
            critic_scores={"faithfulness": 0.9},
        )

        # Should defer, not extract facts
        assert result["facts_extracted"] == 0
        assert len(pipeline._episode_buffer) == 1

    def test_procedure_extraction(self) -> None:
        """Should extract procedures from successful interactions."""
        pipeline = ConsolidationPipeline(user_id="test")

        # Add successful episodes
        for i in range(3):
            pipeline._episode_buffer.append({
                "query": f"Summarize document {i}",
                "answer": f"Summary {i}",
                "scores": {"faithfulness": 0.95},
                "modality": "text",
                "timestamp": "2024-01-01T00:00:00Z",
            })

        procedures = pipeline._extract_procedures()
        assert len(procedures) == 3
        assert all(p["task_type"] == "summarization" for p in procedures)

    def test_low_score_not_stored(self) -> None:
        """Should not store procedures with low success scores."""
        pipeline = ConsolidationPipeline(user_id="test")

        pipeline._episode_buffer.append({
            "query": "Bad query",
            "answer": "Bad answer",
            "scores": {"faithfulness": 0.3},  # Below threshold
            "modality": "text",
            "timestamp": "2024-01-01T00:00:00Z",
        })

        procedures = pipeline._extract_procedures()
        assert len(procedures) == 0
