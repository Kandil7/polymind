"""Tests for 4-Layer Memory System — comprehensive coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polymind.infrastructure.memory.four_layer_memory import FourLayerMemory
from polymind.infrastructure.memory.procedural_store import ProceduralStore


class TestFourLayerMemoryInit:
    def test_init_with_defaults(self) -> None:
        memory = FourLayerMemory(user_id="test_user")
        assert memory._user_id == "test_user"

    def test_init_with_custom_qdrant_url(self) -> None:
        memory = FourLayerMemory(
            user_id="test", qdrant_url="http://custom:6333"
        )
        assert memory._qdrant_url == "http://custom:6333"


class TestEpisodicOperations:
    def test_store_episode(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._semantic = None
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        memory.store_episode("query", "answer", faithfulness=0.9, modality="text")
        memory._episodic.store.assert_called_once_with(
            "query", "answer", 0.9, "text"
        )

    def test_recall_episodes(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._episodic.recall.return_value = [
            {"query": "q1", "answer": "a1"}
        ]
        memory._semantic = None
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        result = memory.recall_episodes("query", top_k=3)
        assert len(result) == 1
        memory._episodic.recall.assert_called_once_with("query", 3)


class TestSemanticOperations:
    def test_recall_semantic_returns_empty_when_no_store(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._semantic = None
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        # Patch _ensure_semantic to keep _semantic as None
        with patch.object(memory, "_ensure_semantic"):
            result = memory.recall_semantic("query")
            assert result == []


class TestProceduralOperations:
    def test_store_and_recall_procedure(self, tmp_path: Path) -> None:
        proc_path = str(tmp_path / "proc.json")
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._semantic = None
        memory._procedural = ProceduralStore(store_path=proc_path)
        memory._qdrant_url = "http://localhost:6333"

        memory.store_procedure("summarization", ["step1", "step2"], 0.9)
        steps = memory.recall_procedure("summarization")
        assert steps == ["step1", "step2"]


class TestGetContext:
    def test_get_context_includes_episodes_and_facts(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._episodic.recall.return_value = [{"query": "q1"}]
        memory._semantic = MagicMock()
        memory._semantic.recall.return_value = ["fact1", "fact2"]
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        context = memory.get_context("test query")
        assert "episodes" in context
        assert "semantic_facts" in context
        assert len(context["episodes"]) == 1
        assert len(context["semantic_facts"]) == 2

    def test_get_context_handles_empty_memory(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._episodic.recall.return_value = []
        memory._semantic = MagicMock()
        memory._semantic.recall.return_value = []
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        context = memory.get_context("query")
        assert context["episodes"] == []
        assert context["semantic_facts"] == []


class TestConsolidation:
    def test_consolidate_skips_with_few_episodes(self) -> None:
        memory = FourLayerMemory.__new__(FourLayerMemory)
        memory._user_id = "test"
        memory._episodic = MagicMock()
        memory._episodic.recall.return_value = [{"q": "1"}]  # < 3 episodes
        memory._semantic = MagicMock()
        memory._procedural = MagicMock()
        memory._qdrant_url = "http://localhost:6333"

        # Should not call semantic.store
        memory.consolidate("query", "answer")
        memory._semantic.store.assert_not_called()
