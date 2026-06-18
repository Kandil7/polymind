"""Tests for 4-Layer Memory System."""

from __future__ import annotations

from pathlib import Path

from polymind.infrastructure.memory.episodic_store import EpisodicStore
from polymind.infrastructure.memory.procedural_store import ProceduralStore


class TestEpisodicStore:
    def test_init_without_mem0(self) -> None:
        store = EpisodicStore(user_id="test")
        # May or may not load depending on mem0 availability
        assert isinstance(store.is_available, bool)

    def test_store_does_not_raise(self) -> None:
        store = EpisodicStore.__new__(EpisodicStore)
        store._user_id = "test"
        store._memory = None
        # Should not raise, just skip
        store.store("query", "answer", faithfulness=0.9)

    def test_recall_returns_empty_when_no_memory(self) -> None:
        store = EpisodicStore.__new__(EpisodicStore)
        store._user_id = "test"
        store._memory = None
        result = store.recall("query")
        assert result == []


class TestProceduralStore:
    def test_store_and_recall(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")
        store = ProceduralStore(store_path=store_path)

        store.store("summarization", ["step1", "step2"], success_score=0.9)
        steps = store.recall("summarization")
        assert steps == ["step1", "step2"]

    def test_low_score_not_stored(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")
        store = ProceduralStore(store_path=store_path)

        store.store("test", ["step1"], success_score=0.3)
        steps = store.recall("test")
        assert steps is None

    def test_recall_returns_none_for_unknown(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")
        store = ProceduralStore(store_path=store_path)

        steps = store.recall("nonexistent")
        assert steps is None

    def test_count(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")
        store = ProceduralStore(store_path=store_path)

        assert store.count() == 0
        store.store("task1", ["s1"], 0.9)
        assert store.count() == 1

    def test_persistence(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")

        # Store
        store1 = ProceduralStore(store_path=store_path)
        store1.store("task", ["a", "b"], 0.9)

        # Reload
        store2 = ProceduralStore(store_path=store_path)
        steps = store2.recall("task")
        assert steps == ["a", "b"]

    def test_usage_count_increments(self, tmp_path: object) -> None:
        store_path = str(Path(str(tmp_path)) / "proc.json")
        store = ProceduralStore(store_path=store_path)

        store.store("task", ["s1"], 0.9)
        store.recall("task")
        store.recall("task")

        # Reload and check
        store2 = ProceduralStore(store_path=store_path)
        store2.recall("task")
        # Usage count should be tracked
        assert store2._store["task"]["used_count"] >= 2
