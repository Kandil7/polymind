"""Tests for User Feedback Store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from polymind.infrastructure.feedback import FeedbackStore


class TestFeedbackStore:
    def test_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")
            assert store.count() == 0

    def test_record_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            store.record(
                query_id="q1",
                query="What is RAG?",
                rating=5,
                feedback="thumbs_up",
            )

            assert store.count() == 1
            entries = store.get_feedback(query_id="q1")
            assert len(entries) == 1
            assert entries[0]["rating"] == 5

    def test_rating_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            store.record(query_id="q1", query="test", rating=10)  # Should clamp to 5
            store.record(query_id="q2", query="test", rating=-5)  # Should clamp to 1

            entries = store.get_feedback()
            assert entries[0]["rating"] == 5
            assert entries[1]["rating"] == 1

    def test_get_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            # Add some feedback
            store.record(query_id="q1", query="test", rating=5, feedback="thumbs_up")
            store.record(query_id="q2", query="test", rating=4, feedback="thumbs_up")
            store.record(query_id="q3", query="test", rating=2, feedback="thumbs_down")

            stats = store.get_stats()
            assert stats["total"] == 3
            assert stats["average_rating"] == 3.67
            assert stats["thumbs_up"] == 2
            assert stats["thumbs_down"] == 1

    def test_satisfaction_by_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            store.record(
                query_id="q1", query="test", rating=5,
                metadata={"intent": "factual_qa"}
            )
            store.record(
                query_id="q2", query="test", rating=3,
                metadata={"intent": "summarization"}
            )

            by_intent = store.get_satisfaction_by_intent()
            assert by_intent["factual_qa"] == 5.0
            assert by_intent["summarization"] == 3.0

    def test_satisfaction_by_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")

            store.record(
                query_id="q1", query="test", rating=5,
                metadata={"strategy": "hipporag"}
            )
            store.record(
                query_id="q2", query="test", rating=3,
                metadata={"strategy": "standard"}
            )

            by_strategy = store.get_satisfaction_by_strategy()
            assert by_strategy["hipporag"] == 5.0
            assert by_strategy["standard"] == 3.0

    def test_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/feedback.json"

            # Store feedback
            store1 = FeedbackStore(store_path=path)
            store1.record(query_id="q1", query="test", rating=5)

            # Reload
            store2 = FeedbackStore(store_path=path)
            assert store2.count() == 1

    def test_empty_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(store_path=f"{tmpdir}/feedback.json")
            stats = store.get_stats()
            assert stats["total"] == 0
            assert stats["average_rating"] == 0.0
