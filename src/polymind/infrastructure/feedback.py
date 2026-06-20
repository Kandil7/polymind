"""User Feedback Store — stores and analyzes user feedback on answers.

Captures user satisfaction signals to improve future responses:
- Thumbs up/down on answers
- Rating scores (1-5)
- Free-text comments
- Feedback patterns for learning

Usage:
    from polymind.infrastructure.feedback import FeedbackStore

    store = FeedbackStore()
    store.record(query_id="abc", query="What is RAG?", rating=5, feedback="thumbs_up")
    stats = store.get_stats()
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

DEFAULT_STORE_PATH = "data/feedback.json"
MAX_FEEDBACK_ENTRIES = 10000


class FeedbackStore:
    """Stores and analyzes user feedback on answers.

    Provides feedback recording, retrieval, and analysis for
    improving system performance over time.
    """

    def __init__(self, store_path: str = DEFAULT_STORE_PATH) -> None:
        """Initialize the feedback store.

        Args:
            store_path: Path to the JSON feedback file.
        """
        self._store_path = Path(store_path)
        self._feedback: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load feedback from disk."""
        if self._store_path.exists():
            try:
                self._feedback = json.loads(
                    self._store_path.read_text(encoding="utf-8")
                )
                logger.info("feedback.loaded", count=len(self._feedback))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("feedback.load_failed", error=str(e))
                self._feedback = []

    def _save(self) -> None:
        """Persist feedback to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        # Trim to max entries
        if len(self._feedback) > MAX_FEEDBACK_ENTRIES:
            self._feedback = self._feedback[-MAX_FEEDBACK_ENTRIES:]

        self._store_path.write_text(
            json.dumps(self._feedback, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record(
        self,
        query_id: str,
        query: str,
        rating: int,
        feedback: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Record user feedback on an answer.

        Args:
            query_id: Unique identifier for the query.
            query: The user's query text.
            rating: Rating (1-5, where 5 is best).
            feedback: Optional text feedback ("thumbs_up", "thumbs_down", or comment).
            metadata: Optional additional metadata (modality, intent, etc.).
        """
        entry = {
            "query_id": query_id,
            "query": query,
            "rating": max(1, min(5, rating)),  # Clamp to 1-5
            "feedback": feedback,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }

        self._feedback.append(entry)
        self._save()

        logger.info(
            "feedback.recorded",
            query_id=query_id,
            rating=rating,
            feedback=feedback,
        )

    def get_feedback(
        self,
        query_id: str | None = None,
        min_rating: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve feedback entries.

        Args:
            query_id: Filter by query ID.
            min_rating: Filter by minimum rating.
            limit: Maximum entries to return.

        Returns:
            List of feedback entries.
        """
        results = self._feedback

        if query_id:
            results = [f for f in results if f["query_id"] == query_id]

        if min_rating is not None:
            results = [f for f in results if f["rating"] >= min_rating]

        return results[-limit:]

    def get_stats(self) -> dict:
        """Get feedback statistics."""
        if not self._feedback:
            return {
                "total": 0,
                "average_rating": 0.0,
                "satisfaction_rate": 0.0,
                "thumbs_up": 0,
                "thumbs_down": 0,
            }

        ratings = [f["rating"] for f in self._feedback]
        avg_rating = sum(ratings) / len(ratings)

        # Count thumbs up/down
        thumbs_up = sum(
            1 for f in self._feedback
            if f.get("feedback") == "thumbs_up" or f["rating"] >= 4
        )
        thumbs_down = sum(
            1 for f in self._feedback
            if f.get("feedback") == "thumbs_down" or f["rating"] <= 2
        )

        total = len(self._feedback)
        satisfaction_rate = thumbs_up / max(total, 1)

        return {
            "total": total,
            "average_rating": round(avg_rating, 2),
            "satisfaction_rate": round(satisfaction_rate, 2),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
        }

    def get_satisfaction_by_intent(self) -> dict[str, float]:
        """Get satisfaction rates by intent type."""
        intent_ratings: dict[str, list[int]] = {}

        for f in self._feedback:
            intent = f.get("metadata", {}).get("intent", "unknown")
            if intent not in intent_ratings:
                intent_ratings[intent] = []
            intent_ratings[intent].append(f["rating"])

        return {
            intent: round(sum(ratings) / len(ratings), 2)
            for intent, ratings in intent_ratings.items()
            if ratings
        }

    def get_satisfaction_by_strategy(self) -> dict[str, float]:
        """Get satisfaction rates by retrieval strategy."""
        strategy_ratings: dict[str, list[int]] = {}

        for f in self._feedback:
            strategy = f.get("metadata", {}).get("strategy", "unknown")
            if strategy not in strategy_ratings:
                strategy_ratings[strategy] = []
            strategy_ratings[strategy].append(f["rating"])

        return {
            strategy: round(sum(ratings) / len(ratings), 2)
            for strategy, ratings in strategy_ratings.items()
            if ratings
        }

    def get_recent_feedback(self, hours: int = 24) -> list[dict]:
        """Get feedback from the last N hours."""
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()

        return [
            f for f in self._feedback
            if f.get("timestamp", "") >= cutoff_str
        ]

    def count(self) -> int:
        """Return total number of feedback entries."""
        return len(self._feedback)
