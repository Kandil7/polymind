"""Episodic Memory Store — Mem0-backed conversation history.

Stores complete interaction episodes (query → answer → scores) for
semantic recall of past conversations.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()


class EpisodicStore:
    """Mem0-backed episodic memory for conversation history.

    Stores episodes with metadata (timestamp, faithfulness, modality).
    Recalls by semantic similarity.
    """

    def __init__(self, user_id: str = "default") -> None:
        """Initialize episodic store.

        Args:
            user_id: User identifier for memory isolation.
        """
        self._user_id = user_id
        self._memory = None
        self._lazy_load()

    def _lazy_load(self) -> None:
        """Lazy-load Mem0 to avoid import-time overhead."""
        try:
            from mem0 import Memory

            self._memory = Memory()
            logger.info("episodic.loaded", user_id=self._user_id)
        except Exception as e:
            logger.warning("episodic.load_failed", error=str(e))
            self._memory = None

    def store(
        self,
        query: str,
        answer: str,
        faithfulness: float | None = None,
        modality: str = "text",
    ) -> None:
        """Store a conversation episode.

        Args:
            query: User's original query.
            answer: Generated answer.
            faithfulness: Critic faithfulness score.
            modality: Input modality.
        """
        if self._memory is None:
            logger.debug("episodic.store.skipped", reason="memory_not_loaded")
            return

        try:
            self._memory.add(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ],
                user_id=self._user_id,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "faithfulness": faithfulness,
                    "modality": modality,
                },
            )
            logger.info("episodic.stored", query_length=len(query))
        except Exception as e:
            logger.error("episodic.store.failed", error=str(e))

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Recall similar past episodes.

        Args:
            query: Search query.
            top_k: Number of episodes to recall.

        Returns:
            List of recalled episodes with metadata.
        """
        if self._memory is None:
            return []

        try:
            results = self._memory.search(
                query=query,
                user_id=self._user_id,
                limit=top_k,
            )
            episodes = results.get("results", [])
            logger.info("episodic.recalled", count=len(episodes))
            return episodes
        except Exception as e:
            logger.error("episodic.recall.failed", error=str(e))
            return []

    @property
    def is_available(self) -> bool:
        """Check if episodic store is loaded and available."""
        return self._memory is not None
