"""Procedural Memory Store — JSON-backed successful patterns.

Stores step-by-step procedures that worked for specific task types.
Used for procedural learning (which approach worked for this kind of query).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

DEFAULT_STORE_PATH = "data/procedural_memory.json"


class ProceduralStore:
    """JSON-backed procedural memory for successful patterns."""

    def __init__(self, store_path: str = DEFAULT_STORE_PATH) -> None:
        """Initialize procedural store.

        Args:
            store_path: Path to the JSON store file.
        """
        self._store_path = Path(store_path)
        self._store: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load procedures from disk."""
        if self._store_path.exists():
            try:
                self._store = json.loads(self._store_path.read_text(encoding="utf-8"))
                logger.info("procedural.loaded", count=len(self._store))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("procedural.load_failed", error=str(e))
                self._store = {}

    def _save(self) -> None:
        """Persist procedures to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps(self._store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def store(
        self,
        task_type: str,
        steps: list[str],
        success_score: float,
    ) -> None:
        """Store a successful procedure.

        Args:
            task_type: Category of task (e.g., "summarization", "factual_qa").
            steps: Steps that were taken.
            success_score: How successful the procedure was (0-1).
        """
        if success_score < 0.7:
            # Don't store low-quality procedures
            return

        self._store[task_type] = {
            "steps": steps,
            "score": success_score,
            "used_count": self._store.get(task_type, {}).get("used_count", 0),
            "last_used": datetime.now(UTC).isoformat(),
        }
        self._save()
        logger.info("procedural.stored", task_type=task_type, score=success_score)

    def recall(self, task_type: str) -> list[str] | None:
        """Recall a procedure for a task type.

        Args:
            task_type: Category of task.

        Returns:
            List of steps if found, None otherwise.
        """
        procedure = self._store.get(task_type)
        if procedure is None:
            return None

        # Increment usage count
        procedure["used_count"] = procedure.get("used_count", 0) + 1
        procedure["last_used"] = datetime.now(UTC).isoformat()
        self._save()

        return procedure.get("steps", [])

    def count(self) -> int:
        """Return the number of stored procedures."""
        return len(self._store)

    def list_task_types(self) -> list[str]:
        """Return all stored task types."""
        return list(self._store.keys())
