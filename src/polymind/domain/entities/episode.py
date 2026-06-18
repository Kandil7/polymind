"""ConversationEpisode domain entity for memory systems."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ConversationEpisode(BaseModel):
    """A recorded conversation turn for episodic memory."""

    query: str
    answer: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    modality: str = "text"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}
