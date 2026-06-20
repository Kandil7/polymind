"""Query and QueryResult domain entities."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from polymind.domain.entities.answer import Answer
from polymind.domain.entities.chunk import DocumentChunk


class ScoreResult(BaseModel):
    """A single evaluation score from the Critic agent."""

    name: str
    value: float
    threshold: float
    passed: bool
    reason: str | None = None

    model_config = {"frozen": True}


class Query(BaseModel):
    """A user query entering the PolyMind system."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    user_id: str
    audio_path: str | None = None
    image_path: str | None = None
    file_path: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}


class QueryResult(BaseModel):
    """The final output returned to the user after all agents have processed."""

    query_id: UUID
    answer: Answer
    citations: list[DocumentChunk] = Field(default_factory=list)
    critic_scores: dict[str, ScoreResult] = Field(default_factory=dict)
    modality: str = "text"
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}
