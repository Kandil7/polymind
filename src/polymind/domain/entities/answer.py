"""Answer domain entity."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """A generated answer with confidence metadata."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}
