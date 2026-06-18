"""Score value object — wraps a named metric with threshold logic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Score(BaseModel):
    """A named metric score with pass/fail threshold."""

    name: str
    value: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)

    @property
    def passed(self) -> bool:
        """Score passes if value meets or exceeds threshold."""
        return self.value >= self.threshold

    model_config = {"frozen": True}
