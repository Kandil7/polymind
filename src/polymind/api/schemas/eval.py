"""Request/Response schemas for /eval endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    """Evaluation request."""

    limit: int | None = Field(default=None, description="Max cases to evaluate")
    run_full: bool = Field(default=False, description="Run full benchmark")


class EvalResponse(BaseModel):
    """Evaluation response."""

    status: str = Field(..., description="Evaluation status")
    total: int = Field(default=0, description="Total cases evaluated")
    passed: int = Field(default=0, description="Cases that passed")
    failed: int = Field(default=0, description="Cases that failed")
    pass_rate: float = Field(default=0.0, description="Pass rate (0-1)")
    averages: dict = Field(default_factory=dict, description="Average metric scores")
    thresholds: dict = Field(default_factory=dict, description="Eval thresholds")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
