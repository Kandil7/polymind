"""Health endpoint schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(default="0.1.0", description="PolyMind version")
    service: str = Field(default="polymind", description="Service name")
