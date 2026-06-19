"""Request/Response schemas for /ingest endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Document ingestion request."""

    source_name: str | None = Field(default=None, description="Source name override")
    collection: str = Field(default="polymind", description="Target collection")


class IngestResponse(BaseModel):
    """Ingestion response."""

    status: str = Field(..., description="Ingestion status")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    source: str = Field(default="", description="Source file name")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
