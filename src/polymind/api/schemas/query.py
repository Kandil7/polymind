"""Request/Response schemas for /query endpoint.

QueryRequest is used for programmatic API clients (non-multipart).
QueryResponse is used by the /query route for all responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Multimodal query request (JSON body for programmatic clients).

    Note: The /query route uses Form/File parameters for multipart uploads.
    This schema is provided for API documentation and non-file queries.
    """

    question: str = Field(..., description="User's question or instruction")
    user_id: str = Field(default="anonymous", description="User identifier")
    audio_path: str | None = Field(default=None, description="Path to audio file")
    image_path: str | None = Field(default=None, description="Path to image file")
    file_path: str | None = Field(default=None, description="Path to document file")


class QueryResponse(BaseModel):
    """Query response with answer, citations, and critic scores."""

    answer: str = Field(..., description="Generated answer")
    modality: str = Field(..., description="Detected input modality")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence")
    citations: list[dict] = Field(default_factory=list, description="Source citations")
    critic_scores: dict = Field(default_factory=dict, description="Critic evaluation scores")
    retry_count: int = Field(default=0, description="Number of retries performed")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
