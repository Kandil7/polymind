"""Request/Response schemas for /feedback endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """User feedback on a query answer."""

    query_id: str = Field(..., description="Unique identifier for the query")
    query: str = Field(..., description="The user's original query")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5, 5=best)")
    feedback: str = Field(
        default="",
        description="Text feedback ('thumbs_up', 'thumbs_down', or comment)",
    )
    intent: str = Field(default="general", description="Query intent category")
    strategy: str = Field(default="standard", description="Retrieval strategy used")
    modality: str = Field(default="text", description="Input modality")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    status: str = Field(..., description="Submission status")
    stats: dict = Field(default_factory=dict, description="Updated feedback statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
