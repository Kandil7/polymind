"""Feedback endpoint — captures user satisfaction signals."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter

from polymind.api.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback on a query answer.

    Captures user satisfaction signals to improve future responses.
    """
    start_time = time.time()

    try:
        from polymind.infrastructure.feedback import FeedbackStore

        store = FeedbackStore()

        # Record feedback
        store.record(
            query_id=request.query_id,
            query=request.query,
            rating=request.rating,
            feedback=request.feedback,
            metadata={
                "intent": request.intent,
                "strategy": request.strategy,
                "modality": request.modality,
            },
        )

        # Get updated stats
        stats = store.get_stats()

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "feedback.submitted",
            query_id=request.query_id,
            rating=request.rating,
        )

        return FeedbackResponse(
            status="recorded",
            stats=stats,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("feedback.failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return FeedbackResponse(
            status="error",
            stats={},
            processing_time_ms=round(processing_time, 2),
        )


@router.get("/stats")
async def get_feedback_stats() -> dict:
    """Get feedback statistics."""
    try:
        from polymind.infrastructure.feedback import FeedbackStore

        store = FeedbackStore()
        stats = store.get_stats()
        stats["by_intent"] = store.get_satisfaction_by_intent()
        stats["by_strategy"] = store.get_satisfaction_by_strategy()
        return stats
    except Exception as e:
        logger.error("feedback.stats.failed", error=str(e))
        return {"error": str(e)}
