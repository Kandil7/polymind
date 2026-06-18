"""Synthesizer node — merges outputs and formats final response."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Synthesize the final answer with citations and confidence.

    Reads: final_answer, citations, critic_scores, retry_count
    Writes: final_answer (enhanced), citations (formatted)
    """
    answer = state.get("final_answer", "")
    citations = state.get("citations", [])
    scores = state.get("critic_scores", {})
    retry_count = state.get("retry_count", 0)

    # Calculate confidence from critic scores
    faithfulness = scores.get("faithfulness", 0.5) if isinstance(scores, dict) else 0.5
    if hasattr(faithfulness, "value"):
        faithfulness = faithfulness.value

    confidence = _calculate_confidence(faithfulness, retry_count)

    # Format answer with citations
    formatted_answer = _format_answer(answer, citations, confidence)

    logger.info(
        "synthesizer.done",
        confidence=f"{confidence:.2f}",
        citations=len(citations),
    )

    return {
        **state,
        "final_answer": formatted_answer,
        "citations": citations,
    }


def _calculate_confidence(faithfulness: float, retry_count: int) -> float:
    """Calculate overall confidence score."""
    base = faithfulness
    # Penalize for retries
    penalty = retry_count * 0.1
    return max(0.0, min(1.0, base - penalty))


def _format_answer(
    answer: str, citations: list[dict], confidence: float
) -> str:
    """Format the final answer with metadata."""
    parts = [answer]

    if citations:
        sources = []
        for c in citations[:3]:
            source = c.get("source", "unknown")
            score = c.get("score", 0.0)
            sources.append(f"  - {source} (relevance: {score:.2f})")
        parts.append("\n\n**Sources:**\n" + "\n".join(sources))

    parts.append(f"\n\n*Confidence: {confidence:.0%}*")

    return "".join(parts)
