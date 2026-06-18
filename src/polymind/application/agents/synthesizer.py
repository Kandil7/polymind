"""Synthesizer node — merges outputs and formats final response."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Synthesize the final answer with citations and confidence.

    Reads: final_answer, citations, critic_scores, retry_count, user_query, user_id
    Writes: final_answer (enhanced), citations (formatted)
    Side effects: Stores episode in memory, consolidates patterns.
    """
    answer = state.get("final_answer", "")
    citations = state.get("citations", [])
    scores = state.get("critic_scores", {})
    retry_count = state.get("retry_count", 0)
    query = state.get("user_query", "")
    user_id = state.get("user_id", "default")
    modality = state.get("modality", "text")

    # Calculate confidence from critic scores
    faithfulness = scores.get("faithfulness", 0.5) if isinstance(scores, dict) else 0.5
    if hasattr(faithfulness, "value"):
        faithfulness = faithfulness.value

    confidence = _calculate_confidence(faithfulness, retry_count)

    # Format answer with citations
    formatted_answer = _format_answer(answer, citations, confidence)

    # Store episode in memory
    _store_episode(query, answer, faithfulness, modality, user_id)

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


def _store_episode(
    query: str,
    answer: str,
    faithfulness: float,
    modality: str,
    user_id: str,
) -> None:
    """Store the interaction episode in memory."""
    try:
        from polymind.infrastructure.memory.four_layer_memory import (
            FourLayerMemory,
        )

        memory = FourLayerMemory(user_id=user_id)
        memory.store_episode(
            query=query,
            answer=answer,
            faithfulness=faithfulness,
            modality=modality,
        )
    except Exception as e:
        logger.debug("synthesizer.episode_store.failed", error=str(e))


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
