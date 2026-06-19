"""Planner node — detects modality, intent, and recalls memory."""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Classify modality and intent, recall memory context.

    Reads: user_query, audio_path, image_path, file_path, user_id
    Writes: modality, intent, retry_count, passed_critic, past_episodes, semantic_facts
    """
    query = state.get("user_query", "")
    user_id = state.get("user_id", "default")

    # ── Detect modality from file extensions ──
    modality = "text"
    if state.get("audio_path"):
        modality = "audio"
    elif state.get("image_path"):
        modality = "image"
    elif fpath := state.get("file_path"):
        ext = Path(fpath).suffix.lower()
        modality = "table" if ext in (".csv", ".xlsx", ".xls") else "document"

    # ── Classify intent ──
    intent = _classify_intent(query)

    # ── Recall memory context ──
    past_episodes, semantic_facts = _recall_memory(query, user_id)

    logger.info(
        "planner.done",
        modality=modality,
        intent=intent,
        episodes_recalled=len(past_episodes),
        facts_recalled=len(semantic_facts),
    )

    return {
        **state,
        "modality": modality,
        "intent": intent,
        "retry_count": 0,
        "passed_critic": False,
        "past_episodes": past_episodes,
        "semantic_facts": semantic_facts,
    }


def _recall_memory(query: str, user_id: str) -> tuple[list[dict], list[str]]:
    """Recall episodic and semantic memory."""
    try:
        from polymind.infrastructure.memory.four_layer_memory import (
            FourLayerMemory,
        )

        memory = FourLayerMemory(user_id=user_id)
        episodes = memory.recall_episodes(query, top_k=3)
        facts = memory.recall_semantic(query, top_k=5)
        return episodes, facts
    except Exception as e:
        logger.debug("planner.memoryrecall.failed", error=str(e))
        return [], []


def _classify_intent(query: str) -> str:
    """Classify query intent using keyword matching.

    In production, replace with LLM-based classification.
    """
    q = query.lower()

    summary_keywords = ("summarize", "summary", "tldr", "tl;dr", "brief")
    if any(w in q for w in summary_keywords):
        return "summarization"

    compare_keywords = ("compare", "difference", "vs", "versus", "contrast")
    if any(w in q for w in compare_keywords):
        return "comparison"

    qa_keywords = ("what", "who", "when", "where", "how", "why", "which")
    if any(q.startswith(w) or f" {w} " in q for w in qa_keywords):
        return "factual_qa"

    translate_keywords = ("translate", "translation", "แปล", "ترجم")
    if any(w in q for w in translate_keywords):
        return "translation"

    return "general"
