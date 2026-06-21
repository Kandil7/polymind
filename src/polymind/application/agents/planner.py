"""Planner node — detects modality, intent, and recalls memory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()

# ── Intent labels ────────────────────────────────────────
VALID_INTENTS = frozenset({
    "summarization",
    "comparison",
    "factual_qa",
    "translation",
    "extraction",
    "reasoning",
    "creative",
    "general",
})


def run(state: PolyMindState) -> PolyMindState:
    """Classify modality and intent, recall memory context.

    Reads: user_query, audio_path, image_path, file_path, user_id
    Writes: modality, intent, retry_count, passed_critic, past_episodes, semantic_facts
    """
    from polymind.infrastructure.tracing import trace_span

    query = state.get("user_query", "")
    user_id = state.get("user_id", "default")

    with trace_span("planner", {"query.length": len(query)}) as span:
        # ── Detect modality from file extensions ──
        modality = "text"
        if state.get("audio_path"):
            modality = "audio"
        elif state.get("image_path"):
            modality = "image"
        elif fpath := state.get("file_path"):
            ext = Path(fpath).suffix.lower()
            modality = "table" if ext in (".csv", ".xlsx", ".xls") else "document"

        # ── Classify intent (LLM-first, keyword fallback) ──
        intent = _classify_intent(query)

        # ── Recall memory context ──
        past_episodes, semantic_facts = _recall_memory(query, user_id)

        if span:
            span.set_attribute("planner.modality", modality)
            span.set_attribute("planner.intent", intent)

        logger.info(
            "planner.done",
            modality=modality,
            intent=intent,
            episodes_recalled=len(past_episodes),
            facts_recalled=len(semantic_facts),
        )

    return {
        **state,
        "current_node": "planner",
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
    """Classify query intent using LLM with keyword fallback.

    Uses Groq's fast tier (Llama 3.1 8B) for intent classification.
    Falls back to keyword-based classification if LLM is unavailable.
    Caches results for identical queries.
    """
    # Check cache first
    from polymind.infrastructure.cache import cached_classification, store_classification

    cached = cached_classification(query)
    if cached is not None:
        return cached

    # Check if LLM is available via degradation manager
    from polymind.infrastructure.degradation import degradation

    if degradation.should_use_heuristic_classification():
        logger.debug("planner.intent.heuristic_fallback", reason="llm_unavailable")
        intent = _classify_intent_keywords(query)
        store_classification(query, intent)
        return intent

    # Try LLM classification first
    try:
        intent = _classify_intent_llm(query)
    except Exception as e:
        logger.debug("planner.intent.llm_failed", error=str(e))
        intent = _classify_intent_keywords(query)

    # Cache the result
    store_classification(query, intent)
    return intent


def _classify_intent_llm(query: str) -> str:
    """Classify intent using Groq LLM."""
    from langchain_core.messages import HumanMessage

    from polymind.infrastructure.llm.llm_factory import LLMFactory

    factory = LLMFactory()
    llm = factory.get_llm(tier="fast")

    prompt = f"""Classify this user query into exactly ONE intent category.

Categories:
- summarization: user wants a summary, TLDR, or brief overview
- comparison: user wants to compare two or more things
- factual_qa: user asks a specific factual question (what, who, when, where, how, why)
- translation: user wants text translated to another language
- extraction: user wants specific information extracted from text
- reasoning: user wants analysis, explanation, or logical reasoning
- creative: user wants creative content (writing, brainstorming, ideas)
- general: none of the above

Query: {query}

Reply with ONLY the category name, nothing else:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    # Validate and normalize
    intent = intent.strip().strip('"').strip("'").strip(".")
    if intent in VALID_INTENTS:
        return intent

    # Try partial matching
    for valid in VALID_INTENTS:
        if valid in intent:
            return valid

    return "general"


def _classify_intent_keywords(query: str) -> str:
    """Fallback keyword-based intent classification."""
    q = query.lower()

    summary_keywords = ("summarize", "summary", "tldr", "tl;dr", "brief")
    if any(w in q for w in summary_keywords):
        return "summarization"

    compare_keywords = ("compare", "difference", "vs", "versus", "contrast")
    if any(w in q for w in compare_keywords):
        return "comparison"

    translate_keywords = ("translate", "translation", "แปล", "ترجم")
    if any(w in q for w in translate_keywords):
        return "translation"

    extract_keywords = ("extract", "list", "find all", "pull out")
    if any(w in q for w in extract_keywords):
        return "extraction"

    reasoning_keywords = ("explain", "why", "analyze", "reason", "think")
    if any(w in q for w in reasoning_keywords):
        return "reasoning"

    creative_keywords = ("write", "create", "story", "poem", "brainstorm")
    if any(w in q for w in creative_keywords):
        return "creative"

    qa_keywords = ("what", "who", "when", "where", "how", "which")
    if any(q.startswith(w) or f" {w} " in q for w in qa_keywords):
        return "factual_qa"

    return "general"
