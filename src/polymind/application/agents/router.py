"""Router node — dispatches to specialist agents based on modality."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Classify retrieval strategy based on query complexity.

    Reads: user_query, modality, past_episodes
    Writes: retrieval_strategy
    """
    query = state.get("user_query", "")
    strategy = _classify_retrieval(query)

    logger.info(
        "router.done",
        modality=state.get("modality"),
        strategy=strategy,
    )

    return {**state, "retrieval_strategy": strategy}


def decide(state: PolyMindState) -> str:
    """LangGraph routing function — returns next node name.

    Routes to the appropriate specialist based on modality.
    """
    modality = state.get("modality", "text")

    route_map = {
        "audio": "asr",
        "image": "vqa",
        "document": "docqa",
        "table": "tableqa",
        "text": "rag",
        "multi": "rag",  # Multi-modal: combine specialist outputs via RAG
    }

    next_node = route_map.get(modality, "rag")
    logger.info("router.decide", next_node=next_node)
    return next_node


def _classify_retrieval(query: str) -> str:
    """Classify retrieval strategy from query complexity."""
    q = query.lower()

    # Multi-hop signals → HippoRAG
    multi_hop = (
        "who founded", "which company", "what led to",
        "connection between", "relationship", "compare",
        "difference between", "how does", "why did",
    )
    if any(p in q for p in multi_hop):
        return "hipporag"

    # Simple factual → skip retrieval
    simple = (
        "what is the capital", "who is the president",
        "when was", "how old", "what year",
    )
    if any(p in q for p in simple):
        return "skip"

    # Time-sensitive → speculative
    speculative = ("latest", "current", "today", "recent", "now")
    if any(p in q for p in speculative):
        return "speculative"

    return "standard"
