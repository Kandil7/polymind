"""Router node — dispatches to specialist agents based on modality."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()

# ── Strategy labels ──────────────────────────────────────
VALID_STRATEGIES = frozenset({
    "skip",
    "standard",
    "hipporag",
    "speculative",
})


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
    """Classify retrieval strategy using LLM with keyword fallback.

    Uses Groq's fast tier for strategy classification.
    Falls back to keyword patterns if LLM is unavailable.
    """
    try:
        return _classify_retrieval_llm(query)
    except Exception as e:
        logger.debug("router.strategy.llm_failed", error=str(e))
        return _classify_retrieval_keywords(query)


def _classify_retrieval_llm(query: str) -> str:
    """Classify retrieval strategy using Groq LLM."""
    from langchain_core.messages import HumanMessage

    from polymind.infrastructure.llm.llm_factory import LLMFactory

    factory = LLMFactory()
    llm = factory.get_llm(tier="fast")

    prompt = f"""Classify this query's retrieval strategy for a RAG system.

Strategies:
- skip: Simple factual question answerable from parametric knowledge (e.g., "What is the capital of France?")
- standard: Single-hop document lookup (most queries)
- hipporag: Multi-hop reasoning requiring connections across documents (e.g., "How does X relate to Y?", "Compare A and B")
- speculative: Time-sensitive query needing current/recent information (e.g., "latest news", "current status")

Query: {query}

Reply with ONLY the strategy name, nothing else:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    strategy = response.content.strip().lower()

    # Validate and normalize
    strategy = strategy.strip().strip('"').strip("'").strip(".")
    if strategy in VALID_STRATEGIES:
        return strategy

    # Try partial matching
    for valid in VALID_STRATEGIES:
        if valid in strategy:
            return valid

    return "standard"


def _classify_retrieval_keywords(query: str) -> str:
    """Fallback keyword-based retrieval strategy classification."""
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
