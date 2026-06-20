"""HyDE — Hypothetical Document Embeddings for improved retrieval.

Generates a hypothetical answer to the query, then uses its embedding
for retrieval instead of the original query embedding.

Why it works:
- Hypothetical answers are semantically closer to relevant documents
- Bridges the "vocabulary gap" between queries and documents
- Especially effective for complex or ambiguous queries

Based on: https://arxiv.org/abs/2212.10496

Usage:
    from polymind.infrastructure.hyde import expand_query_hyde

    expanded = expand_query_hyde("What is RAG?")
    # Returns: "RAG is a technique that combines retrieval and generation..."
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# ── HyDE Prompt ──────────────────────────────────────────
HYDE_PROMPT = """Write a short, factual passage that would answer this question.
The passage should be written as if it were an excerpt from a knowledge base.
Be concise (2-4 sentences). Focus on factual information, not opinions.

Question: {query}

Passage:"""


def expand_query_hyde(query: str) -> str:
    """Generate a hypothetical answer for HyDE retrieval.

    Uses Groq's fast tier to generate a brief hypothetical passage
    that would answer the query. This passage is then embedded
    and used for retrieval instead of the original query.

    Args:
        query: The user's search query.

    Returns:
        Hypothetical passage (or original query on failure).
    """
    try:
        from langchain_core.messages import HumanMessage

        from polymind.infrastructure.llm.llm_factory import LLMFactory

        factory = LLMFactory()
        llm = factory.get_llm(tier="fast")

        prompt = HYDE_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        hypothetical = response.content.strip()

        if hypothetical and len(hypothetical) > 20:
            logger.debug(
                "hyde.expanded",
                original_length=len(query),
                expanded_length=len(hypothetical),
            )
            return hypothetical
        else:
            logger.debug("hyde.fallback", reason="response_too_short")
            return query

    except Exception as e:
        logger.warning("hyde.failed", error=str(e))
        return query


def expand_query_multi(query: str, num_variants: int = 3) -> list[str]:
    """Generate multiple hypothetical answers for diversified retrieval.

    Args:
        query: The user's search query.
        num_variants: Number of hypothetical passages to generate.

    Returns:
        List of hypothetical passages (may be shorter if some fail).
    """
    variants = [query]  # Always include original query

    try:
        from langchain_core.messages import HumanMessage

        from polymind.infrastructure.llm.llm_factory import LLMFactory

        factory = LLMFactory()
        llm = factory.get_llm(tier="fast")

        for i in range(num_variants):
            try:
                prompt = f"""Write a short, factual passage that answers this question.
Use a different perspective or emphasis than a typical answer.
Passage {i + 1} of {num_variants}:

Question: {query}

Passage:"""

                response = llm.invoke([HumanMessage(content=prompt)])
                variant = response.content.strip()

                if variant and len(variant) > 20:
                    variants.append(variant)

            except Exception as e:
                logger.debug("hyde.variant.failed", variant=i, error=str(e))

        logger.info("hyde.multi_done", variants=len(variants))
        return variants

    except Exception as e:
        logger.warning("hyde.multi.failed", error=str(e))
        return variants
