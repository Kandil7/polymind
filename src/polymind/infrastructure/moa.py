"""Mixture-of-Agents — generates multiple answers and merges the best parts.

Uses diversity in generation to improve answer quality:
1. Multiple generator agents produce answers with different prompts/temperatures
2. A merger agent combines the best parts into a final answer

Based on: https://arxiv.org/abs/2401.04088

Usage:
    from polymind.infrastructure.moa import generate_with_moa

    answer = await generate_with_moa(query, context, num_agents=3)
"""

from __future__ import annotations

import json
import re

import structlog

logger = structlog.get_logger()

# ── Agent configurations ─────────────────────────────────
AGENT_CONFIGS = [
    {
        "name": "precise",
        "temperature": 0.0,
        "system_prompt": "You are a precise, factual assistant. Answer concisely based only on the provided context. Cite sources when possible.",
    },
    {
        "name": "comprehensive",
        "temperature": 0.3,
        "system_prompt": "You are a thorough assistant. Provide a comprehensive answer covering all relevant aspects from the context. Be detailed but clear.",
    },
    {
        "name": "creative",
        "temperature": 0.7,
        "system_prompt": "You are an insightful assistant. Explain the answer in an engaging way that helps the user understand. Use analogies or examples if helpful.",
    },
]

MERGER_PROMPT = """You are a quality-focused answer merger. You will receive multiple candidate answers to the same question.

Your task:
1. Identify the most accurate and complete information from all candidates
2. Merge the best parts into a single high-quality answer
3. Remove any contradictions or redundancies
4. Ensure the answer is grounded in the provided context

## Question
{query}

## Candidate Answers
{candidates}

## Context
{context}

## Instructions
- Combine the strengths of each candidate
- Prefer factual accuracy over length
- Ensure all claims are supported by the context
- Return ONLY the merged answer, no meta-commentary
"""


async def generate_with_moa(
    query: str,
    context: str,
    num_agents: int = 3,
    merger_model_tier: str = "reasoning",
) -> str:
    """Generate an answer using Mixture-of-Agents.

    Args:
        query: The user's question.
        context: Retrieved context for grounding.
        num_agents: Number of generator agents (1-5).
        merger_model_tier: LLM tier for the merger agent.

    Returns:
        Merged answer combining the best parts from all agents.
    """
    from polymind.infrastructure.llm.llm_factory import LLMFactory

    # Limit num_agents to available configs
    configs = AGENT_CONFIGS[:num_agents]

    logger.info("moa.start", agents=len(configs), query_length=len(query))

    factory = LLMFactory()

    # Generate answers from multiple agents
    candidates = []
    for config in configs:
        try:
            answer = await _generate_single_agent(
                query=query,
                context=context,
                config=config,
                factory=factory,
            )
            candidates.append({
                "name": config["name"],
                "answer": answer,
            })
            logger.debug("moa.agent.done", agent=config["name"], length=len(answer))
        except Exception as e:
            logger.warning("moa.agent.failed", agent=config["name"], error=str(e))

    if not candidates:
        # All agents failed — fall back to simple generation
        logger.warning("moa.all_agents_failed")
        return await _generate_single_agent(
            query=query,
            context=context,
            config=AGENT_CONFIGS[0],
            factory=factory,
        )

    if len(candidates) == 1:
        # Only one agent succeeded — return its answer
        return candidates[0]["answer"]

    # Merge the answers
    merged = await _merge_answers(
        query=query,
        context=context,
        candidates=candidates,
        factory=factory,
        model_tier=merger_model_tier,
    )

    logger.info("moa.done", candidates=len(candidates), merged_length=len(merged))
    return merged


async def _generate_single_agent(
    query: str,
    context: str,
    config: dict,
    factory: object,
) -> str:
    """Generate an answer using a single agent configuration.

    Args:
        query: The user's question.
        context: Retrieved context for grounding.
        config: Agent configuration dict with name, temperature, system_prompt.
        factory: LLMFactory instance for model access.

    Returns:
        Generated answer text from the agent.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = factory.get_llm(tier="fast")

    messages = [
        SystemMessage(content=config["system_prompt"]),
        HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
    ]

    # Create LLM with specific temperature
    from langchain_openai import ChatOpenAI
    from polymind.infrastructure.llm.llm_factory import GROQ_BASE_URL

    agent_llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=factory._api_key,
        base_url=GROQ_BASE_URL,
        temperature=config["temperature"],
        max_tokens=1024,
    )

    response = agent_llm.invoke(messages)
    return response.content.strip()


async def _merge_answers(
    query: str,
    context: str,
    candidates: list[dict],
    factory: object,
    model_tier: str = "reasoning",
) -> str:
    """Merge multiple candidate answers into one high-quality answer.

    Args:
        query: The user's question.
        context: Retrieved context for grounding.
        candidates: List of dicts with 'name' and 'answer' keys.
        factory: LLMFactory instance for model access.
        model_tier: LLM tier for the merger agent.

    Returns:
        Merged answer text combining the best parts.
    """
    from langchain_core.messages import HumanMessage

    # Format candidates
    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        candidates_text += f"\n### Candidate {i} ({c['name']})\n{c['answer']}\n"

    prompt = MERGER_PROMPT.format(
        query=query,
        candidates=candidates_text,
        context=context[:2000],  # Truncate for context window
    )

    llm = factory.get_llm(tier=model_tier)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def get_agent_configs(num_agents: int = 3) -> list[dict]:
    """Get agent configurations for MoA.

    Args:
        num_agents: Number of agent configs to return (max 3).

    Returns:
        List of agent configuration dicts.
    """
    return AGENT_CONFIGS[:num_agents]
