"""Critic node — self-evaluates answer quality before delivery."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()

# ── Thresholds ──────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.72
RELEVANCY_THRESHOLD = 0.75
HALLUCINATION_THRESHOLD = 0.25
MAX_RETRIES = 2


def run(state: PolyMindState) -> PolyMindState:
    """Evaluate answer quality using LLM-as-Judge.

    Reads: user_query, final_answer, retrieved_chunks
    Writes: critic_scores, passed_critic, should_retry, retry_count
    """
    from polymind.infrastructure.tracing import trace_span

    query = state.get("user_query", "")
    answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)

    with trace_span("critic", {"retry_count": retry_count}) as span:
        contexts = [c.get("text", "") for c in chunks]

        try:
            scores = _evaluate_with_llm(query, answer, contexts)
        except Exception as e:
            logger.error("critic.llm_failed", error=str(e))
            scores = _evaluate_heuristic(query, answer, contexts)

        # Determine pass/fail
        faithfulness = scores.get("faithfulness", 0.0)
        relevancy = scores.get("answer_relevancy", 0.0)
        hallucination = scores.get("hallucination_rate", 1.0)

        passed = (
            faithfulness >= FAITHFULNESS_THRESHOLD
            and relevancy >= RELEVANCY_THRESHOLD
            and hallucination <= HALLUCINATION_THRESHOLD
        )

        should_retry = not passed and retry_count < MAX_RETRIES

        if span:
            span.set_attribute("critic.faithfulness", faithfulness)
            span.set_attribute("critic.relevancy", relevancy)
            span.set_attribute("critic.hallucination", hallucination)
            span.set_attribute("critic.passed", passed)

        logger.info(
            "critic.done",
            faithfulness=f"{faithfulness:.3f}",
            relevancy=f"{relevancy:.3f}",
            hallucination=f"{hallucination:.3f}",
            passed=passed,
            should_retry=should_retry,
        )

    return {
        **state,
        "critic_scores": scores,
        "passed_critic": passed,
        "should_retry": should_retry,
        "retry_count": retry_count + (1 if should_retry else 0),
    }


def decide(state: PolyMindState) -> str:
    """LangGraph routing function — retry or proceed to synthesizer."""
    if state.get("should_retry"):
        logger.info("critic.retry", retry_count=state.get("retry_count"))
        return "retry"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("critic.max_retries")
        return "fail_max"

    return "pass"


def _evaluate_with_llm(
    query: str, answer: str, contexts: list[str]
) -> dict[str, float]:
    """Evaluate using Groq LLM-as-Judge."""
    from langchain_core.messages import HumanMessage

    from polymind.infrastructure.llm.llm_factory import LLMFactory

    factory = LLMFactory()
    llm = factory.get_llm(tier="fast")

    context_text = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts[:5]))

    prompt = f"""You are a strict evaluator. Score the answer on 3 metrics.

## Question
{query}

## Answer
{answer}

## Retrieved Context
{context_text}

Score each metric from 0.0 to 1.0:
1. faithfulness: Is the answer grounded in the context? (1.0 = fully grounded, 0.0 = completely fabricated)
2. answer_relevancy: Does the answer address the question? (1.0 = directly answers, 0.0 = off-topic)
3. hallucination_rate: How much is fabricated? (0.0 = no hallucination, 1.0 = fully hallucinated)

Reply with ONLY a JSON object:
{{"faithfulness": 0.X, "answer_relevancy": 0.X, "hallucination_rate": 0.X}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_scores(response.content.strip())


def _evaluate_heuristic(
    query: str, answer: str, contexts: list[str]
) -> dict[str, float]:
    """Fallback heuristic evaluation when LLM is unavailable."""
    # Simple keyword overlap
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    context_words = set(" ".join(contexts).lower().split())

    if not answer_words:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "hallucination_rate": 1.0}

    # Relevancy = overlap between query and answer
    relevancy = len(query_words & answer_words) / max(len(query_words), 1)

    # Faithfulness = overlap between answer and context
    faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)

    # Hallucination = inverse of faithfulness
    hallucination = 1.0 - faithfulness

    return {
        "faithfulness": min(faithfulness, 1.0),
        "answer_relevancy": min(relevancy, 1.0),
        "hallucination_rate": max(hallucination, 0.0),
    }


def _parse_scores(text: str) -> dict[str, float]:
    """Parse LLM response into score dict."""
    import json
    import re

    # Extract JSON from response
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            return {
                "faithfulness": float(scores.get("faithfulness", 0.5)),
                "answer_relevancy": float(scores.get("answer_relevancy", 0.5)),
                "hallucination_rate": float(scores.get("hallucination_rate", 0.5)),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "hallucination_rate": 0.5}
