"""Generator node — synthesizes answer using LLM with retrieved context."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Generate an answer grounded in retrieved context.

    Reads: user_query, retrieved_chunks, asr_transcript, vqa_result, docqa_result
    Writes: draft_answers, final_answer, citations
    """
    from polymind.infrastructure.tracing import trace_span

    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])

    with trace_span("generator", {"chunks": len(chunks)}) as span:
        # Build context from retrieved chunks
        context_parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            context_parts.append(f"[Source: {source}] {text}")

        context = "\n\n".join(context_parts) if context_parts else "No context available."

        # Add specialist outputs to context
        if transcript := state.get("asr_transcript"):
            context += f"\n\n[Audio Transcript]\n{transcript[:1000]}"

        if vqa := state.get("vqa_result"):
            if answer := vqa.get("answer"):
                context += f"\n\n[Image Analysis]\n{answer}"

        if docqa := state.get("docqa_result"):
            if answer := docqa.get("answer"):
                context += f"\n\n[Document Analysis]\n{answer}"

        if tableqa := state.get("tableqa_result"):
            if answer := tableqa.get("answer"):
                context += f"\n\n[Table Analysis]\n{answer}"

        # Generate answer using LLM or MoA
        # Check if LLM is available via degradation manager
        from polymind.infrastructure.degradation import degradation

        if degradation.should_use_heuristic_generation():
            logger.warning("generator.degraded", reason="llm_unavailable")
            answer = _generate_fallback(query, context)
        else:
            try:
                # Check if MoA mode is enabled
                use_moa = state.get("use_moa", False)

                if use_moa and len(chunks) > 0:
                    # Use Mixture-of-Agents for higher quality
                    answer = _generate_with_moa(query, context)
                else:
                    answer = _generate_with_llm(query, context)

                degradation.record_service_success("llm")
            except Exception as e:
                degradation.record_service_failure("llm")
                logger.error("generator.llm_failed", error=str(e))
                answer = _generate_fallback(query, context)

        # Extract citations from chunks
        citations = [
            {"source": c.get("source", "unknown"), "score": c.get("score", 0.0)}
            for c in chunks
            if c.get("score", 0) > 0.3
        ]

        if span:
            span.set_attribute("generator.answer_length", len(answer))
            span.set_attribute("generator.citations", len(citations))

        logger.info(
            "generator.done",
            answer_length=len(answer),
            citations=len(citations),
        )

    return {
        **state,
        "draft_answers": [answer],
        "final_answer": answer,
        "citations": citations,
    }


def _generate_with_llm(query: str, context: str) -> str:
    """Generate answer using Groq LLM."""
    from langchain_core.messages import HumanMessage

    from polymind.infrastructure.llm.llm_factory import LLMFactory

    factory = LLMFactory()
    llm = factory.get_llm(tier="reasoning")

    prompt = f"""You are PolyMind, a helpful knowledge assistant.

Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Be concise and cite sources when possible.

## User Question
{query}

## Retrieved Context
{context}

## Answer
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _generate_fallback(query: str, context: str) -> str:
    """Fallback generation when LLM is unavailable."""
    if context == "No context available.":
        return (
            "I don't have enough context to answer this question. "
            "Please provide more information or upload relevant documents."
        )

    # Simple extractive fallback
    return (
        f"Based on the available context, here is what I found:\n\n"
        f"{context[:1000]}"
    )


def _generate_with_moa(query: str, context: str) -> str:
    """Generate answer using Mixture-of-Agents.

    Spawns multiple generator agents with different prompts/temperatures,
    then merges the best parts into a final answer.
    """
    from polymind.infrastructure.async_utils import run_async

    from polymind.infrastructure.moa import generate_with_moa

    return run_async(generate_with_moa(query, context, num_agents=3))
