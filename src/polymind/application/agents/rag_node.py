"""RAG node — retrieves context from the knowledge base."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def run(state: PolyMindState) -> PolyMindState:
    """Retrieve relevant context for the query.

    Reads: user_query, asr_transcript, vqa_result, docqa_result, tableqa_result
    Writes: retrieved_chunks, retrieval_scores
    """
    # Build the effective query from available inputs
    query = _build_effective_query(state)

    try:
        chunks = _retrieve(query, state)

        retrieved = [
            {
                "text": c.text,
                "source": c.metadata.source,
                "score": c.score or 0.0,
            }
            for c in chunks
        ]
        scores = [c.score or 0.0 for c in chunks]

        logger.info("rag.done", chunks=len(retrieved))
        return {
            **state,
            "retrieved_chunks": retrieved,
            "retrieval_scores": scores,
        }

    except Exception as e:
        logger.error("rag.failed", error=str(e))
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_scores": [],
        }


def _build_effective_query(state: PolyMindState) -> str:
    """Combine query with specialist outputs for better retrieval."""
    parts = [state.get("user_query", "")]

    if transcript := state.get("asr_transcript"):
        parts.append(f"Transcript: {transcript[:500]}")

    if vqa := state.get("vqa_result"):
        if answer := vqa.get("answer"):
            parts.append(f"Image shows: {answer}")

    if docqa := state.get("docqa_result"):
        if answer := docqa.get("answer"):
            parts.append(f"Document says: {answer}")

    if tableqa := state.get("tableqa_result"):
        if answer := tableqa.get("answer"):
            parts.append(f"Table shows: {answer}")

    return " ".join(parts)


def _retrieve(query: str, state: PolyMindState) -> list:
    """Retrieve from Qdrant using the best available strategy."""
    from polymind.infrastructure.qdrant.chunk_repository import (
        QdrantChunkRepository,
    )
    from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
    from polymind.infrastructure.rag.embedder import Embedder

    client = get_qdrant_client()
    embedder = Embedder()
    repo = QdrantChunkRepository(client, embedder)

    import asyncio

    return asyncio.get_event_loop().run_until_complete(
        repo.retrieve(query, top_k=5)
    )
