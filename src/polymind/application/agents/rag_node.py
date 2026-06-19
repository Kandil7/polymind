"""RAG node — retrieves context from the knowledge base.

Supports multiple retrieval strategies:
- skip: No retrieval (simple factual queries)
- standard: Dense vector search via Qdrant
- hipporag: Knowledge Graph + Personalized PageRank
- speculative: Draft-first, verify-after (standard + extra validation)
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import structlog

from polymind.application.state import PolyMindState

logger = structlog.get_logger()


def _run_async(coro):
    """Run an async coroutine from a sync context safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    else:
        return asyncio.run(coro)


def run(state: PolyMindState) -> PolyMindState:
    """Retrieve relevant context for the query.

    Reads: user_query, retrieval_strategy, asr_transcript, vqa_result,
           docqa_result, tableqa_result
    Writes: retrieved_chunks, retrieval_scores
    """
    query = _build_effective_query(state)
    strategy = state.get("retrieval_strategy", "standard")

    logger.info("rag.start", strategy=strategy, query_length=len(query))

    try:
        if strategy == "skip":
            # Skip retrieval — rely on LLM parametric knowledge
            logger.info("rag.skip", reason="simple_factual_query")
            return {
                **state,
                "retrieved_chunks": [],
                "retrieval_scores": [],
            }

        chunks = _retrieve_by_strategy(query, state, strategy)

        retrieved = [
            {
                "text": c.text,
                "source": c.metadata.source,
                "score": c.score or 0.0,
            }
            for c in chunks
        ]
        scores = [c.score or 0.0 for c in chunks]

        logger.info("rag.done", chunks=len(retrieved), strategy=strategy)
        return {
            **state,
            "retrieved_chunks": retrieved,
            "retrieval_scores": scores,
        }

    except Exception as e:
        logger.error("rag.failed", error=str(e), strategy=strategy)
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


def _retrieve_by_strategy(
    query: str, state: PolyMindState, strategy: str
) -> list:
    """Route retrieval to the appropriate strategy implementation."""
    if strategy == "hipporag":
        return _retrieve_hipporag(query, state)
    elif strategy == "speculative":
        return _retrieve_speculative(query, state)
    else:
        return _retrieve_standard(query, state)


def _retrieve_standard(query: str, state: PolyMindState) -> list:
    """Standard dense vector retrieval via Qdrant."""
    from polymind.infrastructure.qdrant.chunk_repository import (
        QdrantChunkRepository,
    )
    from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
    from polymind.infrastructure.rag.embedder import Embedder

    client = get_qdrant_client()
    embedder = Embedder()
    repo = QdrantChunkRepository(client, embedder)

    return _run_async(repo.retrieve(query, top_k=5))


def _retrieve_hipporag(query: str, state: PolyMindState) -> list:
    """HippoRAG retrieval using Knowledge Graph + Personalized PageRank."""
    try:
        from polymind.infrastructure.qdrant.hipporag_retriever import (
            HippoRAGRetriever,
        )
        from polymind.infrastructure.rag.embedder import Embedder

        embedder = Embedder()
        retriever = HippoRAGRetriever(embedder)

        # Check if we have indexed passages
        if retriever.passage_count == 0:
            # No knowledge graph built yet — fall back to standard
            logger.info("hipporag.empty_fallback")
            return _retrieve_standard(query, state)

        return _run_async(retriever.retrieve(query, top_k=5))

    except Exception as e:
        logger.warning("hipporag.failed_fallback", error=str(e))
        return _retrieve_standard(query, state)


def _retrieve_speculative(query: str, state: PolyMindState) -> list:
    """Speculative retrieval: retrieve more chunks, let generator verify."""
    from polymind.infrastructure.qdrant.chunk_repository import (
        QdrantChunkRepository,
    )
    from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
    from polymind.infrastructure.rag.embedder import Embedder

    client = get_qdrant_client()
    embedder = Embedder()
    repo = QdrantChunkRepository(client, embedder)

    # Retrieve more chunks for speculative verification
    return _run_async(repo.retrieve(query, top_k=8))
