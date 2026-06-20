"""RAG node — retrieves context from the knowledge base.

Supports multiple retrieval strategies:
- skip: No retrieval (simple factual queries)
- standard: Dense vector search via Qdrant + cross-encoder reranking
- hipporag: Knowledge Graph + Personalized PageRank
- speculative: Draft-first, verify-after (standard + extra validation)
"""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState
from polymind.infrastructure.async_utils import run_async

logger = structlog.get_logger()

# Retrieve more candidates for reranking
INITIAL_RETRIEVAL_K = 20
FINAL_TOP_K = 5


def run(state: PolyMindState) -> PolyMindState:
    """Retrieve relevant context for the query.

    Reads: user_query, retrieval_strategy, asr_transcript, vqa_result,
           docqa_result, tableqa_result
    Writes: retrieved_chunks, retrieval_scores
    """
    from polymind.infrastructure.tracing import trace_span

    query = _build_effective_query(state)
    strategy = state.get("retrieval_strategy", "standard")

    with trace_span("rag", {"strategy": strategy, "query.length": len(query)}) as span:
        logger.info("rag.start", strategy=strategy, query_length=len(query))

        try:
            # Check if retrieval should be skipped due to service degradation
            from polymind.infrastructure.degradation import degradation

            if degradation.should_skip_retrieval():
                logger.warning("rag.degraded", reason="qdrant_or_embedder_unavailable")
                if span:
                    span.set_attribute("rag.degraded", True)
                return {
                    **state,
                    "retrieved_chunks": [],
                    "retrieval_scores": [],
                }

            if strategy == "skip":
                # Skip retrieval — rely on LLM parametric knowledge
                logger.info("rag.skip", reason="simple_factual_query")
                return {
                    **state,
                    "retrieved_chunks": [],
                    "retrieval_scores": [],
                }

            chunks = _retrieve_by_strategy(query, state, strategy)

            # Record success for circuit breaker
            degradation.record_service_success("qdrant")
            degradation.record_service_success("embedder")

            retrieved = [
                {
                    "text": c.text,
                    "source": c.metadata.source,
                    "score": c.score or 0.0,
                }
                for c in chunks
            ]
            scores = [c.score or 0.0 for c in chunks]

            if span:
                span.set_attribute("rag.chunks_found", len(retrieved))

            logger.info("rag.done", chunks=len(retrieved), strategy=strategy)
            return {
                **state,
                "retrieved_chunks": retrieved,
                "retrieval_scores": scores,
            }

        except Exception as e:
            # Record failure for circuit breaker
            try:
                from polymind.infrastructure.degradation import degradation
                degradation.record_service_failure("qdrant")
                degradation.record_service_failure("embedder")
            except Exception:
                pass

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
        chunks = _retrieve_hipporag(query, state)
    elif strategy == "speculative":
        chunks = _retrieve_speculative(query, state)
    else:
        chunks = _retrieve_standard(query, state)

    # Apply reranking to all strategies (except skip)
    if chunks:
        chunks = _rerank(query, chunks)

    return chunks


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

    # Retrieve more candidates for reranking
    return run_async(repo.retrieve(query, top_k=INITIAL_RETRIEVAL_K))


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

        return run_async(retriever.retrieve(query, top_k=INITIAL_RETRIEVAL_K))

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
    return run_async(repo.retrieve(query, top_k=INITIAL_RETRIEVAL_K))


def _rerank(query: str, chunks: list) -> list:
    """Rerank retrieved chunks using cross-encoder.

    Args:
        query: The search query.
        chunks: List of DocumentChunks from initial retrieval.

    Returns:
        Reranked list of DocumentChunks (top FINAL_TOP_K).
    """
    if len(chunks) <= FINAL_TOP_K:
        return chunks

    try:
        from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
        from polymind.infrastructure.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        if not reranker.is_available:
            # Fallback: just return first FINAL_TOP_K
            logger.debug("reranker.unavailable")
            return chunks[:FINAL_TOP_K]

        # Extract texts for reranking
        texts = [c.text for c in chunks]

        # Rerank
        ranked = reranker.rerank(query, texts, top_k=FINAL_TOP_K)

        # Rebuild chunks with reranked scores
        reranked_chunks = []
        for idx, score in ranked:
            original = chunks[idx]
            reranked_chunks.append(
                DocumentChunk(
                    text=original.text,
                    metadata=original.metadata,
                    score=score,
                )
            )

        logger.info(
            "reranker.applied",
            before=len(chunks),
            after=len(reranked_chunks),
            top_score=f"{reranked_chunks[0].score:.3f}" if reranked_chunks else "N/A",
        )

        return reranked_chunks

    except Exception as e:
        logger.warning("reranker.failed_fallback", error=str(e))
        return chunks[:FINAL_TOP_K]
