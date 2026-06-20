"""Cross-Encoder Reranker — improves retrieval quality by rescoring candidates.

Uses a cross-encoder model to precisely score (query, document) pairs.
Pipeline: bi-encoder retrieves 20 candidates → cross-encoder reranks → top 5 returned.

Based on: BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20


class CrossEncoderReranker:
    """Cross-encoder reranker for RAG retrieval quality.

    Reranks initial retrieval candidates using a more precise but slower
    cross-encoder model that scores query-document relevance directly.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_id: HuggingFace cross-encoder model identifier.
            top_k: Number of final results to return after reranking.
        """
        self._model_id = model_id
        self._top_k = top_k
        self._model = None
        self._lazy_load()

    def _lazy_load(self) -> None:
        """Lazy-load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_id)
            logger.info("reranker.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("reranker.model.load_failed", error=str(e))
            self._model = None

    @property
    def is_available(self) -> bool:
        """Check if the reranker model is loaded."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: list[str],
        scores: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            scores: Optional original scores (unused, kept for API compat).
            top_k: Override default top_k.

        Returns:
            List of (index, score) tuples, sorted by relevance descending.
            If model unavailable, returns original order with simulated scores.
        """
        if not documents:
            return []

        k = top_k or self._top_k

        if self._model is None:
            # Fallback: return original order with declining scores
            logger.debug("reranker.fallback", reason="model_not_loaded")
            return [
                (i, max(0.1, 1.0 - i * 0.1))
                for i in range(min(k, len(documents)))
            ]

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Score all pairs
        rerank_scores = self._model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            enumerate(rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top_k
        result = [
            (idx, float(score))
            for idx, score in ranked[:k]
        ]

        logger.info(
            "reranker.done",
            candidates=len(documents),
            returned=len(result),
            top_score=f"{result[0][1]:.3f}" if result else "N/A",
        )

        return result
