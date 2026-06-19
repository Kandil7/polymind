"""Embedding model wrapper — BAAI/bge-m3 for dense vector search."""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

DEFAULT_MODEL = "BAAI/bge-m3"


class Embedder:
    """Dense embedding model wrapper using sentence-transformers."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize the embedder.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._model = None
        self._lazy_load()

    def _lazy_load(self) -> None:
        """Lazy-load the model to avoid import-time allocation."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id)
            logger.info("embedder.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("embedder.model.load_failed", error=str(e))
            self._model = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._model is None:
            return 1024  # bge-m3 default
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError(
                f"Embedding model '{self._model_id}' failed to load."
            )

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: String to embed.

        Returns:
            Embedding vector.
        """
        return self.embed([text])[0]
