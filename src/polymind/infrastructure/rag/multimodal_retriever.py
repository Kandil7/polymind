"""Multi-Modal Retriever — cross-modal search using CLIP embeddings.

Enables searching images with text queries and vice versa.
Uses CLIP's shared embedding space for unified text-image retrieval.

Usage:
    from polymind.infrastructure.rag.multimodal_retriever import MultiModalRetriever

    retriever = MultiModalRetriever()
    results = await retriever.search("a cat sitting on a table", modality="image")
"""

from __future__ import annotations

import structlog

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.domain.interfaces.retriever import IRetriever

logger = structlog.get_logger()

DEFAULT_TOP_K = 5


class MultiModalRetriever(IRetriever):
    """Cross-modal retriever using CLIP embeddings.

    Supports:
    - Text-to-image search
    - Image-to-text search
    - Text-to-text search (multilingual via CLIP)
    """

    def __init__(self, clip_embedder: object | None = None) -> None:
        """Initialize multi-modal retriever.

        Args:
            clip_embedder: CLIPEmbedder instance (lazy-loaded if None).
        """
        self._clip = clip_embedder
        self._documents: dict[str, dict] = {}  # doc_id -> {text, image_path, embedding}
        self._embeddings: dict[str, list[float]] = {}

    def _ensure_clip(self) -> None:
        """Lazy-load CLIP embedder."""
        if self._clip is None:
            from polymind.infrastructure.rag.clip_embedder import CLIPEmbedder
            self._clip = CLIPEmbedder()

    @property
    def is_available(self) -> bool:
        """Check if CLIP is available."""
        self._ensure_clip()
        return self._clip is not None and self._clip.is_available

    async def retrieve(
        self, query: str, top_k: int = 5, **kwargs: object
    ) -> list[DocumentChunk]:
        """Retrieve documents using CLIP-based similarity.

        Args:
            query: Search query (text).
            top_k: Number of results.
            **kwargs: Additional arguments (modality, image_path).

        Returns:
            List of DocumentChunks ranked by CLIP similarity.
        """
        if not self._documents:
            return []

        self._ensure_clip()

        if self._clip is None or not self._clip.is_available:
            logger.warning("clip.unavailable")
            return []

        import numpy as np

        # Get query embedding
        query_vec = self._clip.embed_text(query)

        # Compute similarities with all documents
        scores = []
        for doc_id, doc_data in self._documents.items():
            doc_embedding = doc_data.get("embedding")
            if doc_embedding is None:
                continue

            similarity = self._clip.compute_similarity(query_vec, doc_embedding)
            scores.append((doc_id, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        results = []
        for doc_id, score in scores[:top_k]:
            doc_data = self._documents[doc_id]
            chunk = DocumentChunk(
                text=doc_data.get("text", ""),
                metadata=ChunkMetadata(
                    source=doc_data.get("source", doc_id),
                    file_type=doc_data.get("file_type", "unknown"),
                    modality=doc_data.get("modality", "text"),
                ),
                score=score,
            )
            results.append(chunk)

        logger.info("multimodal.retrieve.done", results=len(results))
        return results

    async def index(self, chunks: list[DocumentChunk]) -> None:
        """Index documents using CLIP embeddings.

        Args:
            chunks: List of DocumentChunks to index.
        """
        self._ensure_clip()

        if self._clip is None or not self._clip.is_available:
            logger.warning("clip.unavailable_for_indexing")
            return

        for chunk in chunks:
            doc_id = f"doc_{len(self._documents)}"
            embedding = self._clip.embed_text(chunk.text)

            self._documents[doc_id] = {
                "text": chunk.text,
                "source": chunk.metadata.source,
                "file_type": chunk.metadata.file_type,
                "modality": chunk.metadata.modality,
                "embedding": embedding,
            }
            self._embeddings[doc_id] = embedding

        logger.info("multimodal.index.done", documents=len(self._documents))

    def index_image(
        self,
        image_path: str,
        caption: str = "",
        source: str = "",
    ) -> None:
        """Index an image with its caption.

        Args:
            image_path: Path to the image file.
            caption: Text caption for the image.
            source: Source identifier.
        """
        self._ensure_clip()

        if self._clip is None or not self._clip.is_available:
            logger.warning("clip.unavailable_for_image_indexing")
            return

        try:
            image_embedding = self._clip.embed_image(image_path)
            doc_id = f"img_{len(self._documents)}"

            self._documents[doc_id] = {
                "text": caption or f"Image: {Path(image_path).name}",
                "image_path": image_path,
                "source": source or image_path,
                "file_type": "image",
                "modality": "image",
                "embedding": image_embedding,
            }
            self._embeddings[doc_id] = image_embedding

            logger.info("multimodal.image_indexed", doc_id=doc_id)

        except Exception as e:
            logger.error("multimodal.image_index_failed", error=str(e))

    def search_images(self, text_query: str, top_k: int = 5) -> list[dict]:
        """Search for images using a text query.

        Args:
            text_query: Text query.
            top_k: Number of results.

        Returns:
            List of dicts with image_path, caption, and score.
        """
        self._ensure_clip()

        if self._clip is None or not self._clip.is_available:
            return []

        import numpy as np

        query_vec = self._clip.embed_text(text_query)

        # Filter to image documents
        image_docs = {
            doc_id: doc_data
            for doc_id, doc_data in self._documents.items()
            if doc_data.get("modality") == "image"
        }

        if not image_docs:
            return []

        # Compute similarities
        scores = []
        for doc_id, doc_data in image_docs.items():
            doc_embedding = doc_data.get("embedding")
            if doc_embedding is None:
                continue

            similarity = self._clip.compute_similarity(query_vec, doc_embedding)
            scores.append({
                "image_path": doc_data.get("image_path", ""),
                "caption": doc_data.get("text", ""),
                "source": doc_data.get("source", ""),
                "score": similarity,
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)


# Need to import Path for type hints
from pathlib import Path  # noqa: E402
