"""Qdrant chunk repository — IRetriever implementation with Qdrant."""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient  # noqa: TCH002
from qdrant_client.models import (
    Distance,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.domain.interfaces.retriever import IRetriever
from polymind.infrastructure.rag.embedder import Embedder  # noqa: TCH001

logger = structlog.get_logger()

COLLECTION_NAME = "polymind"


class QdrantChunkRepository(IRetriever):
    """Qdrant-backed document chunk repository.

    Stores and retrieves DocumentChunks using dense vector search.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str = COLLECTION_NAME,
    ) -> None:
        """Initialize the repository.

        Args:
            client: Qdrant client instance.
            embedder: Embedding model for query encoding.
            collection: Collection name.
        """
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant.collection.created", collection=self._collection)

    async def retrieve(
        self, query: str, top_k: int = 5, **kwargs: object
    ) -> list[DocumentChunk]:
        """Retrieve top-k chunks by semantic similarity.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of DocumentChunks ranked by relevance.
        """
        modality_filter = kwargs.get("modality")

        query_vec = self._embedder.embed_single(query)

        query_filter = None
        if modality_filter:
            query_filter = FieldCondition(
                key="modality",
                match=MatchValue(value=modality_filter),
            )

        # qdrant-client >= 1.12 uses query_points instead of search
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for r in results.points:
            payload = r.payload or {}
            chunk = DocumentChunk(
                text=payload.get("text", ""),
                metadata=ChunkMetadata(
                    source=payload.get("source", "unknown"),
                    file_type=payload.get("file_type", "text"),
                    page=payload.get("page"),
                    chunk_index=payload.get("chunk_index", 0),
                    modality=payload.get("modality", "text"),
                ),
                score=r.score,
            )
            chunks.append(chunk)

        logger.info("qdrant.retrieve.done", query_length=len(query), results=len(chunks))
        return chunks

    async def index(self, chunks: list[DocumentChunk]) -> None:
        """Index document chunks into Qdrant.

        Args:
            chunks: List of DocumentChunks to index.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            point = PointStruct(
                id=str(chunk.id),
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source": chunk.metadata.source,
                    "file_type": chunk.metadata.file_type,
                    "page": chunk.metadata.page,
                    "chunk_index": chunk.metadata.chunk_index,
                    "modality": chunk.metadata.modality,
                },
            )
            points.append(point)

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

        logger.info("qdrant.index.done", count=len(points))
