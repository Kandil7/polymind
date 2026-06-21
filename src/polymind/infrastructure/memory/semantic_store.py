"""Semantic Memory Store — Qdrant-backed extracted facts.

Stores reusable semantic facts extracted from repeated episodic patterns.
Used for long-term knowledge accumulation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from polymind.domain.interfaces.memory_store import IMemoryStore

logger = structlog.get_logger()

COLLECTION_NAME = "semantic_memory"


class SemanticStore(IMemoryStore):
    """Qdrant-backed semantic memory for extracted facts."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: object,
        collection: str = COLLECTION_NAME,
    ) -> None:
        """Initialize semantic store.

        Args:
            client: Qdrant client instance.
            embedder: Embedding model for vector encoding.
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
            logger.info("semantic.collection.created", collection=self._collection)

    def store(self, fact: str, source_query: str = "") -> None:
        """Store a semantic fact.

        Args:
            fact: The extracted semantic fact.
            source_query: Original query that led to this fact.
        """
        embedding = self._embedder.embed_single(fact)
        point_id = uuid.uuid4().int >> 64  # 64-bit UUID for Qdrant

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "fact": fact,
                        "source_query": source_query,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            ],
        )
        logger.info("semantic.stored", fact_length=len(fact))

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """Recall relevant semantic facts.

        Args:
            query: Search query.
            top_k: Number of facts to recall.

        Returns:
            List of relevant fact strings.
        """
        try:
            embedding = self._embedder.embed_single(query)

            results = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            facts = [r.payload.get("fact", "") for r in results.points if r.payload]
            logger.info("semantic.recalled", count=len(facts))
            return facts
        except Exception as e:
            logger.error("semantic.recall.failed", error=str(e))
            return []

    def count(self) -> int:
        """Return the number of stored facts."""
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    async def consolidate(self) -> None:
        """Consolidate semantic facts.

        Semantic store doesn't consolidate — this is a no-op.
        Consolidation is handled by FourLayerMemory.
        """
        pass
