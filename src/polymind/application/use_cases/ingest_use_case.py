"""IngestUseCase — orchestrates document ingestion into the knowledge base.

Handles the full pipeline: file → text extraction → chunking → embedding → Qdrant indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from polymind.domain.exceptions import IngestionError

logger = structlog.get_logger()


@dataclass
class IngestResult:
    """Result of a document ingestion operation.

    Attributes:
        status: Ingestion status ("success" or "error").
        chunks_created: Number of chunks produced.
        source: Source file name or identifier.
        collection: Qdrant collection name.
        processing_time_ms: Total processing time in milliseconds.
        errors: List of error messages (empty on success).
    """

    status: str
    chunks_created: int
    source: str
    collection: str
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class IngestUseCase:
    """Orchestrates document ingestion pipeline.

    Pipeline: File → Text Extraction → Chunking → Embedding → Qdrant Indexing

    Attributes:
        _collection: Qdrant collection name.
        _qdrant_url: Qdrant server URL.
    """

    def __init__(
        self,
        collection: str = "polymind",
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        """Initialize the ingestion use case.

        Args:
            collection: Qdrant collection name.
            qdrant_url: Qdrant server URL.
        """
        self._collection = collection
        self._qdrant_url = qdrant_url

    async def execute(
        self,
        file_path: str,
        source_name: str | None = None,
    ) -> IngestResult:
        """Ingest a document into the knowledge base.

        Args:
            file_path: Path to the document to ingest.
            source_name: Optional override for the source name.

        Returns:
            IngestResult with status and chunk count.

        Raises:
            IngestionError: If any step of the pipeline fails.
        """
        import time
        from pathlib import Path

        start = time.time()
        path = Path(file_path)

        if not path.exists():
            raise IngestionError(
                f"File not found: {path}",
                details={"file_path": str(path)},
            )

        source = source_name or path.name

        try:
            # Lazy-load infrastructure to avoid import-time side effects
            from polymind.infrastructure.qdrant.chunk_repository import (
                QdrantChunkRepository,
            )
            from polymind.infrastructure.qdrant.client_factory import (
                get_qdrant_client,
            )
            from polymind.infrastructure.rag.embedder import Embedder
            from polymind.infrastructure.rag.ingestion import IngestionPipeline

            # Initialize pipeline components
            embedder = Embedder()
            pipeline = IngestionPipeline(
                embedder, collection=self._collection
            )

            # Step 1: Extract text and create chunks
            chunks = await pipeline.ingest_file(
                str(path), source_name=source
            )

            if not chunks:
                raise IngestionError(
                    "No chunks produced from document",
                    details={"source": source},
                )

            # Step 2: Index chunks into Qdrant
            client = get_qdrant_client(url=self._qdrant_url)
            repo = QdrantChunkRepository(
                client, embedder, collection=self._collection
            )
            await repo.index(chunks)

            elapsed = (time.time() - start) * 1000

            logger.info(
                "ingest.usecase.done",
                source=source,
                chunks=len(chunks),
                elapsed_ms=round(elapsed, 2),
            )

            return IngestResult(
                status="success",
                chunks_created=len(chunks),
                source=source,
                collection=self._collection,
                processing_time_ms=round(elapsed, 2),
            )

        except IngestionError:
            raise
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(
                "ingest.usecase.failed",
                source=source,
                error=str(e),
            )
            raise IngestionError(
                f"Ingestion failed: {e}",
                details={
                    "source": source,
                    "processing_time_ms": round(elapsed, 2),
                },
            ) from e
