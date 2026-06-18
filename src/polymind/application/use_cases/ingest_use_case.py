"""IngestUseCase — orchestrates document ingestion into the knowledge base."""

from __future__ import annotations

from polymind.domain.exceptions import IngestionError


class IngestUseCase:
    """Orchestrates document ingestion pipeline.

    Phase 1: Stub — will be implemented in Phase 3 (HippoRAG).
    """

    async def execute(self, file_path: str) -> dict[str, str]:
        """Ingest a document into the knowledge base.

        Raises:
            IngestionError: Always in Phase 1 (not yet implemented).
        """
        raise IngestionError("Ingestion pipeline not yet implemented — Phase 3")
