"""DocumentChunk domain entity."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to a document chunk."""

    source: str
    file_type: str
    page: int | None = None
    chunk_index: int = 0
    modality: str = "text"

    model_config = {"frozen": True}


class DocumentChunk(BaseModel):
    """A chunk of ingested content stored in the vector database."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding_id: str | None = None
    metadata: ChunkMetadata
    score: float | None = None

    model_config = {"frozen": True}
