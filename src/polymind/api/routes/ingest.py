"""Ingest endpoint — document ingestion into the knowledge base."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, File, Form, UploadFile

from polymind.api.schemas.ingest import IngestResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),  # noqa: B008
    source_name: str | None = Form(default=None),
    collection: str = Form(default="polymind"),
) -> IngestResponse:
    """Ingest a document into the knowledge base.

    Accepts PDF, CSV, or text files. Chunks, embeds, and stores in Qdrant.
    """
    start_time = time.time()

    import shutil
    import uuid
    from pathlib import Path

    if not file.filename:
        return IngestResponse(
            status="error",
            chunks_created=0,
            source="",
            processing_time_ms=0,
        )

    tmp = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from polymind.infrastructure.rag.embedder import Embedder
        from polymind.infrastructure.rag.ingestion import IngestionPipeline

        embedder = Embedder()
        pipeline = IngestionPipeline(embedder, collection=collection)

        import asyncio

        chunks = asyncio.get_event_loop().run_until_complete(
            pipeline.ingest_file(str(tmp), source_name=source_name)
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "ingest.done",
            source=file.filename,
            chunks=len(chunks),
        )

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            source=file.filename,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("ingest.failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return IngestResponse(
            status="error",
            chunks_created=0,
            source=file.filename,
            processing_time_ms=round(processing_time, 2),
        )
