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
    import tempfile
    import uuid
    from pathlib import Path

    if not file.filename:
        return IngestResponse(
            status="error",
            chunks_created=0,
            source="",
            processing_time_ms=0,
        )

    tmp_dir = Path(tempfile.gettempdir()) / "polymind"
    tmp_dir.mkdir(exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from polymind.application.use_cases.ingest_use_case import (
            IngestUseCase,
        )

        use_case = IngestUseCase(collection=collection)
        result = await use_case.execute(
            str(tmp), source_name=source_name
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "ingest.done",
            source=file.filename,
            chunks=result.chunks_created,
        )

        return IngestResponse(
            status=result.status,
            chunks_created=result.chunks_created,
            source=result.source,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("ingest.failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return IngestResponse(
            status="error",
            chunks_created=0,
            source=file.filename or "",
            processing_time_ms=round(processing_time, 2),
        )
