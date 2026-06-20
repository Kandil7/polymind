"""Query endpoint — multimodal query processing."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, File, Form, UploadFile

from polymind.api.schemas.query import QueryResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    question: str = Form(...),
    user_id: str = Form(default="anonymous"),
    audio_file: UploadFile | None = File(default=None),  # noqa: B008
    image_file: UploadFile | None = File(default=None),  # noqa: B008
    doc_file: UploadFile | None = File(default=None),  # noqa: B008
) -> QueryResponse:
    """Process a multimodal query.

    Accepts text questions with optional audio, image, or document attachments.
    Routes through the full agent graph: Planner -> Router -> Specialists -> RAG -> Generator -> Critic.
    """
    start_time = time.time()

    # Save uploaded files to system temp directory
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir()) / "polymind"
    tmp_dir.mkdir(exist_ok=True)

    audio_path = image_path = file_path = None

    for upload, attr_name in [
        (audio_file, "audio"),
        (image_file, "image"),
        (doc_file, "doc"),
    ]:
        if upload and upload.filename:
            tmp = tmp_dir / f"{uuid.uuid4()}_{upload.filename}"
            with tmp.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            if attr_name == "audio":
                audio_path = str(tmp)
            elif attr_name == "image":
                image_path = str(tmp)
            elif attr_name == "doc":
                file_path = str(tmp)

    try:
        from polymind.application.use_cases.query_use_case import QueryUseCase
        from polymind.domain.entities.query import Query

        use_case = QueryUseCase()
        query = Query(
            text=question,
            user_id=user_id,
            audio_path=audio_path,
            image_path=image_path,
            file_path=file_path,
        )
        result = await use_case.execute(query)

        processing_time = (time.time() - start_time) * 1000

        # Record Prometheus metrics
        try:
            from polymind.api.middleware.metrics import record_query

            faithfulness = result.answer.confidence
            passed = result.answer.confidence >= 0.7
            record_query(result.modality, passed, faithfulness)
        except Exception:
            pass  # Don't fail request if metrics recording fails

        return QueryResponse(
            answer=result.answer.text,
            modality=result.modality,
            confidence=result.answer.confidence,
            citations=[{"source": c.metadata.source, "score": c.score or 0.0} for c in result.citations],
            critic_scores={name: {"score": s.value, "passed": s.passed} for name, s in result.critic_scores.items()},
            retry_count=result.retry_count,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("query.failed", error=str(e))
        processing_time = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=f"Error: {e}",
            modality="unknown",
            confidence=0.0,
            citations=[],
            critic_scores={},
            retry_count=0,
            processing_time_ms=round(processing_time, 2),
        )
