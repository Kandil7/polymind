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
        from polymind.application.graph import build_graph

        graph = build_graph()

        result = graph.invoke({
            "user_query": question,
            "user_id": user_id,
            "audio_path": audio_path,
            "image_path": image_path,
            "file_path": file_path,
        })

        # Extract confidence from critic scores
        scores = result.get("critic_scores", {})
        faithfulness = 0.5
        if isinstance(scores, dict):
            f = scores.get("faithfulness", 0.5)
            faithfulness = f.get("score", 0.5) if isinstance(f, dict) else f

        processing_time = (time.time() - start_time) * 1000

        # Record Prometheus metrics
        try:
            from polymind.api.middleware.metrics import record_query

            modality = result.get("modality", "text")
            passed = result.get("passed_critic", False)
            record_query(modality, passed, faithfulness)
        except Exception:
            pass  # Don't fail request if metrics recording fails

        return QueryResponse(
            answer=result.get("final_answer", ""),
            modality=result.get("modality", "text"),
            confidence=faithfulness,
            citations=result.get("citations", []),
            critic_scores=scores,
            retry_count=result.get("retry_count", 0),
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
