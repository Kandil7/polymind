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
    Routes through the full agent graph: Planner → Router → Specialists → RAG → Generator → Critic.
    """
    start_time = time.time()

    # Save uploaded files to temp
    import shutil
    import uuid
    from pathlib import Path

    audio_path = image_path = file_path = None

    for upload, attr in [
        (audio_file, "audio_path"),
        (image_file, "image_path"),
        (doc_file, "file_path"),
    ]:
        if upload and upload.filename:
            tmp = Path(f"/tmp/{uuid.uuid4()}_{upload.filename}")
            with tmp.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            locals()[attr]  # noqa: B015
            if attr == "audio_path":
                audio_path = str(tmp)
            elif attr == "image_path":
                image_path = str(tmp)
            elif attr == "file_path":
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
