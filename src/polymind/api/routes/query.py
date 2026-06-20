"""Query endpoint — multimodal query processing with streaming support."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from polymind.api.schemas.query import QueryResponse

logger = structlog.get_logger()
router = APIRouter()

# ── Node display names for SSE progress events ──────────
NODE_LABELS = {
    "planner": "Planning",
    "router": "Routing",
    "asr": "Transcribing audio",
    "vqa": "Analyzing image",
    "docqa": "Reading document",
    "tableqa": "Reading table",
    "rag": "Retrieving context",
    "generator": "Generating answer",
    "critic": "Evaluating quality",
    "synthesizer": "Formatting response",
}


def _save_uploads(
    audio_file: UploadFile | None,
    image_file: UploadFile | None,
    doc_file: UploadFile | None,
) -> tuple[str | None, str | None, str | None]:
    """Save uploaded files to temp directory and return paths."""
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

    return audio_path, image_path, file_path


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Non-streaming endpoint (backward compatible) ────────
@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    question: str = Form(...),
    user_id: str = Form(default="anonymous"),
    audio_file: UploadFile | None = File(default=None),  # noqa: B008
    image_file: UploadFile | None = File(default=None),  # noqa: B008
    doc_file: UploadFile | None = File(default=None),  # noqa: B008
) -> QueryResponse:
    """Process a multimodal query (non-streaming).

    Returns the complete response after all agents have processed.
    For real-time progress, use POST /query/stream instead.
    """
    start_time = time.time()
    audio_path, image_path, file_path = _save_uploads(audio_file, image_file, doc_file)

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
            pass

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


# ── Streaming endpoint (SSE) ────────────────────────────
@router.post("/stream")
async def query_stream_endpoint(
    question: str = Form(...),
    user_id: str = Form(default="anonymous"),
    audio_file: UploadFile | None = File(default=None),  # noqa: B008
    image_file: UploadFile | None = File(default=None),  # noqa: B008
    doc_file: UploadFile | None = File(default=None),  # noqa: B008
) -> StreamingResponse:
    """Process a multimodal query with real-time streaming progress.

    Returns Server-Sent Events (SSE) with per-node progress updates.

    Event types:
    - **node_start**: Agent node started processing
    - **node_done**: Agent node completed with partial state
    - **complete**: Final answer with all metadata
    - **error**: An error occurred

    Example SSE stream:
    ```
    event: node_start
    data: {"node": "planner", "label": "Planning", "elapsed_ms": 0}

    event: node_done
    data: {"node": "planner", "label": "Planning", "modality": "text", "intent": "factual_qa", "elapsed_ms": 120}

    event: node_start
    data: {"node": "router", "label": "Routing", "elapsed_ms": 120}

    ...

    event: complete
    data: {"answer": "...", "confidence": 0.85, "modality": "text", "retry_count": 0, "elapsed_ms": 5230}
    ```
    """
    audio_path, image_path, file_path = _save_uploads(audio_file, image_file, doc_file)

    return StreamingResponse(
        _stream_graph(
            question=question,
            user_id=user_id,
            audio_path=audio_path,
            image_path=image_path,
            file_path=file_path,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_graph(
    question: str,
    user_id: str,
    audio_path: str | None,
    image_path: str | None,
    file_path: str | None,
) -> AsyncIterator[str]:
    """Generator that streams SSE events from the LangGraph agent graph."""
    start_time = time.time()

    initial_state = {
        "user_query": question,
        "user_id": user_id,
        "audio_path": audio_path,
        "image_path": image_path,
        "file_path": file_path,
    }

    try:
        from polymind.application.graph import build_graph

        graph = build_graph()

        # Stream with "updates" mode — yields state deltas after each node
        async for event in graph.astream(initial_state, stream_mode="updates"):
            # event is a dict: {node_name: state_delta}
            for node_name, state_delta in event.items():
                elapsed_ms = round((time.time() - start_time) * 1000, 1)
                label = NODE_LABELS.get(node_name, node_name)

                # Emit node_start
                yield _sse_event("node_start", {
                    "node": node_name,
                    "label": label,
                    "elapsed_ms": elapsed_ms,
                })

                # Emit node_done with relevant state changes
                progress_data = _extract_progress(node_name, state_delta)
                progress_data.update({
                    "node": node_name,
                    "label": label,
                    "elapsed_ms": elapsed_ms,
                })
                yield _sse_event("node_done", progress_data)

        # Build final response from the completed graph state
        # Re-invoke to get the final state (stream doesn't return it directly)
        final_state = await graph.ainvoke(initial_state)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # Emit complete event
        yield _sse_event("complete", {
            "answer": final_state.get("final_answer", ""),
            "modality": final_state.get("modality", "text"),
            "intent": final_state.get("intent", ""),
            "confidence": _extract_confidence(final_state),
            "retry_count": final_state.get("retry_count", 0),
            "citations": final_state.get("citations", []),
            "critic_scores": _extract_critic_scores(final_state),
            "elapsed_ms": elapsed_ms,
        })

    except Exception as e:
        logger.error("query.stream.failed", error=str(e))
        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        yield _sse_event("error", {
            "error": str(e),
            "elapsed_ms": elapsed_ms,
        })


def _extract_progress(node_name: str, state_delta: dict) -> dict:
    """Extract relevant progress info from a state delta."""
    progress = {}

    if node_name == "planner":
        progress["modality"] = state_delta.get("modality", "")
        progress["intent"] = state_delta.get("intent", "")

    elif node_name == "router":
        progress["strategy"] = state_delta.get("retrieval_strategy", "")

    elif node_name == "asr":
        transcript = state_delta.get("asr_transcript", "")
        progress["transcript_preview"] = transcript[:100] + "..." if len(transcript) > 100 else transcript

    elif node_name == "rag":
        chunks = state_delta.get("retrieved_chunks", [])
        progress["chunks_found"] = len(chunks)

    elif node_name == "generator":
        answer = state_delta.get("final_answer", "")
        progress["answer_preview"] = answer[:200] + "..." if len(answer) > 200 else answer

    elif node_name == "critic":
        scores = state_delta.get("critic_scores", {})
        passed = state_delta.get("passed_critic", False)
        progress["passed"] = passed
        progress["score_count"] = len(scores)

    elif node_name == "synthesizer":
        progress["status"] = "formatting"

    return progress


def _extract_confidence(state: dict) -> float:
    """Extract confidence score from final state."""
    scores = state.get("critic_scores", {})
    if isinstance(scores, dict):
        f = scores.get("faithfulness", 0.5)
        if isinstance(f, dict):
            return f.get("score", 0.5) if "score" in f else f.get("value", 0.5)
        return float(f) if f else 0.5
    return 0.5


def _extract_critic_scores(state: dict) -> dict:
    """Extract critic scores in a serializable format."""
    scores = state.get("critic_scores", {})
    result = {}
    for name, value in scores.items():
        if isinstance(value, dict):
            result[name] = value
        else:
            result[name] = {"score": float(value) if value else 0.0}
    return result
