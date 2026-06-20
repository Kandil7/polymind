"""Specialist nodes — LangGraph nodes for each modality specialist."""

from __future__ import annotations

import structlog

from polymind.application.state import PolyMindState
from polymind.infrastructure.async_utils import run_async

logger = structlog.get_logger()


def asr_node(state: PolyMindState) -> PolyMindState:
    """Process audio input using ASR specialist.

    Reads: audio_path
    Writes: asr_transcript
    """
    audio_path = state.get("audio_path", "")

    try:
        from polymind.infrastructure.specialists.asr_wrapper import ASRWrapper

        asr = ASRWrapper()
        result = run_async(asr.process(audio_path))
        transcript = result.get("text", "")

        logger.info("asr.done", transcript_length=len(transcript))
        return {**state, "asr_transcript": transcript}

    except Exception as e:
        logger.error("asr.failed", error=str(e))
        return {**state, "asr_transcript": f"ASR failed: {e}"}


def vqa_node(state: PolyMindState) -> PolyMindState:
    """Process image input using VQA specialist.

    Reads: image_path, user_query
    Writes: vqa_result
    """
    image_path = state.get("image_path", "")
    question = state.get("user_query", "What is in this image?")

    try:
        from polymind.infrastructure.specialists.vqa_wrapper import VQAWrapper

        vqa = VQAWrapper()
        result = run_async(vqa.process(image_path, question=question))

        logger.info("vqa.done", answer=result.get("answer"))
        return {**state, "vqa_result": result}

    except Exception as e:
        logger.error("vqa.failed", error=str(e))
        return {**state, "vqa_result": {"answer": f"VQA failed: {e}"}}


def docqa_node(state: PolyMindState) -> PolyMindState:
    """Process document input using DocQA specialist.

    Reads: file_path, user_query
    Writes: docqa_result
    """
    file_path = state.get("file_path", "")
    question = state.get("user_query", "Summarize this document.")

    try:
        from polymind.infrastructure.specialists.docqa_wrapper import (
            DocQAWrapper,
        )

        docqa = DocQAWrapper()
        result = run_async(docqa.process(file_path, question=question))

        logger.info("docqa.done", answer=result.get("answer"))
        return {**state, "docqa_result": result}

    except Exception as e:
        logger.error("docqa.failed", error=str(e))
        return {**state, "docqa_result": {"answer": f"DocQA failed: {e}"}}


def tableqa_node(state: PolyMindState) -> PolyMindState:
    """Process table input using TableQA specialist.

    Reads: file_path, user_query
    Writes: tableqa_result
    """
    file_path = state.get("file_path", "")
    question = state.get("user_query", "What does this table show?")

    try:
        from polymind.infrastructure.specialists.tableqa_wrapper import (
            TableQAWrapper,
        )

        tableqa = TableQAWrapper()
        result = run_async(tableqa.process(file_path, question=question))

        logger.info("tableqa.done", answer=result.get("answer"))
        return {**state, "tableqa_result": result}

    except Exception as e:
        logger.error("tableqa.failed", error=str(e))
        return {**state, "tableqa_result": {"answer": f"TableQA failed: {e}"}}
