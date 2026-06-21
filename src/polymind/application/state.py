"""PolyMindState — LangGraph state definition.

This TypedDict is the single source of truth for all data flowing through the agent graph.
Every node reads from and writes to this state.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from polymind.domain.entities.query import ScoreResult  # noqa: TCH001


class PolyMindState(TypedDict, total=False):
    """State shared across all LangGraph nodes."""

    # ── Input ──────────────────────────────────
    user_query: str
    audio_path: str | None
    image_path: str | None
    file_path: str | None
    user_id: str

    # ── Routing ────────────────────────────────
    modality: Literal["text", "audio", "image", "document", "table", "multi"]
    intent: str
    retrieval_strategy: Literal["skip", "standard", "hipporag", "speculative", "sparc"]

    # ── Specialist outputs ──────────────────────
    asr_transcript: str | None
    vqa_result: dict | None
    docqa_result: dict | None
    tableqa_result: dict | None

    # ── Memory context ──────────────────────────
    past_episodes: list[dict]
    semantic_facts: list[str]
    planning_context: dict

    # ── RAG ────────────────────────────────────
    retrieved_chunks: list[dict]
    retrieval_scores: list[float]

    # ── Generation ─────────────────────────────
    draft_answers: list[str]
    final_answer: str | None
    citations: list[dict]

    # ── Evaluation ─────────────────────────────
    critic_scores: dict[str, ScoreResult]
    passed_critic: bool
    retry_count: int
    should_retry: bool

    # ── Streaming tracking ─────────────────────
    current_node: str
