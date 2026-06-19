"""Shared test fixtures — provides reusable mocks, stubs, and test data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest


# ── Query Fixtures ────────────────────────────────────────
@pytest.fixture
def sample_query_text() -> str:
    """A sample text query for testing."""
    return "What is the refund policy for digital goods?"


@pytest.fixture
def sample_query_state() -> dict[str, Any]:
    """A minimal PolyMindState for testing graph nodes."""
    return {
        "user_query": "What is RAG?",
        "user_id": "test_user",
        "audio_path": None,
        "image_path": None,
        "file_path": None,
        "modality": "text",
        "intent": "",
        "retrieval_strategy": "standard",
        "asr_transcript": None,
        "vqa_result": None,
        "docqa_result": None,
        "tableqa_result": None,
        "past_episodes": [],
        "semantic_facts": [],
        "planning_context": {},
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "draft_answers": [],
        "final_answer": None,
        "citations": [],
        "critic_scores": {},
        "passed_critic": False,
        "retry_count": 0,
        "should_retry": False,
    }


@pytest.fixture
def sample_query_with_chunks() -> dict[str, Any]:
    """State with pre-populated retrieved chunks for generator testing."""
    return {
        "user_query": "What is RAG?",
        "user_id": "test_user",
        "audio_path": None,
        "image_path": None,
        "file_path": None,
        "modality": "text",
        "intent": "factual_qa",
        "retrieval_strategy": "standard",
        "asr_transcript": None,
        "vqa_result": None,
        "docqa_result": None,
        "tableqa_result": None,
        "past_episodes": [],
        "semantic_facts": [],
        "planning_context": {},
        "retrieved_chunks": [
            {
                "text": "RAG stands for Retrieval Augmented Generation. It combines retrieval from a knowledge base with LLM generation.",
                "source": "rag_overview.txt",
                "score": 0.92,
            },
            {
                "text": "RAG systems retrieve relevant documents before generating answers to reduce hallucination.",
                "source": "hallucination_paper.txt",
                "score": 0.85,
            },
        ],
        "retrieval_scores": [0.92, 0.85],
        "draft_answers": [],
        "final_answer": None,
        "citations": [],
        "critic_scores": {},
        "passed_critic": False,
        "retry_count": 0,
        "should_retry": False,
    }


# ── Domain Fixtures ───────────────────────────────────────
@pytest.fixture
def sample_chunks() -> list:
    """Create sample DocumentChunk objects for testing."""
    from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk

    return [
        DocumentChunk(
            text="RAG combines retrieval and generation.",
            metadata=ChunkMetadata(
                source="doc1.txt",
                file_type="txt",
                chunk_index=0,
                modality="text",
            ),
            score=0.9,
        ),
        DocumentChunk(
            text="Vector databases store embeddings for similarity search.",
            metadata=ChunkMetadata(
                source="doc2.txt",
                file_type="txt",
                chunk_index=1,
                modality="text",
            ),
            score=0.75,
        ),
    ]


# ── Mock Fixtures ─────────────────────────────────────────
@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM that returns a fixed response."""
    mock = MagicMock()
    response = MagicMock()
    response.content = "RAG is Retrieval Augmented Generation."
    mock.invoke.return_value = response
    return mock


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns fixed vectors."""
    mock = MagicMock()
    mock.embed.return_value = [[0.1] * 384, [0.2] * 384]
    mock.embed_single.return_value = [0.1] * 384
    mock.dimension = 384
    return mock


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client that returns empty results."""
    mock = MagicMock()
    collections_mock = MagicMock()
    collections_mock.collections = []
    mock.get_collections.return_value = collections_mock
    mock.search.return_value = []
    return mock


# ── File Fixtures ─────────────────────────────────────────
@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file for ingestion testing."""
    content = (
        "PolyMind is a self-evaluating, multimodal, multi-agent "
        "knowledge assistant. It routes queries across 7+ HuggingFace "
        "task types and uses a Critic agent for self-evaluation.\n\n"
        "The system uses Qdrant for vector storage, LangGraph for "
        "agent orchestration, and HippoRAG for multi-hop retrieval."
    )
    file = tmp_path / "test_doc.txt"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file for table testing."""
    content = "question,answer\nWhat is RAG?,Retrieval Augmented Generation\nHow does it work?,By retrieving docs then generating\n"
    file = tmp_path / "test_table.csv"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def sample_benchmark_path() -> Path:
    """Path to the benchmark dataset."""
    return Path(__file__).parent / "eval" / "benchmark_v1.json"


@pytest.fixture
def sample_benchmark_cases() -> list[dict]:
    """Load benchmark cases from the JSON file."""
    path = Path(__file__).parent / "eval" / "benchmark_v1.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []
