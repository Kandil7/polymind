"""Seed script — populate Qdrant with test data for development."""

from __future__ import annotations

import asyncio
import structlog

from polymind.infrastructure.qdrant.client_factory import get_qdrant_client
from polymind.infrastructure.qdrant.chunk_repository import QdrantChunkRepository
from polymind.infrastructure.rag.embedder import Embedder
from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk

logger = structlog.get_logger()

SAMPLE_PASSAGES = [
    {
        "text": "PolyMind is a self-evaluating, multimodal, multi-agent knowledge assistant. "
                "It routes user queries across 7+ HuggingFace task types.",
        "source": "polymind_overview.txt",
    },
    {
        "text": "The Critic agent evaluates outputs using RAGAS metrics before delivery. "
                "If faithfulness drops below 0.75, it triggers re-retrieval.",
        "source": "critic_agent.txt",
    },
    {
        "text": "HippoRAG v2 uses Knowledge Graphs and Personalized PageRank for "
                "multi-hop retrieval, achieving 86% accuracy on multi-hop QA.",
        "source": "hipporag_paper.txt",
    },
    {
        "text": "Qdrant is used as the vector database with hybrid sparse+dense search. "
                "It supports rich payload filtering for metadata-based queries.",
        "source": "qdrant_config.txt",
    },
    {
        "text": "LangGraph orchestrates the agent graph: Planner → Router → "
                "Specialist Agents → Critic → Synthesizer.",
        "source": "langgraph_architecture.txt",
    },
    {
        "text": "The ASR specialist uses openai/whisper-large-v3 for speech-to-text "
                "transcription with support for English and Arabic.",
        "source": "asr_specialist.txt",
    },
    {
        "text": "VQA specialist uses Salesforce/blip-vqa-base for visual question "
                "answering on images and screenshots.",
        "source": "vqa_specialist.txt",
    },
    {
        "text": "The 4-layer memory system includes episodic (Mem0), semantic (Qdrant), "
                "procedural, and working memory for context retention.",
        "source": "memory_system.txt",
    },
    {
        "text": "Mixture-of-Agents uses 3 proposer models (Qwen3, Gemma-3, Mistral) "
                "plus 1 aggregator to generate superior answers.",
        "source": "moa_generator.txt",
    },
    {
        "text": "DeepEval provides 6+ evaluation metrics: faithfulness, answer relevancy, "
                "hallucination, toxicity, bias, and contextual precision.",
        "source": "deepeval_eval.txt",
    },
]


async def seed() -> None:
    """Seed Qdrant with sample passages."""
    logger.info("seed.start", passages=len(SAMPLE_PASSAGES))

    embedder = Embedder()
    client = get_qdrant_client()
    repo = QdrantChunkRepository(client, embedder)

    chunks = []
    for i, passage in enumerate(SAMPLE_PASSAGES):
        chunk = DocumentChunk(
            text=passage["text"],
            metadata=ChunkMetadata(
                source=passage["source"],
                file_type="txt",
                chunk_index=i,
                modality="text",
            ),
        )
        chunks.append(chunk)

    await repo.index(chunks)

    logger.info("seed.done", indexed=len(chunks))


if __name__ == "__main__":
    asyncio.run(seed())
