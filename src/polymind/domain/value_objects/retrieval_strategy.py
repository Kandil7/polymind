"""RetrievalStrategy value object — selects RAG retrieval approach."""

from enum import Enum


class RetrievalStrategy(str, Enum):
    """Strategy for retrieving context from the knowledge base.

    Strategies:
    - SKIP: Simple factual queries (LLM answers from parametric knowledge)
    - STANDARD: Dense vector search via Qdrant
    - HIPPORAG: Multi-hop reasoning (Knowledge Graph + Personalized PageRank)
    - SPECULATIVE: Draft first, verify after (more chunks retrieved)
    """

    SKIP = "skip"
    STANDARD = "standard"
    HIPPORAG = "hipporag"
    SPECULATIVE = "speculative"
