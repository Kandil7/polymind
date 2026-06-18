"""RetrievalStrategy value object — selects RAG retrieval approach."""

from enum import Enum


class RetrievalStrategy(str, Enum):
    """Strategy for retrieving context from the knowledge base."""

    SKIP = "skip"
    STANDARD = "standard"
    HIPPORAG = "hipporag"
    SPECULATIVE = "speculative"
    SPARC = "sparc"
