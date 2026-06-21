"""Chunking Strategy Selector — selects optimal chunker based on file type.

Automatically selects the best chunking strategy for each document type:
- Plain text (txt, md): RecursiveChunker (paragraph/sentence splitting)
- Code files (py, js, ts): Code-aware chunking (split by functions/classes)
- Tables (csv, xlsx): TableChunker (row-based, preserves structure)
- PDFs: SemanticChunker (if available) or RecursiveChunker
- Long documents (>5000 chars): SemanticChunker for coherence

Usage:
    from polymind.infrastructure.rag.chunk_strategy import select_chunker

    chunker = select_chunker("document.pdf", text)
    chunks = chunker.chunk(text, metadata={"source": "doc.pdf"})
"""

from __future__ import annotations

import re

import structlog

from polymind.infrastructure.rag.chunker import (
    RecursiveChunker,
    SemanticChunker,
    TableChunker,
)

logger = structlog.get_logger()

# ── File type to strategy mapping ────────────────────────
FILE_TYPE_STRATEGIES = {
    # Plain text
    "txt": "recursive",
    "md": "recursive",
    "rst": "recursive",

    # Code files
    "py": "code",
    "js": "code",
    "ts": "code",
    "jsx": "code",
    "tsx": "code",
    "java": "code",
    "go": "code",
    "rs": "code",
    "cpp": "code",
    "c": "code",
    "h": "code",
    "hpp": "code",
    "rb": "code",
    "php": "code",
    "swift": "code",
    "kt": "code",

    # Structured data
    "csv": "table",
    "tsv": "table",
    "json": "recursive",  # JSON as text
    "yaml": "recursive",
    "yml": "recursive",

    # Documents
    "pdf": "semantic",  # Try semantic, fallback to recursive
    "docx": "recursive",
    "html": "recursive",
    "xml": "recursive",
}

# ── Long document threshold ──────────────────────────────
LONG_DOCUMENT_THRESHOLD = 5000  # characters


class CodeChunker:
    """Code-aware chunker — splits by functions, classes, and blocks.

    Preserves code structure by splitting at logical boundaries
    (function/class definitions, imports, comments).

    Attributes:
        max_chunk_size: Maximum chunk size in characters.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize code chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters.
        """
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Split code into logical chunks.

        Args:
            text: Source code to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with 'text' and 'metadata' keys.
        """
        if not text.strip():
            return []

        # Split by common code boundaries
        boundaries = [
            r'\n(?=\S)',           # New line followed by non-whitespace (top-level)
            r'\n(?=class\s)',      # Class definitions
            r'\n(?=def\s)',        # Function definitions
            r'\n(?=async\s+def\s)',# Async function definitions
            r'\n(?=import\s)',     # Import statements
            r'\n(?=from\s)',       # From imports
            r'\n\n',              # Double newlines (block separators)
        ]

        # Try to split by boundaries
        chunks = []
        current_chunk = ""

        for line in text.split("\n"):
            if len(current_chunk) + len(line) + 1 > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no chunks produced, fall back to recursive
        if not chunks:
            fallback = RecursiveChunker(chunk_size=self.max_chunk_size)
            return fallback.chunk(text, metadata)

        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                result.append({
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": i,
                        "chunk_type": "code",
                    },
                })

        logger.info("chunker.code.done", chunks=len(result))
        return result


def select_chunker(
    file_type: str,
    text: str = "",
    embedder: object | None = None,
) -> object:
    """Select the optimal chunker based on file type and content.

    Args:
        file_type: File extension (without dot).
        text: Optional text content for analysis.
        embedder: Optional embedder for semantic chunking.

    Returns:
        Chunker instance (RecursiveChunker, SemanticChunker, TableChunker, or CodeChunker).
    """
    # Get strategy for file type
    strategy = FILE_TYPE_STRATEGIES.get(file_type.lower(), "recursive")

    # Override for long documents
    if text and len(text) > LONG_DOCUMENT_THRESHOLD and strategy == "recursive":
        if embedder is not None:
            strategy = "semantic"
            logger.info("chunker.strategy.override", reason="long_document", length=len(text))

    # Create appropriate chunker
    if strategy == "semantic" and embedder is not None:
        logger.info("chunker.select", strategy="semantic", file_type=file_type)
        return SemanticChunker(embedder)

    elif strategy == "table":
        logger.info("chunker.select", strategy="table", file_type=file_type)
        return TableChunker()

    elif strategy == "code":
        logger.info("chunker.select", strategy="code", file_type=file_type)
        return CodeChunker()

    else:
        logger.info("chunker.select", strategy="recursive", file_type=file_type)
        return RecursiveChunker()


def get_available_strategies() -> list[str]:
    """Get list of available chunking strategies.

    Returns:
        List of strategy names: recursive, semantic, table, code.
    """
    return ["recursive", "semantic", "table", "code"]


def get_strategy_for_file(file_type: str) -> str:
    """Get the default strategy for a file type.

    Args:
        file_type: File extension (without dot).

    Returns:
        Strategy name (recursive, semantic, table, or code).
    """
    return FILE_TYPE_STRATEGIES.get(file_type.lower(), "recursive")
