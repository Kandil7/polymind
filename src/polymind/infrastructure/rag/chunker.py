"""Document chunking strategies — recursive, semantic, and table-aware."""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class RecursiveChunker:
    """Recursive text chunker — splits by paragraphs, then sentences, then chars."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Split text into overlapping chunks.

        Args:
            text: Raw text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with 'text' and 'metadata' keys.
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        idx = 0
        prev_end = -1

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Prevent infinite loop: if we haven't advanced, force break
            if end == prev_end:
                break
            prev_end = end

            # Try to break at a sentence boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": idx,
                    },
                })
                idx += 1

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        logger.info("chunker.recursive.done", chunks=len(chunks))
        return chunks


class SemanticChunker:
    """Semantic chunker — groups sentences by semantic similarity.

    Uses the embedder to compute sentence embeddings and groups
    semantically similar consecutive sentences into chunks.
    """

    def __init__(self, embedder: object, threshold: float = 0.5) -> None:
        """Initialize semantic chunker.

        Args:
            embedder: Embedder instance for computing embeddings.
            threshold: Cosine similarity threshold for grouping.
        """
        self._embedder = embedder
        self._threshold = threshold

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Split text into semantically coherent chunks.

        Args:
            text: Raw text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with 'text' and 'metadata' keys.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return [{"text": text, "metadata": metadata or {}}]

        embeddings = self._embedder.embed(sentences)

        chunks = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim >= self._threshold:
                current_group.append(sentences[i])
            else:
                chunks.append({
                    "text": " ".join(current_group),
                    "metadata": {**(metadata or {}), "chunk_index": len(chunks)},
                })
                current_group = [sentences[i]]

        if current_group:
            chunks.append({
                "text": " ".join(current_group),
                "metadata": {**(metadata or {}), "chunk_index": len(chunks)},
            })

        logger.info("chunker.semantic.done", chunks=len(chunks))
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (
            np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8
        ))


class TableChunker:
    """Table-aware chunker — preserves table structure as JSON rows."""

    def chunk(self, csv_path: str, metadata: dict | None = None) -> list[dict]:
        """Parse a CSV file and return structured table chunks.

        Args:
            csv_path: Path to the CSV file.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with 'text' and 'metadata' keys.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        chunks = []

        for idx, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
            chunks.append({
                "text": row_text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": idx,
                    "modality": "table",
                },
            })

        logger.info("chunker.table.done", rows=len(chunks))
        return chunks
