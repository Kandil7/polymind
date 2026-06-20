"""Document ingestion pipeline — file → chunks → embeddings → Qdrant."""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.domain.entities.chunk import ChunkMetadata, DocumentChunk
from polymind.infrastructure.rag.chunk_strategy import select_chunker

logger = structlog.get_logger()


class IngestionPipeline:
    """Orchestrates document ingestion into the vector store.

    Automatically selects optimal chunking strategy based on file type.
    """

    def __init__(self, embedder: object, collection: str = "polymind") -> None:
        """Initialize ingestion pipeline.

        Args:
            embedder: Embedding model instance.
            collection: Qdrant collection name.
        """
        self._embedder = embedder
        self._collection = collection

    async def ingest_file(
        self, file_path: str, source_name: str | None = None
    ) -> list[DocumentChunk]:
        """Ingest a file into the knowledge base.

        Args:
            file_path: Path to the file to ingest.
            source_name: Optional source name override.

        Returns:
            List of indexed DocumentChunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        source = source_name or path.name
        file_type = path.suffix.lower().lstrip(".")

        logger.info("ingestion.start", source=source, type=file_type)

        # Extract text
        text = self._extract_text(path, file_type)

        # Select optimal chunker
        chunker = select_chunker(file_type, text, self._embedder)

        # Chunk the text
        chunks = chunker.chunk(text, metadata={
            "source": source,
            "file_type": file_type,
            "modality": "text",
        })

        # Convert to DocumentChunks
        doc_chunks = []
        for chunk_data in chunks:
            doc_chunk = DocumentChunk(
                text=chunk_data["text"],
                metadata=ChunkMetadata(
                    source=source,
                    file_type=file_type,
                    chunk_index=chunk_data["metadata"]["chunk_index"],
                    modality="text",
                ),
            )
            doc_chunks.append(doc_chunk)

        logger.info(
            "ingestion.chunks.created",
            count=len(doc_chunks),
            strategy=type(chunker).__name__,
        )
        return doc_chunks

    def _extract_text(self, path: Path, file_type: str) -> str:
        """Extract text content from a file based on its type."""
        if file_type in ("txt", "md", "py", "json", "yaml", "yml"):
            return path.read_text(encoding="utf-8")

        if file_type == "pdf":
            return self._extract_pdf(path)

        if file_type == "csv":
            return self._extract_csv(path)

        if file_type in ("html", "xml"):
            return self._extract_markup(path)

        # Fallback: try reading as text
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            import pymupdf

            doc = pymupdf.open(str(path))
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("ingestion.pdf.no_pymupdf")
            raise ImportError(
                "pymupdf is required for PDF text extraction. "
                "Install with: pip install pymupdf"
            ) from None

    @staticmethod
    def _extract_csv(path: Path) -> str:
        """Extract text from a CSV file."""
        import pandas as pd

        df = pd.read_csv(path)
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
            rows.append(row_text)
        return "\n".join(rows)

    @staticmethod
    def _extract_markup(path: Path) -> str:
        """Extract text from HTML/XML files."""
        import re

        content = path.read_text(encoding="utf-8")
        # Simple HTML tag removal
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
