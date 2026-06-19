"""Tests for chunking strategies."""

from __future__ import annotations

from polymind.infrastructure.rag.chunker import RecursiveChunker, TableChunker


class TestRecursiveChunker:
    def test_chunk_simple_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("Hello world. This is a test.")
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_chunk_empty_text(self) -> None:
        chunker = RecursiveChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_preserves_metadata(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        meta = {"source": "test.txt", "file_type": "txt"}
        chunks = chunker.chunk("Hello world", metadata=meta)
        assert chunks[0]["metadata"]["source"] == "test.txt"

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunker.chunk(text)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_overlap(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=20)
        text = "A" * 100
        chunks = chunker.chunk(text)
        # With overlap, we should get more chunks than without
        chunker_no_overlap = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks_no_overlap = chunker_no_overlap.chunk(text)
        assert len(chunks) >= len(chunks_no_overlap)


class TestTableChunker:
    def test_chunk_csv(self, tmp_path: object) -> None:
        from pathlib import Path

        import pandas as pd

        csv_path = Path(str(tmp_path)) / "test.csv"
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        df.to_csv(csv_path, index=False)

        chunker = TableChunker()
        chunks = chunker.chunk(str(csv_path))
        assert len(chunks) == 2
        assert chunks[0]["metadata"]["modality"] == "table"
        assert "Alice" in chunks[0]["text"]
