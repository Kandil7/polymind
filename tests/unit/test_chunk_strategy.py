"""Tests for Document Chunking Strategy Selection."""

from __future__ import annotations

import pytest

from polymind.infrastructure.rag.chunk_strategy import (
    CodeChunker,
    select_chunker,
    get_available_strategies,
    get_strategy_for_file,
    FILE_TYPE_STRATEGIES,
)
from polymind.infrastructure.rag.chunker import (
    RecursiveChunker,
    SemanticChunker,
    TableChunker,
)


class TestChunkerSelection:
    def test_txt_selects_recursive(self) -> None:
        chunker = select_chunker("txt")
        assert isinstance(chunker, RecursiveChunker)

    def test_md_selects_recursive(self) -> None:
        chunker = select_chunker("md")
        assert isinstance(chunker, RecursiveChunker)

    def test_csv_selects_table(self) -> None:
        chunker = select_chunker("csv")
        assert isinstance(chunker, TableChunker)

    def test_tsv_selects_table(self) -> None:
        chunker = select_chunker("tsv")
        assert isinstance(chunker, TableChunker)

    def test_py_selects_code(self) -> None:
        chunker = select_chunker("py")
        assert isinstance(chunker, CodeChunker)

    def test_js_selects_code(self) -> None:
        chunker = select_chunker("js")
        assert isinstance(chunker, CodeChunker)

    def test_pdf_selects_semantic_when_embedder_available(self) -> None:
        from unittest.mock import MagicMock
        embedder = MagicMock()
        chunker = select_chunker("pdf", embedder=embedder)
        assert isinstance(chunker, SemanticChunker)

    def test_pdf_selects_recursive_without_embedder(self) -> None:
        chunker = select_chunker("pdf")
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_type_selects_recursive(self) -> None:
        chunker = select_chunker("xyz")
        assert isinstance(chunker, RecursiveChunker)

    def test_long_text_selects_semantic(self) -> None:
        from unittest.mock import MagicMock
        embedder = MagicMock()
        text = "x" * 6000  # > 5000 threshold
        chunker = select_chunker("txt", text=text, embedder=embedder)
        assert isinstance(chunker, SemanticChunker)

    def test_short_text_stays_recursive(self) -> None:
        from unittest.mock import MagicMock
        embedder = MagicMock()
        text = "Short text"
        chunker = select_chunker("txt", text=text, embedder=embedder)
        assert isinstance(chunker, RecursiveChunker)


class TestCodeChunker:
    def test_chunk_code(self) -> None:
        code = '''
def hello():
    print("hello")

def world():
    print("world")

class MyClass:
    def method(self):
        pass
'''
        chunker = CodeChunker()
        chunks = chunker.chunk(code, metadata={"source": "test.py"})
        assert len(chunks) > 0
        assert all(c["text"].strip() for c in chunks)

    def test_chunk_empty_code(self) -> None:
        chunker = CodeChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_long_code(self) -> None:
        code = "x = 1\n" * 500  # Long code
        chunker = CodeChunker(max_chunk_size=200)
        chunks = chunker.chunk(code)
        assert len(chunks) > 1


class TestStrategyMapping:
    def test_all_strategies_mapped(self) -> None:
        strategies = get_available_strategies()
        assert "recursive" in strategies
        assert "semantic" in strategies
        assert "table" in strategies
        assert "code" in strategies

    def test_get_strategy_for_file(self) -> None:
        assert get_strategy_for_file("txt") == "recursive"
        assert get_strategy_for_file("csv") == "table"
        assert get_strategy_for_file("py") == "code"
        assert get_strategy_for_file("unknown") == "recursive"

    def test_file_type_coverage(self) -> None:
        # Verify common file types are covered
        common_types = ["txt", "md", "py", "js", "csv", "pdf", "html"]
        for ft in common_types:
            assert ft in FILE_TYPE_STRATEGIES, f"Missing strategy for {ft}"
