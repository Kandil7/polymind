"""Tests for Embedder."""

from __future__ import annotations

import pytest

from polymind.infrastructure.rag.embedder import Embedder


class TestEmbedderStructure:
    def test_default_model(self) -> None:
        from polymind.infrastructure.rag.embedder import DEFAULT_MODEL
        assert DEFAULT_MODEL == "BAAI/bge-m3"

    def test_dimension_without_model(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        assert embedder.dimension == 1024  # Default

    def test_embed_raises_without_model(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        with pytest.raises(RuntimeError, match="failed to load"):
            embedder.embed(["hello"])

    def test_embed_single_raises_without_model(self) -> None:
        embedder = Embedder.__new__(Embedder)
        embedder._model = None
        embedder._model_id = "test"
        with pytest.raises(RuntimeError, match="failed to load"):
            embedder.embed_single("hello")
