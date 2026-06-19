"""Tests for IngestUseCase."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymind.application.use_cases.ingest_use_case import (
    IngestResult,
    IngestUseCase,
)
from polymind.domain.exceptions import IngestionError


class TestIngestResult:
    def test_ingest_result_defaults(self) -> None:
        result = IngestResult(
            status="success",
            chunks_created=5,
            source="test.txt",
            collection="polymind",
        )
        assert result.status == "success"
        assert result.chunks_created == 5
        assert result.errors == []

    def test_ingest_result_with_errors(self) -> None:
        result = IngestResult(
            status="partial",
            chunks_created=2,
            source="test.txt",
            collection="polymind",
            errors=["chunk 3 failed"],
        )
        assert len(result.errors) == 1


class TestIngestUseCase:
    def test_execute_raises_for_missing_file(self) -> None:
        use_case = IngestUseCase()
        with pytest.raises(IngestionError, match="File not found"):
            import asyncio

            asyncio.run(use_case.execute("/nonexistent/file.txt"))

    def test_init_stores_config(self) -> None:
        use_case = IngestUseCase(
            collection="test_collection",
            qdrant_url="http://custom:6333",
        )
        assert use_case._collection == "test_collection"
        assert use_case._qdrant_url == "http://custom:6333"

    def test_init_defaults(self) -> None:
        use_case = IngestUseCase()
        assert use_case._collection == "polymind"
        assert use_case._qdrant_url == "http://localhost:6333"
