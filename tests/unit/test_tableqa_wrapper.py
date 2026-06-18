"""Tests for TableQA Wrapper (TAPAS)."""

from __future__ import annotations

import pytest

from polymind.infrastructure.specialists.tableqa_wrapper import TableQAWrapper


class TestTableQAWrapperStructure:
    """Test TableQA wrapper structure and interface compliance."""

    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist

        assert issubclass(TableQAWrapper, ISpecialist)

    def test_name_property(self) -> None:
        wrapper = TableQAWrapper.__new__(TableQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        assert wrapper.name == "tableqa"

    def test_default_model(self) -> None:
        from polymind.infrastructure.specialists.tableqa_wrapper import DEFAULT_MODEL

        assert DEFAULT_MODEL == "google/tapas-base-finetuned-wtq"


class TestTableQAWrapperErrorHandling:
    """Test TableQA error handling without loading the model."""

    def _make_wrapper_without_model(self) -> TableQAWrapper:
        wrapper = TableQAWrapper.__new__(TableQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        return wrapper

    @pytest.mark.asyncio
    async def test_process_raises_when_model_not_loaded(self) -> None:
        wrapper = self._make_wrapper_without_model()
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("dummy.csv", question="How many rows?")

    @pytest.mark.asyncio
    async def test_process_raises_when_file_not_found(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            await wrapper.process("/nonexistent/data.csv", question="What?")

    @pytest.mark.asyncio
    async def test_process_raises_when_no_question(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(ValueError, match="question"):
            await wrapper.process("dummy.csv")

    @pytest.mark.asyncio
    async def test_process_with_mock_csv(self, tmp_path: object) -> None:
        """Test with a real CSV file and mocked pipeline."""
        from pathlib import Path

        import pandas as pd

        csv_path = Path(str(tmp_path)) / "test.csv"
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": ["90", "85"]})
        df.to_csv(csv_path, index=False)

        wrapper = TableQAWrapper.__new__(TableQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = lambda table, query: {
            "answer": "Alice",
            "cells": ["Alice"],
            "aggregator": "MAX",
        }

        result = await wrapper.process(str(csv_path), question="Who has the highest score?")
        assert result["answer"] == "Alice"
        assert result["aggregator"] == "MAX"
