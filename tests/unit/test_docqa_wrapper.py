"""Tests for DocQA Wrapper (LayoutLM)."""

from __future__ import annotations

import pytest

from polymind.infrastructure.specialists.docqa_wrapper import DocQAWrapper


class TestDocQAWrapperStructure:
    """Test DocQA wrapper structure and interface compliance."""

    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist

        assert issubclass(DocQAWrapper, ISpecialist)

    def test_name_property(self) -> None:
        wrapper = DocQAWrapper.__new__(DocQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        assert wrapper.name == "docqa"

    def test_default_model(self) -> None:
        from polymind.infrastructure.specialists.docqa_wrapper import DEFAULT_MODEL

        assert DEFAULT_MODEL == "impira/layoutlm-document-qa"


class TestDocQAWrapperErrorHandling:
    """Test DocQA error handling without loading the model."""

    def _make_wrapper_without_model(self) -> DocQAWrapper:
        wrapper = DocQAWrapper.__new__(DocQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        return wrapper

    @pytest.mark.asyncio
    async def test_process_raises_when_model_not_loaded(self) -> None:
        wrapper = self._make_wrapper_without_model()
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("dummy.pdf", question="What is this?")

    @pytest.mark.asyncio
    async def test_process_raises_when_file_not_found(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(FileNotFoundError, match="Document file not found"):
            await wrapper.process("/nonexistent/doc.pdf", question="What?")

    @pytest.mark.asyncio
    async def test_process_raises_when_no_question(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(ValueError, match="question"):
            await wrapper.process("dummy.pdf")

    @pytest.mark.asyncio
    async def test_process_returns_empty_when_no_results(self, tmp_path: object) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = lambda *a, **kw: []

        from pathlib import Path

        dummy_file = Path(str(tmp_path)) / "dummy.pdf"
        dummy_file.write_text("fake")

        result = await wrapper.process(str(dummy_file), question="What?")
        assert result["answer"] == ""
        assert result["score"] == 0.0
