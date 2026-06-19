"""Tests for VQA Wrapper (BLIP)."""

from __future__ import annotations

import pytest

from polymind.infrastructure.specialists.vqa_wrapper import VQAWrapper


class TestVQAWrapperStructure:
    """Test VQA wrapper structure and interface compliance."""

    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist

        assert issubclass(VQAWrapper, ISpecialist)

    def test_name_property(self) -> None:
        wrapper = VQAWrapper.__new__(VQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        assert wrapper.name == "vqa"

    def test_default_model(self) -> None:
        from polymind.infrastructure.specialists.vqa_wrapper import DEFAULT_MODEL

        assert DEFAULT_MODEL == "Salesforce/blip-vqa-base"


class TestVQAWrapperErrorHandling:
    """Test VQA error handling without loading the model."""

    def _make_wrapper_without_model(self) -> VQAWrapper:
        wrapper = VQAWrapper.__new__(VQAWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        return wrapper

    @pytest.mark.asyncio
    async def test_process_raises_when_model_not_loaded(self) -> None:
        wrapper = self._make_wrapper_without_model()
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("dummy.jpg", question="What is this?")

    @pytest.mark.asyncio
    async def test_process_raises_when_file_not_found(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            await wrapper.process("/nonexistent/image.jpg", question="What?")

    @pytest.mark.asyncio
    async def test_process_raises_when_no_question(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(ValueError, match="question"):
            await wrapper.process("dummy.jpg")

    @pytest.mark.asyncio
    async def test_process_raises_when_empty_question(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()
        with pytest.raises(ValueError, match="question"):
            await wrapper.process("dummy.jpg", question="")
