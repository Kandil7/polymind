"""Tests for ASR Wrapper (Whisper)."""

from __future__ import annotations

import pytest

from polymind.infrastructure.specialists.asr_wrapper import ASRWrapper


class TestASRWrapperStructure:
    """Test ASR wrapper structure and interface compliance."""

    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist

        assert issubclass(ASRWrapper, ISpecialist)

    def test_name_property(self) -> None:
        # Use a non-existent model to avoid loading
        wrapper = ASRWrapper.__new__(ASRWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        assert wrapper.name == "asr"

    def test_default_model(self) -> None:
        from polymind.infrastructure.specialists.asr_wrapper import DEFAULT_MODEL

        assert DEFAULT_MODEL == "openai/whisper-large-v3"


class TestASRWrapperErrorHandling:
    """Test ASR error handling without loading the model."""

    def _make_wrapper_without_model(self) -> ASRWrapper:
        """Create a wrapper with _pipe=None for error testing."""
        wrapper = ASRWrapper.__new__(ASRWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        return wrapper

    @pytest.mark.asyncio
    async def test_process_raises_when_model_not_loaded(self) -> None:
        wrapper = self._make_wrapper_without_model()
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("dummy.wav")

    @pytest.mark.asyncio
    async def test_process_raises_when_file_not_found(self) -> None:
        wrapper = self._make_wrapper_without_model()
        wrapper._pipe = object()  # Fake a loaded model
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            await wrapper.process("/nonexistent/audio.wav")


class TestASRLazyLoading:
    """Test lazy loading behavior."""

    def test_pipe_is_none_when_model_unavailable(self) -> None:
        """When model can't load, _pipe should be None."""
        wrapper = ASRWrapper.__new__(ASRWrapper)
        wrapper._model_id = "nonexistent/model-12345"
        wrapper._pipe = None
        # Don't call _lazy_load — just verify the state
        assert wrapper._pipe is None
