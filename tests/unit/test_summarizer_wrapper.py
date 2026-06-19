"""Tests for Summarizer Wrapper (BART)."""

from __future__ import annotations

import pytest

from polymind.infrastructure.specialists.summarizer_wrapper import SummarizerWrapper


class TestSummarizerWrapperStructure:
    """Test Summarizer wrapper structure and interface compliance."""

    def test_implements_specialist_interface(self) -> None:
        from polymind.domain.interfaces.specialist import ISpecialist

        assert issubclass(SummarizerWrapper, ISpecialist)

    def test_name_property(self) -> None:
        wrapper = SummarizerWrapper.__new__(SummarizerWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        assert wrapper.name == "summarizer"

    def test_default_model(self) -> None:
        from polymind.infrastructure.specialists.summarizer_wrapper import DEFAULT_MODEL

        assert DEFAULT_MODEL == "facebook/bart-large-cnn"


class TestSummarizerWrapperErrorHandling:
    """Test Summarizer error handling without loading the model."""

    def _make_wrapper_without_model(self) -> SummarizerWrapper:
        wrapper = SummarizerWrapper.__new__(SummarizerWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = None
        return wrapper

    @pytest.mark.asyncio
    async def test_process_raises_when_model_not_loaded(self) -> None:
        wrapper = self._make_wrapper_without_model()
        with pytest.raises(RuntimeError, match="failed to load"):
            await wrapper.process("Some long text to summarize.")

    @pytest.mark.asyncio
    async def test_process_with_mock_pipeline(self) -> None:
        """Test with a mocked pipeline."""
        wrapper = SummarizerWrapper.__new__(SummarizerWrapper)
        wrapper._model_id = "test/model"
        wrapper._pipe = lambda text, **kw: [{"summary_text": "Short summary."}]

        result = await wrapper.process(
            "This is a very long document that needs to be summarized "
            "into a concise form for the user to read quickly.",
            max_length=50,
            min_length=10,
        )

        assert result["summary"] == "Short summary."
        assert result["original_length"] > 0
        assert result["summary_length"] > 0
        assert 0 < result["compression_ratio"] < 1

    @pytest.mark.asyncio
    async def test_process_default_params(self) -> None:
        """Test that default max/min length are applied."""
        wrapper = SummarizerWrapper.__new__(SummarizerWrapper)
        wrapper._model_id = "test/model"

        captured_kw = {}

        def mock_pipe(text: str, **kw: object) -> list[dict[str, str]]:
            captured_kw.update(kw)
            return [{"summary_text": "Summary"}]

        wrapper._pipe = mock_pipe
        await wrapper.process("Some text")

        assert captured_kw["max_length"] == 150
        assert captured_kw["min_length"] == 40
