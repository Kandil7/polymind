"""Summarizer specialist — text summarization wrapper.

Uses a HuggingFace summarization model for long-context compression.
"""

from __future__ import annotations

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "facebook/bart-large-cnn"


class SummarizerWrapper(ISpecialist):
    """Specialist for text summarization using BART.

    Compresses long texts into concise summaries.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize Summarizer wrapper with the specified model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._pipe = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "summarizer"

    def _lazy_load(self) -> None:
        """Lazy-load the pipeline to avoid import-time GPU allocation."""
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="summarization",
                model=self._model_id,
            )
            logger.info("summarizer.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("summarizer.model.load_failed", error=str(e))
            self._pipe = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Summarize the provided text.

        Args:
            input_data: The text to summarize.
            **kwargs: Optional 'max_length' (int) and 'min_length' (int).

        Returns:
            Dict with keys: summary, original_length, summary_length.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._pipe is None:
            raise RuntimeError(
                f"Summarizer model '{self._model_id}' failed to load. "
                "Check model availability and GPU memory."
            )

        max_length = kwargs.get("max_length", 150)
        min_length = kwargs.get("min_length", 40)

        logger.info(
            "summarizer.process.start",
            text_length=len(input_data),
            max_length=max_length,
        )

        result = self._pipe(
            input_data,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        summary = result[0]["summary_text"].strip()

        output = {
            "summary": summary,
            "original_length": len(input_data),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / max(len(input_data), 1),
            "model": self._model_id,
        }

        logger.info(
            "summarizer.process.done",
            summary_length=output["summary_length"],
            compression_ratio=output["compression_ratio"],
        )

        return output
