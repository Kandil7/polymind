"""TableQA (Table Question Answering) specialist — TAPAS wrapper.

Uses google/tapas-base-finetuned-wtq for structured table question answering.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "google/tapas-base-finetuned-wtq"


class TableQAWrapper(ISpecialist):
    """Specialist for Table Question Answering using TAPAS.

    Answers questions about CSV/Excel tables.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize TableQA wrapper with the specified model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._pipe = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "tableqa"

    def _lazy_load(self) -> None:
        """Lazy-load the pipeline to avoid import-time GPU allocation."""
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="table-question-answering",
                model=self._model_id,
            )
            logger.info("tableqa.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("tableqa.model.load_failed", error=str(e))
            self._pipe = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Answer a question about a CSV table.

        Args:
            input_data: Path to the CSV file.
            **kwargs: Must contain 'question' key.

        Returns:
            Dict with keys: answer, cells, aggregator, question.

        Raises:
            RuntimeError: If model is not loaded.
            FileNotFoundError: If CSV file does not exist.
            ValueError: If 'question' kwarg is missing.
        """
        if self._pipe is None:
            raise RuntimeError(
                f"TableQA model '{self._model_id}' failed to load. "
                "Check model availability and GPU memory."
            )

        question = kwargs.get("question")
        if not question:
            raise ValueError("TableQA requires a 'question' keyword argument")

        csv_path = Path(input_data)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info("tableqa.process.start", file=str(csv_path), question=question)

        import pandas as pd

        df = pd.read_csv(csv_path).astype(str)
        result = self._pipe(table=df, query=question)

        output = {
            "answer": result.get("answer", ""),
            "cells": result.get("cells", []),
            "aggregator": result.get("aggregator", "NONE"),
            "question": question,
            "model": self._model_id,
        }

        logger.info(
            "tableqa.process.done",
            answer=output["answer"],
            aggregator=output["aggregator"],
        )

        return output
