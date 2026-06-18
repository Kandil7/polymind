"""DocQA (Document Question Answering) specialist — LayoutLM wrapper.

Uses impira/layoutlm-document-qa for PDF/scanned document question answering.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "impira/layoutlm-document-qa"


class DocQAWrapper(ISpecialist):
    """Specialist for Document Question Answering using LayoutLM.

    Answers questions about document images (scanned docs, PDFs).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize DocQA wrapper with the specified model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._pipe = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "docqa"

    def _lazy_load(self) -> None:
        """Lazy-load the pipeline to avoid import-time GPU allocation."""
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="document-question-answering",
                model=self._model_id,
            )
            logger.info("docqa.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("docqa.model.load_failed", error=str(e))
            self._pipe = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Answer a question about a document image.

        Args:
            input_data: Path to the document image (PDF page, scan, etc.).
            **kwargs: Must contain 'question' key.

        Returns:
            Dict with keys: answer, score, start, end.

        Raises:
            RuntimeError: If model is not loaded.
            FileNotFoundError: If document file does not exist.
            ValueError: If 'question' kwarg is missing.
        """
        if self._pipe is None:
            raise RuntimeError(
                f"DocQA model '{self._model_id}' failed to load. "
                "Check model availability and GPU memory."
            )

        question = kwargs.get("question")
        if not question:
            raise ValueError("DocQA requires a 'question' keyword argument")

        doc_path = Path(input_data)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document file not found: {doc_path}")

        logger.info("docqa.process.start", file=str(doc_path), question=question)

        results = self._pipe(str(doc_path), question)

        if not results:
            return {
                "answer": "",
                "score": 0.0,
                "start": None,
                "end": None,
                "question": question,
                "model": self._model_id,
            }

        best = results[0]
        output = {
            "answer": best["answer"],
            "score": float(best["score"]),
            "start": best.get("start"),
            "end": best.get("end"),
            "question": question,
            "model": self._model_id,
        }

        logger.info(
            "docqa.process.done",
            answer=output["answer"],
            score=output["score"],
        )

        return output
