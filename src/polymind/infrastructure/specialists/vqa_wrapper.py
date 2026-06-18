"""VQA (Visual Question Answering) specialist — BLIP wrapper.

Uses Salesforce/blip-vqa-base for image-based question answering.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "Salesforce/blip-vqa-base"


class VQAWrapper(ISpecialist):
    """Specialist for Visual Question Answering using BLIP.

    Answers questions about images with confidence scores.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize VQA wrapper with the specified model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._pipe = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "vqa"

    def _lazy_load(self) -> None:
        """Lazy-load the pipeline to avoid import-time GPU allocation."""
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="visual-question-answering",
                model=self._model_id,
            )
            logger.info("vqa.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("vqa.model.load_failed", error=str(e))
            self._pipe = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Answer a question about an image.

        Args:
            input_data: Path to the image file.
            **kwargs: Must contain 'question' key.

        Returns:
            Dict with keys: answer, confidence, candidates.

        Raises:
            RuntimeError: If model is not loaded.
            FileNotFoundError: If image file does not exist.
            ValueError: If 'question' kwarg is missing.
        """
        if self._pipe is None:
            raise RuntimeError(
                f"VQA model '{self._model_id}' failed to load. "
                "Check model availability and GPU memory."
            )

        question = kwargs.get("question")
        if not question:
            raise ValueError("VQA requires a 'question' keyword argument")

        image_path = Path(input_data)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info("vqa.process.start", file=str(image_path), question=question)

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        results = self._pipe(image, question, top_k=3)

        output = {
            "answer": results[0]["answer"],
            "confidence": float(results[0]["score"]),
            "candidates": [
                {"answer": r["answer"], "score": float(r["score"])}
                for r in results
            ],
            "question": question,
            "model": self._model_id,
        }

        logger.info(
            "vqa.process.done",
            answer=output["answer"],
            confidence=output["confidence"],
        )

        return output
