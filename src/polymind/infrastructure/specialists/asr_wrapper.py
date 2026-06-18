"""ASR (Automatic Speech Recognition) specialist — Whisper wrapper.

Uses openai/whisper-large-v3 for speech-to-text transcription.
Supports English and Arabic with automatic language detection.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "openai/whisper-large-v3"


class ASRWrapper(ISpecialist):
    """Specialist for Automatic Speech Recognition using Whisper.

    Transcribes audio files to text with timestamp support.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize ASR wrapper with the specified model.

        Args:
            model_id: HuggingFace model identifier.
        """
        self._model_id = model_id
        self._pipe = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "asr"

    def _lazy_load(self) -> None:
        """Lazy-load the pipeline to avoid import-time GPU allocation."""
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=self._model_id,
                chunk_length_s=30,
                return_timestamps=True,
            )
            logger.info("asr.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("asr.model.load_failed", error=str(e))
            self._pipe = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Transcribe an audio file to text.

        Args:
            input_data: Path to the audio file.

        Returns:
            Dict with keys: text, chunks, duration_s, language.

        Raises:
            RuntimeError: If model is not loaded.
            FileNotFoundError: If audio file does not exist.
        """
        if self._pipe is None:
            raise RuntimeError(
                f"ASR model '{self._model_id}' failed to load. "
                "Check model availability and GPU memory."
            )

        audio_path = Path(input_data)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("asr.process.start", file=str(audio_path))

        result = self._pipe(str(audio_path))

        output = {
            "text": result.get("text", "").strip(),
            "chunks": result.get("chunks", []),
            "duration_s": self._estimate_duration(result),
            "language": result.get("language", "unknown"),
            "model": self._model_id,
        }

        logger.info(
            "asr.process.done",
            text_length=len(output["text"]),
            duration=output["duration_s"],
        )

        return output

    @staticmethod
    def _estimate_duration(result: dict) -> float:
        """Estimate audio duration from timestamp chunks."""
        chunks = result.get("chunks", [])
        if not chunks:
            return 0.0

        last_chunk = chunks[-1]
        return last_chunk.get("end", 0.0)
