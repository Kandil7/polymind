"""Groq ASR — speech-to-text via Groq's Whisper API.

Replaces local Whisper model with Groq's hosted Whisper for faster inference.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from polymind.domain.interfaces.specialist import ISpecialist

logger = structlog.get_logger()

DEFAULT_MODEL = "whisper-large-v3"
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqASRWrapper(ISpecialist):
    """ASR specialist using Groq's hosted Whisper API.

    Advantages over local Whisper:
    - No GPU required
    - Ultra-fast inference (~280 t/s)
    - Automatic language detection
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize Groq ASR wrapper.

        Args:
            model_id: Groq model identifier for Whisper.
        """
        self._model_id = model_id
        self._client = None
        self._lazy_load()

    @property
    def name(self) -> str:
        """Return the specialist name."""
        return "groq_asr"

    def _lazy_load(self) -> None:
        """Lazy-load the Groq client."""
        try:
            import groq

            api_key = os.getenv("GROQ_API_KEY", "")
            self._client = groq.Groq(api_key=api_key)
            logger.info("groq_asr.client.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("groq_asr.client.load_failed", error=str(e))
            self._client = None

    async def process(
        self, input_data: str, **kwargs: object
    ) -> dict[str, object]:
        """Transcribe an audio file using Groq's Whisper API.

        Args:
            input_data: Path to the audio file.

        Returns:
            Dict with keys: text, language, duration_s.

        Raises:
            RuntimeError: If client is not loaded.
            FileNotFoundError: If audio file does not exist.
        """
        if self._client is None:
            raise RuntimeError(
                "Groq ASR client failed to load. "
                "Check GROQ_API_KEY environment variable."
            )

        audio_path = Path(input_data)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("groq_asr.process.start", file=str(audio_path))

        with open(audio_path, "rb") as audio_file:
            transcription = self._client.audio.transcriptions.create(
                file=(str(audio_path), audio_file),
                model=self._model_id,
                response_format="verbose_json",
            )

        output = {
            "text": transcription.text.strip(),
            "language": getattr(transcription, "language", "unknown"),
            "duration_s": getattr(transcription, "duration", 0.0),
            "model": self._model_id,
            "provider": "groq",
        }

        logger.info(
            "groq_asr.process.done",
            text_length=len(output["text"]),
            duration=output["duration_s"],
        )

        return output
