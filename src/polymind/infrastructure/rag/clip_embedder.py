"""CLIP Embedder — unified embedding space for text and images.

Uses OpenAI's CLIP model to embed both text and images into the same
vector space, enabling cross-modal search (text→image, image→text).

Based on: https://arxiv.org/abs/2103.00020

Usage:
    from polymind.infrastructure.rag.clip_embedder import CLIPEmbedder

    embedder = CLIPEmbedder()
    text_vec = embedder.embed_text("a photo of a cat")
    image_vec = embedder.embed_image("photo.jpg")
    similarity = embedder.compute_similarity(text_vec, image_vec)
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_DIMENSION = 512


class CLIPEmbedder:
    """CLIP-based embedder for cross-modal text-image search.

    Embeds text and images into a shared 512-dimensional space.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize CLIP embedder.

        Args:
            model_id: HuggingFace CLIP model identifier.
        """
        self._model_id = model_id
        self._processor = None
        self._model = None
        self._lazy_load()

    def _lazy_load(self) -> None:
        """Lazy-load CLIP model and processor."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self._model_id)
            self._model = CLIPModel.from_pretrained(self._model_id)
            logger.info("clip.model.loaded", model=self._model_id)
        except Exception as e:
            logger.warning("clip.model.load_failed", error=str(e))
            self._processor = None
            self._model = None

    @property
    def is_available(self) -> bool:
        """Check if CLIP model is loaded."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._model is not None:
            return self._model.config.projection_dim
        return DEFAULT_DIMENSION

    def embed_text(self, text: str) -> list[float]:
        """Embed text into the shared CLIP space.

        Args:
            text: Text to embed.

        Returns:
            512-dimensional embedding vector.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("CLIP model not loaded")

        import torch

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        # Normalize
        embedding = features[0].cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.tolist()

    def embed_image(self, image_path: str) -> list[float]:
        """Embed an image into the shared CLIP space.

        Args:
            image_path: Path to the image file.

        Returns:
            512-dimensional embedding vector.

        Raises:
            RuntimeError: If model is not loaded.
            FileNotFoundError: If image file not found.
        """
        if self._model is None:
            raise RuntimeError("CLIP model not loaded")

        import torch
        from pathlib import Path
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")

        inputs = self._processor(
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        # Normalize
        embedding = features[0].cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed_text(text) for text in texts]

    def compute_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity score (-1 to 1).
        """
        import numpy as np

        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def text_image_similarity(
        self, text: str, image_path: str
    ) -> float:
        """Compute similarity between text and image.

        Args:
            text: Text query.
            image_path: Path to image file.

        Returns:
            Similarity score.
        """
        text_vec = self.embed_text(text)
        image_vec = self.embed_image(image_path)
        return self.compute_similarity(text_vec, image_vec)
