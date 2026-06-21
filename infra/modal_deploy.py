"""Modal GPU deployment — on-demand inference for heavy models.

Deploys ASR, VQA, and Embedding models to Modal's serverless GPU infrastructure.

Usage:
    # Deploy all functions
    modal deploy infra/modal_deploy.py

    # Deploy specific function
    modal deploy infra/modal_deploy.py --function run_asr

    # Run locally with Modal
    modal run infra/modal_deploy.py::run_asr --audio-bytes-file test.wav
"""

from __future__ import annotations

import modal

app = modal.App("polymind")

# ── Docker Image ──────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "fastapi",
        "groq",
        "langchain-openai",
        "qdrant-client",
        "sentence-transformers",
        "Pillow",
        "librosa",
        "soundfile",
    )
)


# ── ASR Function ─────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_asr(audio_bytes: bytes) -> dict:
    """Run ASR inference on GPU via Modal.

    Accepts audio file bytes, returns transcription.
    """
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        from transformers import pipeline

        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=0,
        )

        result = pipe(temp_path)
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "duration": result.get("chunks", [{}])[-1].get("end", 0),
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ── VQA Function ──────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_vqa(image_bytes: bytes, question: str) -> dict:
    """Run VQA inference on GPU via Modal.

    Accepts image bytes and question, returns answer.
    """
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    from transformers import pipeline

    pipe = pipeline(
        task="visual-question-answering",
        model="Salesforce/blip-vqa-base",
        device=0,
    )

    results = pipe(image, question, top_k=3)
    return {
        "answer": results[0]["answer"],
        "confidence": float(results[0]["score"]),
        "candidates": [
            {"answer": r["answer"], "score": float(r["score"])}
            for r in results
        ],
    }


# ── Embedding Function ────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_embedding(texts: list[str]) -> list[list[float]]:
    """Run embedding inference on GPU via Modal.

    Accepts list of texts, returns embedding vectors.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ── CLIP Function ─────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_clip_image(image_bytes: bytes) -> list[float]:
    """Run CLIP image embedding on GPU via Modal.

    Accepts image bytes, returns 512-dim embedding.
    """
    import io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    embedding = features[0].cpu().numpy()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    return embedding.tolist()
