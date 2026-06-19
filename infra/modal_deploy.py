"""Modal GPU deployment — on-demand inference for heavy models."""

from __future__ import annotations

import modal

app = modal.App("polymind")

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
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_asr(audio_bytes: bytes) -> dict:
    """Run ASR inference on GPU via Modal."""
    import tempfile
    from pathlib import Path

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        from transformers import pipeline

        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=0,  # GPU
        )

        result = pipe(temp_path)
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=60,
)
@modal.web_endpoint(method="POST")
def run_vqa(image_bytes: bytes, question: str) -> dict:
    """Run VQA inference on GPU via Modal."""
    import tempfile
    from pathlib import Path
    from PIL import Image
    import io

    # Load image
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
