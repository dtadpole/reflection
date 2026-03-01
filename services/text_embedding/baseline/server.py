"""FastAPI server for text embedding.

Run with: uvicorn services.text_embedding.baseline.server:app --host 0.0.0.0 --port 42982
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from services.models import EmbeddingResult, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="text-embedding", version="1.0.0")

# Server state
_model_name: str = "Qwen/Qwen3-Embedding-8B"
_dimension: int = 4096
_max_batch_size: int = 64
_max_seq_length: int = 8192
_device: str = "cuda:0"
_port: int = 42982
_model = None  # lazy-loaded SentenceTransformer


class EmbedRequest(BaseModel):
    texts: list[str]
    instruction: str = ""


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        logger.info("Loading model %s on %s ...", _model_name, _device)
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            _model_name,
            device=_device,
            truncate_dim=_dimension,
        )
        _model.max_seq_length = _max_seq_length
        logger.info("Model loaded: %s (dim=%d)", _model_name, _dimension)
    return _model


@app.post("/embed", response_model=EmbeddingResult)
async def embed(req: EmbedRequest) -> EmbeddingResult:
    """Embed a batch of texts."""
    model = _get_model()

    # Process in batches
    all_embeddings: list[list[float]] = []
    for i in range(0, len(req.texts), _max_batch_size):
        batch = req.texts[i : i + _max_batch_size]
        if req.instruction:
            vectors = model.encode(batch, prompt=req.instruction)
        else:
            vectors = model.encode(batch)
        all_embeddings.extend(v.tolist() for v in vectors)

    return EmbeddingResult(
        embeddings=all_embeddings,
        model=_model_name,
        dimension=_dimension,
    )


@app.get("/health", response_model=ServiceHealth)
async def health() -> ServiceHealth:
    """Health check endpoint."""
    return ServiceHealth(
        name="text_embedding",
        status=ServiceStatus.RUNNING,
        endpoint=f"http://0.0.0.0:{_port}",
        devices=[_device],
    )


def configure(
    model_name: str = "Qwen/Qwen3-Embedding-8B",
    dimension: int = 4096,
    max_batch_size: int = 64,
    max_seq_length: int = 8192,
    device: str = "cuda:0",
    port: int = 42982,
) -> None:
    """Configure server settings (call before starting uvicorn)."""
    global _model_name, _dimension, _max_batch_size, _max_seq_length, _device, _port
    _model_name = model_name
    _dimension = dimension
    _max_batch_size = max_batch_size
    _max_seq_length = max_seq_length
    _device = device
    _port = port
