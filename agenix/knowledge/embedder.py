"""Sentence-transformers embedding wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

from agenix.config import EmbedderConfig


class Embedder:
    """Wraps a sentence-transformers model for text embedding.

    Lazily loads the model on first use to avoid slow import at startup.
    """

    def __init__(self, config: Optional[EmbedderConfig] = None) -> None:
        self.config = config or EmbedderConfig()
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, D) float32 array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, returning a (D,) float32 array."""
        return self.embed([text])[0]
