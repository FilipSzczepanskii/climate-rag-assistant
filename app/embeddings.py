"""Sentence-transformer embedding wrapper.

The model is cached per process so repeated requests in the API do not reload
weights. Embeddings are L2-normalized, which lets cosine similarity be computed
as a plain dot product downstream.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


class Embedder:
    """Encodes text into dense vectors with a sentence-transformer model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    def encode(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
