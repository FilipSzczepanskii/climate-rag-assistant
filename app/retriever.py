"""Semantic retrieval with optional cross-encoder reranking.

The base retrieval is a cosine similarity search over the embedded corpus. When
reranking is enabled, a wider candidate set is pulled first and a cross-encoder
rescores each (query, passage) pair directly, which is more accurate than the
bi-encoder similarity but too slow to run over the whole corpus.
"""

from __future__ import annotations

from functools import lru_cache

from app.embeddings import Embedder
from app.vectorstore import Hit, VectorStore


@lru_cache(maxsize=1)
def _load_cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


class Retriever:
    """Retrieves the most relevant corpus chunks for a query."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        top_k: int,
        use_reranker: bool = False,
        reranker_model: str = "",
        rerank_candidates: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.rerank_candidates = rerank_candidates

    def retrieve(self, query: str) -> list[Hit]:
        query_vector = self.embedder.encode_one(query)
        n_candidates = self.rerank_candidates if self.use_reranker else self.top_k
        hits = self.store.query(query_vector, top_k=n_candidates)

        if self.use_reranker and hits:
            hits = self._rerank(query, hits)

        return hits[: self.top_k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        encoder = _load_cross_encoder(self.reranker_model)
        scores = encoder.predict([(query, hit.text) for hit in hits])
        ranked = sorted(
            zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            Hit(
                text=hit.text,
                title=hit.title,
                source_url=hit.source_url,
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
                score=float(score),
            )
            for hit, score in ranked
        ]
