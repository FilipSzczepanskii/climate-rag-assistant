"""Integration smoke test for retrieval over the built index.

Builds the Chroma store from the committed corpus parquet and checks that a
known question retrieves a chunk from the expected source article. Skipped if
the corpus parquet has not been built yet.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.embeddings import Embedder
from app.vectorstore import VectorStore


@pytest.fixture(scope="module")
def store() -> VectorStore:
    if not settings.corpus_path.exists():
        pytest.skip("corpus parquet not built; run python -m ingest.build_index")
    vector_store = VectorStore(settings.chroma_path, settings.collection_name)
    if vector_store.count() == 0:
        vector_store.build_from_parquet(settings.corpus_path)
    return vector_store


def test_index_is_populated(store: VectorStore):
    assert store.count() > 100


def test_retrieval_returns_the_expected_source(store: VectorStore):
    embedder = Embedder(settings.embedding_model)
    query_vector = embedder.encode_one("What causes acid rain?")
    hits = store.query(query_vector, top_k=4)

    assert hits, "expected at least one retrieved chunk"
    assert "Acid rain" in {hit.title for hit in hits}
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
