"""Chroma vector store wrapper.

The store is rebuilt from a committed corpus parquet that already holds the
precomputed embeddings. That keeps the deployed artifact small and version
stable: no embedding model has to run at deploy time just to populate the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import pandas as pd


@dataclass
class Hit:
    text: str
    title: str
    source_url: str
    doc_id: str
    chunk_index: int
    score: float


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, path: Path, collection_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection_name = collection_name

    def _collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return self._collection().count()
        except Exception:
            return 0

    def build_from_parquet(self, parquet_path: Path) -> int:
        """Drop and recreate the collection from a corpus parquet."""
        df = pd.read_parquet(parquet_path)
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

        collection = self._collection()
        collection.add(
            ids=df["chunk_id"].astype(str).tolist(),
            embeddings=[list(v) for v in df["embedding"]],
            documents=df["text"].tolist(),
            metadatas=[
                {
                    "title": title,
                    "source_url": url,
                    "doc_id": str(doc_id),
                    "chunk_index": int(idx),
                }
                for title, url, doc_id, idx in zip(
                    df["title"], df["source_url"], df["doc_id"], df["chunk_index"]
                )
            ],
        )
        return collection.count()

    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        result = self._collection().query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        hits: list[Hit] = []
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            hits.append(
                Hit(
                    text=doc,
                    title=meta["title"],
                    source_url=meta["source_url"],
                    doc_id=meta["doc_id"],
                    chunk_index=int(meta["chunk_index"]),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return hits
