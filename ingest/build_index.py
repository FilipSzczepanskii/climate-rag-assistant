"""Build the searchable knowledge base.

Pipeline: fetch corpus -> chunk -> embed -> write corpus parquet -> build the
Chroma vector store. The parquet is the committed artifact; the Chroma store is
rebuilt from it (locally or at container start) so it never has to be versioned.

Run:
    python -m ingest.build_index
"""

from __future__ import annotations

import pandas as pd

from app.chunking import chunk_document
from app.config import settings
from app.embeddings import Embedder
from app.vectorstore import VectorStore
from ingest.corpus import fetch_corpus


def build() -> None:
    print("1/4  Fetching corpus from Wikipedia ...")
    documents = fetch_corpus()
    if not documents:
        raise SystemExit("No documents fetched. Aborting.")
    print(f"     {len(documents)} documents fetched")

    print("2/4  Chunking documents ...")
    chunks = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc["text"],
                doc_id=doc["doc_id"],
                title=doc["title"],
                source_url=doc["source_url"],
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    print(f"     {len(chunks)} chunks (size~{settings.chunk_size}, overlap {settings.chunk_overlap})")

    print(f"3/4  Embedding chunks with {settings.embedding_model} ...")
    embedder = Embedder(settings.embedding_model)
    vectors = embedder.encode([c.text for c in chunks])

    rows = [
        {
            "chunk_id": f"{c.doc_id}-{c.chunk_index}",
            "doc_id": c.doc_id,
            "title": c.title,
            "source_url": c.source_url,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "embedding": vector,
        }
        for c, vector in zip(chunks, vectors)
    ]
    frame = pd.DataFrame(rows)
    settings.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.corpus_path, index=False)
    print(f"     wrote {len(frame)} rows -> {settings.corpus_path}")

    print("4/4  Building the Chroma vector store ...")
    store = VectorStore(settings.chroma_path, settings.collection_name)
    indexed = store.build_from_parquet(settings.corpus_path)
    print(f"     collection '{settings.collection_name}' indexed {indexed} chunks")
    print("Done.")


if __name__ == "__main__":
    build()
