"""Split documents into overlapping chunks for embedding.

Deliberately framework-free. The strategy is paragraph packing: paragraphs are
greedily merged until they approach the target size, with a character overlap
carried between chunks so context is not lost at boundaries. A single oversized
paragraph is hard-split as a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    text: str
    doc_id: str
    title: str
    source_url: str
    chunk_index: int


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def chunk_document(
    text: str,
    *,
    doc_id: str,
    title: str,
    source_url: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Return overlapping chunks for one document."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    raw_chunks: list[str] = []
    current = ""

    for para in _paragraphs(text):
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}"
        else:
            raw_chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

        # Hard-split a paragraph that is far larger than the target on its own.
        while len(current) > chunk_size * 1.5:
            raw_chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap:]

    if current.strip():
        raw_chunks.append(current)

    return [
        Chunk(
            text=c.strip(),
            doc_id=doc_id,
            title=title,
            source_url=source_url,
            chunk_index=i,
        )
        for i, c in enumerate(raw_chunks)
        if c.strip()
    ]
