"""Tests for the document chunker."""

from __future__ import annotations

import pytest

from app.chunking import chunk_document


def _chunk(text: str, size: int = 200, overlap: int = 40):
    return chunk_document(
        text,
        doc_id="d1",
        title="Doc One",
        source_url="http://example.com",
        chunk_size=size,
        chunk_overlap=overlap,
    )


def test_empty_text_yields_no_chunks():
    assert _chunk("") == []


def test_short_text_is_a_single_chunk():
    chunks = _chunk("One short paragraph.")
    assert len(chunks) == 1
    assert chunks[0].text == "One short paragraph."
    assert chunks[0].chunk_index == 0


def test_chunks_carry_document_metadata():
    chunks = _chunk("A paragraph.\n\nAnother paragraph.")
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.doc_id == "d1"
        assert chunk.title == "Doc One"
        assert chunk.source_url == "http://example.com"


def test_long_text_splits_into_multiple_indexed_chunks():
    paragraph = "word " * 60
    text = "\n\n".join([paragraph] * 5)
    chunks = _chunk(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_is_hard_split():
    chunks = _chunk("x" * 1000, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= int(200 * 1.5) for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        _chunk("some text", size=100, overlap=100)
