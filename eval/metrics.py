"""Evaluation metrics for retrieval quality and answer quality.

Retrieval metrics (hit rate, MRR) need no LLM and run in CI. Answer metrics
(keyword recall, citation match) score a generated answer and are used in the
local full evaluation.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[([^\[\]]+)\]")


def hit_rate(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    """1.0 if any expected source appears among the retrieved chunks."""
    return 1.0 if set(retrieved_titles) & set(expected_titles) else 0.0


def reciprocal_rank(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    """Reciprocal of the rank of the first relevant retrieved chunk, else 0."""
    expected = set(expected_titles)
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in expected:
            return 1.0 / rank
    return 0.0


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    if not keywords:
        return 1.0
    text = answer.lower()
    found = sum(1 for keyword in keywords if keyword.lower() in text)
    return found / len(keywords)


def citation_match(answer: str, expected_titles: list[str]) -> float:
    """1.0 if the answer cites at least one expected source in square brackets."""
    cited = {c.strip() for c in _CITATION.findall(answer)}
    return 1.0 if cited & set(expected_titles) else 0.0
