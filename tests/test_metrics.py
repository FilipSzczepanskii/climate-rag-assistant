"""Tests for the evaluation metrics."""

from __future__ import annotations

from eval.metrics import citation_match, hit_rate, keyword_recall, reciprocal_rank


def test_hit_rate_detects_any_overlap():
    assert hit_rate(["A", "B"], ["B"]) == 1.0
    assert hit_rate(["A", "B"], ["C"]) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["A", "B", "C"], ["B"]) == 0.5
    assert reciprocal_rank(["A", "B"], ["A"]) == 1.0
    assert reciprocal_rank(["A", "B"], ["Z"]) == 0.0


def test_keyword_recall_returns_fraction_found():
    assert keyword_recall("the sky is blue", ["sky", "blue"]) == 1.0
    assert keyword_recall("the sky is blue", ["sky", "green"]) == 0.5
    assert keyword_recall("anything at all", []) == 1.0


def test_keyword_recall_is_case_insensitive():
    assert keyword_recall("Carbon Dioxide rises", ["carbon", "DIOXIDE"]) == 1.0


def test_citation_match_checks_bracketed_titles():
    assert citation_match("Caused by traffic [Air pollution].", ["Air pollution"]) == 1.0
    assert citation_match("No citation here.", ["Air pollution"]) == 0.0
    assert citation_match("Wrong source [Smog].", ["Air pollution"]) == 0.0
