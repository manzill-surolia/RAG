"""Tests for rag.reranker — BM25, RRF, metadata filtering, tokenizer."""

import pytest

from rag.reranker import tokenize, reciprocal_rank_fusion, apply_metadata_filters


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World programming")
        assert "hello" in tokens
        assert "world" in tokens
        assert "programming" in tokens

    def test_removes_stopwords(self):
        tokens = tokenize("what is the meaning of this")
        assert "the" not in tokens
        assert "what" not in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize("I am OK at AI")
        # All tokens <3 chars should be removed
        for t in tokens:
            assert len(t) >= 3

    def test_empty(self):
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        list_a = [
            {"index": 0, "rank": 1, "content": "a"},
            {"index": 1, "rank": 2, "content": "b"},
        ]
        list_b = [
            {"index": 1, "rank": 1, "content": "b"},
            {"index": 2, "rank": 2, "content": "c"},
        ]
        fused = reciprocal_rank_fusion(list_a, list_b, k=60)
        # Index 1 appears in both lists → should have higher score
        scores = {int(r["index"]): r["rrf_score"] for r in fused}
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_single_list_passthrough(self):
        items = [
            {"index": 0, "rank": 1, "content": "x"},
            {"index": 1, "rank": 2, "content": "y"},
        ]
        fused = reciprocal_rank_fusion(items)
        assert len(fused) == 2
        assert fused[0]["rank"] == 1
        assert fused[1]["rank"] == 2

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# apply_metadata_filters
# ---------------------------------------------------------------------------

class TestApplyMetadataFilters:
    def test_glob_filter(self):
        results = [
            {"source": "ISO_27001.pdf", "page": 1},
            {"source": "PCI_DSS.pdf", "page": 2},
            {"source": "ISO_42001.pdf", "page": 3},
        ]
        filtered = apply_metadata_filters(results, {"source": "ISO*"})
        assert len(filtered) == 2
        assert all("ISO" in r["source"] for r in filtered)

    def test_exact_match(self):
        results = [
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 2},
        ]
        filtered = apply_metadata_filters(results, {"source": "a.pdf"})
        assert len(filtered) == 1

    def test_no_filter_returns_all(self):
        results = [{"source": "x.pdf"}, {"source": "y.pdf"}]
        filtered = apply_metadata_filters(results, {})
        assert len(filtered) == 2

    def test_page_filter(self):
        results = [
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": 5},
        ]
        filtered = apply_metadata_filters(results, {"page": "1"})
        assert len(filtered) == 1
