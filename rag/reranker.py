"""BM25 hybrid search and Reciprocal Rank Fusion reranking."""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared tokenizer (also imported by answerer.py)
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "how", "i", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens (3+ alphanumeric chars, no stopwords)."""
    tokens = re.findall(r"[A-Za-z0-9]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    all_chunks: list[str],
    metadatas: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    """Run BM25 keyword search over *all_chunks* and return ranked results.

    Falls back to an empty list when ``rank_bm25`` is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.debug("rank_bm25 not installed; skipping BM25 search")
        return []

    tokenized_corpus = [tokenize(c) for c in all_chunks]
    query_tokens = tokenize(query)
    if not query_tokens:
        query_tokens = re.findall(r"[A-Za-z0-9]+", query.lower())

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True,
    )

    results: list[dict[str, object]] = []
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        if scores[idx] <= 0:
            break
        meta = metadatas[idx]
        results.append({
            "rank": rank,
            "index": idx,
            "bm25_score": float(scores[idx]),
            "source": meta.get("source", "?"),
            "page": meta.get("page", "?"),
            "content": all_chunks[idx],
        })
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: Sequence[dict[str, object]],
    k: int = 60,
) -> list[dict[str, object]]:
    """Fuse multiple ranked lists with RRF: ``score = Σ 1/(k + rank)``."""
    # Map chunk index → (rrf_score, best result dict)
    fused: dict[int, tuple[float, dict[str, object]]] = {}

    for result_list in result_lists:
        for r in result_list:
            idx = int(r.get("index", -1))
            if idx < 0:
                continue
            rank = int(r.get("rank", 999))
            rrf = 1.0 / (k + rank)
            if idx in fused:
                old_score, old_r = fused[idx]
                fused[idx] = (old_score + rrf, old_r)
            else:
                fused[idx] = (rrf, dict(r))

    sorted_results = sorted(fused.values(), key=lambda x: x[0], reverse=True)

    out: list[dict[str, object]] = []
    for rank, (score, r) in enumerate(sorted_results, start=1):
        r["rank"] = rank
        r["rrf_score"] = score
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def apply_metadata_filters(
    results: list[dict[str, object]],
    filters: dict[str, str],
) -> list[dict[str, object]]:
    """Filter results by ``key=glob_pattern`` pairs (fnmatch-style)."""
    import fnmatch

    filtered: list[dict[str, object]] = []
    for r in results:
        match = True
        for key, pattern in filters.items():
            value = str(r.get(key, ""))
            if not fnmatch.fnmatch(value, pattern):
                match = False
                break
        if match:
            filtered.append(r)
    return filtered
