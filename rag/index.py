"""FAISS index management — build, cache, load, search, with HNSW support."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import faiss
import numpy as np

from .embedder import Embedder, pick_default_backend
from .loaders import SUPPORTED_EXTENSIONS, load_documents
from .reranker import (
    apply_metadata_filters,
    bm25_search,
    reciprocal_rank_fusion,
    tokenize,
)
from .utils import file_content_hash, norm_path, safe_mkdir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & bundle types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RagConfig:
    """Parameters needed to reproduce an index build."""

    documents_dir: str = "documents"
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 80
    cache_dir: str = ".rag_cache"
    index_metric: str = "cosine"  # cosine | l2
    index_type: str = "auto"  # auto | flat | hnsw
    rerank: bool = True
    use_bm25: bool = True
    debug_cache: bool = False


@dataclass
class RagIndexBundle:
    """Runtime artefacts: embedder + FAISS index + chunk data."""

    embedder: Embedder
    index: faiss.Index
    chunks: list[str]
    metadatas: list[dict[str, object]]
    index_metric: str
    index_type: str  # "flat" or "hnsw"
    rerank: bool
    use_bm25: bool


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows to unit length (for cosine similarity via inner product)."""
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "index": os.path.join(cache_dir, "index.faiss"),
        "chunks": os.path.join(cache_dir, "chunks.json"),
        "meta": os.path.join(cache_dir, "metadatas.json"),
        "info": os.path.join(cache_dir, "index_info.json"),
    }


def _collect_file_fingerprints(documents_dir: str) -> list[dict[str, object]]:
    """SHA-256 based fingerprints for reliable cache invalidation."""
    fingerprints: list[dict[str, object]] = []
    for ext in SUPPORTED_EXTENSIONS:
        for filepath in sorted(glob.glob(os.path.join(documents_dir, f"*.{ext}"))):
            try:
                st = os.stat(filepath)
                fingerprints.append({
                    "path": os.path.relpath(filepath, documents_dir).replace("\\", "/"),
                    "size": int(st.st_size),
                    "hash": file_content_hash(filepath),
                })
            except OSError:
                continue
    return fingerprints


# ---------------------------------------------------------------------------
# Shared build core (eliminates duplication between cached / in-memory paths)
# ---------------------------------------------------------------------------

def _build_index_core(
    *,
    documents_dir: str,
    embedding_backend: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_metric: str,
    index_type: str,
) -> tuple[Embedder, faiss.Index, list[str], list[dict[str, object]], str]:
    """Build embeddings + FAISS index.  Returns ``(embedder, index, chunks, metadatas, effective_type)``."""
    all_chunks, metadatas = load_documents(
        documents_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    if not all_chunks:
        raise RuntimeError(
            f"No text extracted from documents in '{documents_dir}'. "
            "Ensure the directory contains supported files "
            f"({', '.join(SUPPORTED_EXTENSIONS)})."
        )

    logger.info("Loading embedding model (%s: %s)...", embedding_backend, embedding_model)
    embedder = Embedder(embedding_backend, embedding_model)

    logger.info("Encoding %d chunks...", len(all_chunks))
    embeddings = embedder.encode(all_chunks, show_progress=True)

    index_metric = (index_metric or "cosine").lower()
    if index_metric not in {"cosine", "l2"}:
        raise ValueError("index_metric must be 'cosine' or 'l2'")

    if index_metric == "cosine":
        embeddings = _normalize_rows(embeddings)

    dim = int(embeddings.shape[1])
    n = int(embeddings.shape[0])

    # Auto-select index type: HNSW for large corpora, flat otherwise.
    effective_type = index_type
    if effective_type == "auto":
        effective_type = "hnsw" if n > 5_000 else "flat"

    logger.info(
        "Building %s index (metric=%s, dim=%d, vectors=%d)...",
        effective_type, index_metric, dim, n,
    )

    if effective_type == "hnsw":
        # IndexHNSWFlat uses L2 internally.
        # For cosine: vectors are pre-normalized, so L2 on unit vectors ≈ cosine.
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(embeddings)
    else:
        if index_metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    return embedder, index, all_chunks, metadatas, effective_type


# ---------------------------------------------------------------------------
# Build + cache
# ---------------------------------------------------------------------------

def _build_and_cache(config: RagConfig) -> tuple[Embedder, faiss.Index, list[str], list[dict[str, object]], str]:
    safe_mkdir(config.cache_dir)
    paths = _cache_paths(config.cache_dir)

    embedder, index, chunks, metadatas, effective_type = _build_index_core(
        documents_dir=config.documents_dir,
        embedding_backend=config.embedding_backend,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        index_metric=config.index_metric,
        index_type=config.index_type,
    )

    logger.info("Saving index cache to '%s'...", config.cache_dir)
    faiss.write_index(index, paths["index"])

    with open(paths["chunks"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)
    with open(paths["info"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "documents_dir": config.documents_dir,
                "documents_dir_abs": norm_path(config.documents_dir),
                "embedding_backend": config.embedding_backend,
                "embedding_model": config.embedding_model,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "index_metric": config.index_metric,
                "index_type": effective_type,
                "file_fingerprints": _collect_file_fingerprints(config.documents_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return embedder, index, chunks, metadatas, effective_type


def _build_in_memory(config: RagConfig) -> tuple[Embedder, faiss.Index, list[str], list[dict[str, object]], str]:
    return _build_index_core(
        documents_dir=config.documents_dir,
        embedding_backend=config.embedding_backend,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        index_metric=config.index_metric,
        index_type=config.index_type,
    )


# ---------------------------------------------------------------------------
# Load cached index
# ---------------------------------------------------------------------------

def _try_load_cached(config: RagConfig) -> tuple[Embedder, faiss.Index, list[str], list[dict[str, object]], str, str] | None:
    """Load cached artefacts.  Returns ``(embedder, index, chunks, metadatas, metric, index_type)`` or ``None``."""
    paths = _cache_paths(config.cache_dir)
    debug = config.debug_cache

    if not all(os.path.exists(p) for p in paths.values()):
        return None

    try:
        with open(paths["info"], "r", encoding="utf-8") as f:
            info = json.load(f)

        # --- validate documents directory ---
        cached_abs = str(info.get("documents_dir_abs") or "").strip()
        current_abs = norm_path(config.documents_dir)
        if cached_abs:
            if cached_abs != current_abs:
                if debug:
                    logger.debug("Cache invalid: documents_dir_abs mismatch (%s vs %s)", cached_abs, current_abs)
                return None
        else:
            cached_raw = str(info.get("documents_dir") or "")
            if norm_path(cached_raw) != current_abs:
                if debug:
                    logger.debug("Cache invalid: documents_dir mismatch")
                return None

        # --- validate embedding settings ---
        for key, expected in [
            ("embedding_backend", config.embedding_backend),
            ("embedding_model", config.embedding_model),
        ]:
            if info.get(key) != expected:
                if debug:
                    logger.debug("Cache invalid: %s mismatch (%s vs %s)", key, info.get(key), expected)
                return None

        # --- validate chunking ---
        if int(info.get("chunk_size", -1)) != config.chunk_size:
            if debug:
                logger.debug("Cache invalid: chunk_size mismatch")
            return None
        if int(info.get("chunk_overlap", -1)) != config.chunk_overlap:
            if debug:
                logger.debug("Cache invalid: chunk_overlap mismatch")
            return None

        # --- validate metric ---
        cached_metric = str(info.get("index_metric", "l2")).lower()
        if cached_metric != config.index_metric.lower():
            if debug:
                logger.debug("Cache invalid: index_metric mismatch")
            return None

        # --- validate file fingerprints (SHA-256 based) ---
        cached_fps = info.get("file_fingerprints") or info.get("pdf_fingerprints", [])
        current_fps = _collect_file_fingerprints(config.documents_dir)
        if cached_fps != current_fps:
            if debug:
                logger.debug("Cache invalid: file fingerprints changed")
            return None

        # --- load ---
        logger.info("Loading cached index from '%s'...", config.cache_dir)
        index = faiss.read_index(paths["index"])
        with open(paths["chunks"], "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(paths["meta"], "r", encoding="utf-8") as f:
            metadatas = json.load(f)

        embedder = Embedder(config.embedding_backend, config.embedding_model)
        cached_type = str(info.get("index_type", "flat")).lower()
        return embedder, index, chunks, metadatas, cached_metric, cached_type

    except Exception:
        return None


# ---------------------------------------------------------------------------
# RagIndex facade
# ---------------------------------------------------------------------------

class RagIndex:
    """High-level facade for load / build / search operations."""

    @staticmethod
    def load_cached(config: RagConfig) -> RagIndexBundle | None:
        loaded = _try_load_cached(config)
        if loaded is None:
            return None
        embedder, index, chunks, metadatas, metric, itype = loaded
        return RagIndexBundle(
            embedder=embedder, index=index, chunks=chunks, metadatas=metadatas,
            index_metric=metric, index_type=itype,
            rerank=config.rerank, use_bm25=config.use_bm25,
        )

    @staticmethod
    def build_and_cache(config: RagConfig) -> RagIndexBundle:
        embedder, index, chunks, metadatas, effective_type = _build_and_cache(config)
        return RagIndexBundle(
            embedder=embedder, index=index, chunks=chunks, metadatas=metadatas,
            index_metric=config.index_metric, index_type=effective_type,
            rerank=config.rerank, use_bm25=config.use_bm25,
        )

    @staticmethod
    def build_in_memory(config: RagConfig) -> RagIndexBundle:
        embedder, index, chunks, metadatas, effective_type = _build_in_memory(config)
        return RagIndexBundle(
            embedder=embedder, index=index, chunks=chunks, metadatas=metadatas,
            index_metric=config.index_metric, index_type=effective_type,
            rerank=config.rerank, use_bm25=config.use_bm25,
        )

    @staticmethod
    def load_or_build(*, config: RagConfig, rebuild: bool, no_cache: bool) -> RagIndexBundle:
        if no_cache:
            logger.info("Cache disabled; building index in-memory...")
            return RagIndex.build_in_memory(config)
        if not rebuild:
            cached = RagIndex.load_cached(config)
            if cached is not None:
                return cached
        if rebuild:
            logger.info("Rebuild requested; rebuilding index cache...")
        else:
            logger.info("No valid cache found; building index (first run may take a few minutes)...")
        return RagIndex.build_and_cache(config)

    @staticmethod
    def search(
        bundle: RagIndexBundle,
        query_text: str,
        top_k: int,
        *,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, object]]:
        """Vector search (+ optional BM25 hybrid) with reranking and filtering."""
        return retrieve_top_k(
            query_text=query_text,
            embedder=bundle.embedder,
            index=bundle.index,
            all_chunks=bundle.chunks,
            metadatas=bundle.metadatas,
            top_k=top_k,
            index_metric=bundle.index_metric,
            index_type=bundle.index_type,
            rerank=bundle.rerank,
            use_bm25=bundle.use_bm25,
            filters=filters,
        )


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    *,
    query_text: str,
    embedder: Embedder,
    index: faiss.Index,
    all_chunks: list[str],
    metadatas: list[dict[str, object]],
    top_k: int,
    index_metric: str = "cosine",
    index_type: str = "flat",
    rerank: bool = True,
    use_bm25: bool = True,
    filters: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Embed query → FAISS search (+ BM25 hybrid) → rerank → filter → top-k."""

    index_metric = (index_metric or "cosine").lower()
    query_vector = embedder.encode([query_text])
    if index_metric == "cosine":
        query_vector = _normalize_rows(query_vector)

    fetch_k = max(1, int(top_k))
    if rerank or use_bm25 or filters:
        fetch_k = min(max(fetch_k * 5, fetch_k), max(1, int(index.ntotal)))

    D, I = index.search(query_vector, fetch_k)

    vector_results: list[dict[str, object]] = []
    for rank in range(len(I[0])):
        idx = int(I[0][rank])
        if idx < 0 or idx >= len(all_chunks):
            continue
        meta = metadatas[idx]
        raw_score = float(D[0][rank]) if D is not None else 0.0

        # Score interpretation depends on index type + metric.
        if index_type == "hnsw" and index_metric == "cosine":
            # HNSW uses L2 on normalized vectors: d = 2 − 2·cos → cos = 1 − d/2
            sim_score = 1.0 - (raw_score / 2.0)
        elif index_metric == "cosine":
            # IndexFlatIP returns cosine directly.
            sim_score = raw_score
        else:
            # L2: lower is better → negate for consistent "higher = better".
            sim_score = -raw_score

        vector_results.append({
            "rank": rank + 1,
            "index": idx,
            "distance": raw_score,
            "sim_score": sim_score,
            "source": meta.get("source", "?"),
            "page": meta.get("page", "?"),
            "content": all_chunks[idx],
        })

    # --- BM25 hybrid ---
    if use_bm25:
        bm25_results = bm25_search(query_text, all_chunks, metadatas, top_k=fetch_k)
        if bm25_results:
            results = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            results = vector_results
    else:
        results = vector_results

    # --- Token overlap reranking ---
    if rerank and len(results) > 1:
        q_tokens = set(tokenize(query_text))
        if not q_tokens:
            q_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", query_text.lower()))

        for r in results:
            content = str(r.get("content", ""))
            c_tokens = set(tokenize(content))
            overlap = float(len(q_tokens & c_tokens))
            r["overlap"] = overlap

            if "rrf_score" in r:
                # Hybrid mode: RRF already fuses signals; add gentle overlap tiebreaker.
                r["final_score"] = float(r["rrf_score"]) + overlap * 0.001
            else:
                # Vector-only mode: preserve original scoring logic.
                sim = float(r.get("sim_score", 0.0))
                rnk = int(r.get("rank", 999))
                rank_bonus = 1.0 / max(1, rnk)
                r["final_score"] = sim + overlap * 0.25 + rank_bonus

        results.sort(key=lambda x: float(x.get("final_score", 0)), reverse=True)

    # --- Metadata filtering ---
    if filters:
        results = apply_metadata_filters(results, filters)

    # --- Final trim + rank reassignment ---
    results = results[:top_k]
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results
