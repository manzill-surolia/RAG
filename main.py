#!/usr/bin/env python3
"""Local document RAG — offline semantic search with extractive answers.

Supports PDF, TXT, MD, DOCX, and HTML documents.
Uses FAISS for vector search, optional BM25 hybrid, and extractive answering.
No API calls — everything runs locally.
"""

from __future__ import annotations

import argparse
import os
import sys

from rag.answerer import (
    filter_boilerplate_results,
    format_sources,
    local_extractive_answer,
    print_results,
)
from rag.embedder import pick_default_backend
from rag.index import RagConfig, RagIndex
from rag.utils import setup_logging


def _parse_filters(raw: list[str] | None) -> dict[str, str]:
    """Parse ``--filter key=value`` arguments into a dict."""
    if not raw:
        return {}
    filters: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            print(
                f"Warning: ignoring malformed filter '{item}' (expected key=value)",
                file=sys.stderr,
            )
            continue
        key, _, value = item.partition("=")
        filters[key.strip()] = value.strip()
    return filters


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local document semantic search + extractive answers (FAISS + embeddings)",
    )

    # --- query sources ---
    parser.add_argument("queries", nargs="*", help="Queries (positional).")
    parser.add_argument("--query", "-q", action="append", default=[], help="Query (repeatable).")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode (REPL).")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo queries.")

    # --- document & embedding ---
    parser.add_argument("--docs-dir", default="documents", help="Directory containing documents.")
    parser.add_argument(
        "--embedding-backend",
        default=None,
        choices=["fastembed", "sentence-transformers"],
        help="Embedding backend (default: fastembed if installed).",
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model name.")

    # --- chunking ---
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Indexing preset: fast, balanced (default), accurate.",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in chars (overrides --profile).")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in chars (overrides --profile).")

    # --- search ---
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Vector search metric.")
    parser.add_argument(
        "--index-type",
        choices=["auto", "flat", "hnsw"],
        default="auto",
        help="FAISS index type (auto selects HNSW for >5 000 chunks).",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Disable token-overlap reranking.")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25 hybrid search.")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter as key=pattern (glob). Repeatable. E.g. --filter source=ISO*",
    )

    # --- cache ---
    parser.add_argument("--cache-dir", default=None, help="Cache directory.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache (always in-memory).")
    parser.add_argument("--debug-cache", action="store_true", help="Print cache rejection reasons.")

    # --- output ---
    parser.add_argument("--max-chars", type=int, default=200, help="Max chars per chunk in output (0=full).")
    parser.add_argument("--answer", action="store_true", help="Generate local extractive answer.")
    parser.add_argument("--answer-sentences", type=int, default=4, help="Max sentences in answer.")
    parser.add_argument("--answer-format", choices=["paragraph", "lines"], default="paragraph")
    parser.add_argument("--show-passages", action="store_true", help="Show passages alongside answer.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    # --- logging ---
    setup_logging(verbose=args.verbose)

    # --- chunking profile ---
    profiles = {
        "fast": (1400, 120),
        "balanced": (900, 100),
        "accurate": (500, 80),
    }
    default_size, default_overlap = profiles[args.profile]
    chunk_size = args.chunk_size if args.chunk_size is not None else default_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else default_overlap

    # --- cache directory discovery ---
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif os.path.isdir(".rag_cache"):
        cache_dir = ".rag_cache"
    elif os.path.isdir("rag_cache"):
        cache_dir = "rag_cache"
    else:
        cache_dir = ".rag_cache"

    # --- embedding defaults ---
    embedding_backend = args.embedding_backend or pick_default_backend()
    if args.embedding_model:
        embedding_model = args.embedding_model
    else:
        embedding_model = (
            "BAAI/bge-small-en-v1.5" if embedding_backend == "fastembed" else "all-MiniLM-L6-v2"
        )

    # --- validate docs dir ---
    documents_dir = args.docs_dir
    if not os.path.isdir(documents_dir):
        print(f"Error: Documents directory not found: {documents_dir}", file=sys.stderr)
        return 2

    # --- determine query mode ---
    cli_queries = list(args.query) + list(args.queries)
    if args.demo:
        cli_queries = ["password policy", "GDPR breach", "clean desk"]
    if not cli_queries and not args.chat:
        args.chat = True

    filters = _parse_filters(args.filter)

    # --- build config & index ---
    config = RagConfig(
        documents_dir=documents_dir,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cache_dir=cache_dir,
        index_metric=args.metric,
        index_type=args.index_type,
        rerank=not args.no_rerank,
        use_bm25=not args.no_bm25,
        debug_cache=args.debug_cache,
    )

    bundle = RagIndex.load_or_build(config=config, rebuild=args.rebuild, no_cache=args.no_cache)

    # --- shared query runner ---
    def _run_query(query_text: str) -> None:
        # Over-fetch and filter boilerplate for better answer quality
        fetch_k = args.top_k * 3
        raw_results = RagIndex.search(bundle, query_text, fetch_k, filters=filters)
        results = filter_boilerplate_results(raw_results)[:args.top_k]
        if not results:
            results = raw_results[:args.top_k]

        if not args.answer:
            print_results(results=results, max_chars=args.max_chars)
            return

        answer = local_extractive_answer(
            question=query_text,
            results=results,
            max_sentences=args.answer_sentences,
            format_style=args.answer_format,
        )

        print("\nAnswer (local):")
        print(answer)
        print("\nSources:")
        print(format_sources(results) or "- (none)")
        if args.show_passages:
            print("\nPassages:")
            print_results(results=results, max_chars=args.max_chars)

    # --- interactive chat ---
    if args.chat:
        print("\n--- RAG Chat ---")
        print("Ask a question about your documents.")
        print("Commands: /exit, /quit")

        while True:
            try:
                query_text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query_text:
                continue
            if query_text.lower() in {"/exit", "/quit"}:
                break

            print(f"\nQuerying: '{query_text}'")
            _run_query(query_text)

    # --- one-shot queries ---
    for query_text in cli_queries:
        print(f"\nQuerying: '{query_text}'")
        _run_query(query_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
