# AGENTS.md

Local, **offline-only** Retrieval-Augmented Generation system: ingest documents
(PDF/TXT/MD/DOCX/HTML) → chunk → embed → FAISS + BM25 hybrid search → extractive
answers with citations. See [README.md](README.md) for the full feature list,
CLI reference, and architecture diagram — don't duplicate it here.

## Non-negotiable rules

- **No network / API calls, ever.** Everything runs locally (the one exception is
  embedding-model download on first run). Never add HTTP clients, cloud SDKs, or
  telemetry. This is the core product promise and a compliance requirement.
- **`faiss` is the only hard dependency** in the retrieval path. All other heavy
  deps are *optional and degrade gracefully* — match the existing pattern: catch
  `ImportError`, log a warning, and fall back (see `_load_docx`/`_load_html` in
  [rag/loaders.py](rag/loaders.py), `bm25_search` in [rag/reranker.py](rag/reranker.py),
  `pick_default_backend` in [rag/embedder.py](rag/embedder.py)). Don't make an
  optional dependency mandatory.

## Environment & commands

This workspace lives inside a **OneDrive "Files On-Demand"** folder. If a file
read fails with *"The cloud file provider is not running"*, OneDrive isn't
running — start it and retry; the files are cloud placeholders until hydrated.

Use the project venv (`.venv`). On Windows the interpreter is
`.venv\Scripts\python.exe`.

```bash
pip install -r requirements.txt   # install deps
pytest                            # run all tests (testpaths=tests, pythonpath=.)
pytest tests/test_chunker.py -v   # run one test file
python main.py -q "your question" --answer   # one-shot CLI query
python web.py                     # Flask UI at http://localhost:5000
```

VS Code tasks **RAG: Ask (prompt)** and **RAG: Chat (REPL)** wrap `main.py`
(see [.vscode/tasks.json](.vscode/tasks.json)).

## Code conventions

Enforced by config in [pyproject.toml](pyproject.toml): **mypy `strict`** (typed
as Python 3.12), **ruff** `line-length = 120`, target `py310`.

- Start every module with `from __future__ import annotations`.
- Full type hints on every signature. Use PEP 604 unions (`str | None`) and
  lowercase builtin generics (`list[str]`, `dict[str, object]`).
- One logger per module: `logger = logging.getLogger(__name__)`. Configure only
  via `setup_logging()` in [rag/utils.py](rag/utils.py). Use `%`-style args
  (`logger.info("Total chunks: %d", n)`), never f-strings in log calls.
- Prose-only docstrings (one imperative line, no Args/Returns sections).
- `UPPERCASE` module constants, `_`-prefixed private helpers, `snake_case` rest.
- Prefer `@dataclass` for structured data over loose dicts (e.g. `RagConfig` is
  `frozen=True`). Use `norm_path()` from utils for any path that feeds the cache.

## Architecture map

[rag/index.py](rag/index.py) `RagIndex` is the facade — `load_or_build()` and
`search()` are the entry points used by both [main.py](main.py) and
[web.py](web.py). Pipeline: loaders → chunker → embedder → FAISS, then
reranker (BM25 + Reciprocal Rank Fusion + metadata filter) → answerer.

| Module | Responsibility |
|---|---|
| [rag/loaders.py](rag/loaders.py) | Multi-format extraction; loader registry |
| [rag/chunker.py](rag/chunker.py) | Paragraph/bullet-aware chunking + text normalization |
| [rag/embedder.py](rag/embedder.py) | `fastembed` ⟷ `sentence-transformers` backend abstraction |
| [rag/reranker.py](rag/reranker.py) | `tokenize`, BM25, RRF, metadata filters |
| [rag/answerer.py](rag/answerer.py) | Extractive answer + boilerplate filtering + citations |
| [rag/index.py](rag/index.py) | FAISS lifecycle, caching, HNSW |
| [rag/utils.py](rag/utils.py) | Logging, hashing, path helpers |

**Cross-module contract:** results flow as plain `dict`s with stable keys
(`content`, `source`, `page`, `distance`, `sim_score`, `final_score`, …).
Preserve these keys when touching the pipeline. Chunk metadata is
`{"source": str, "page": int}`.

## Extension points

- **New file format:** write a `Callable[[str], list[tuple[str, int]]]`
  (filepath → list of `(text, page)`), then `register_loader("ext", fn)` in
  [rag/loaders.py](rag/loaders.py). No core changes needed.
- **New embedding backend:** extend the `if/elif` in `Embedder.__init__` and
  mirror `encode()` in [rag/embedder.py](rag/embedder.py).
- **New web endpoint:** add a Flask route in [web.py](web.py); note the index
  builds in a background thread guarded by `_bundle_lock`, and uploads invalidate
  the bundle to trigger a lazy reindex.

## Caching pitfalls

The index caches to `.rag_cache/` keyed by SHA-256 fingerprints of documents
**plus** config (backend, model, chunk size/overlap, metric). Editing a document
or changing any of those invalidates the cache and forces a rebuild. When you
change indexing/embedding logic, re-run with `--rebuild` (or `--no-cache`) so you
aren't testing against a stale index. `.rag_cache/` is git-ignored.

## Testing notes

Tests in [tests/](tests/) are pure unit tests over synthetic data — **no model
downloads, no FAISS builds, no network**. Keep new tests in that style; never add
a test that downloads a model or hits the network.
