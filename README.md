# Local Document RAG

Offline, local-only Retrieval-Augmented Generation system. Ask natural-language
questions against a corpus of documents and get extractive answers with
inline citations — **no API calls, no network, everything runs locally**.

## Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | PDF, TXT, Markdown, DOCX, HTML |
| **Paragraph & bullet-aware chunking** | Preserves document structure during splitting |
| **Dual embedding backends** | `fastembed` (lightweight, ONNX) or `sentence-transformers` |
| **FAISS vector search** | Flat (exact) or HNSW (approximate, for large corpora) |
| **BM25 hybrid search** | Combines keyword (BM25) + semantic (vector) via Reciprocal Rank Fusion |
| **Token-overlap reranking** | Lightweight post-retrieval reranking for better precision |
| **Metadata filtering** | Scope queries by source, page, or any metadata field (`--filter source=ISO*`) |
| **Extractive answers** | Sentence-level answer extraction with `(source pN)` citations |
| **Digest / summary** | Per-source key-point summaries alongside raw passages |
| **Web interface** | Flask UI with upload, query, document management, and live index status |
| **Cache with SHA-256 fingerprints** | Fast restarts; auto-invalidates when documents or config change |
| **Interactive REPL & one-shot CLI** | Chat mode or batch queries |

## Quick Start

### 1. Install

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Documents

Place your documents (`.pdf`, `.txt`, `.md`, `.docx`, `.html`) in the
`documents/` directory.

### 3. Run

**Web interface** (recommended):

```bash
python web.py
```

Open **http://localhost:5000** in your browser. The index builds in the background —
the UI is usable immediately. Upload documents, ask queries, and see answers with
digest/summary and sources.

**Command-line**:

```bash
# Interactive chat (default)
python main.py

# One-shot query
python main.py -q "What is the password policy?"

# With extractive answer + sources
python main.py -q "GDPR breach notification" --answer

# Demo mode (built-in queries)
python main.py --demo --answer
```

## CLI Reference

```
usage: main.py [-h] [--query QUERY] [--chat] [--demo]
               [--docs-dir DOCS_DIR]
               [--embedding-backend {fastembed,sentence-transformers}]
               [--embedding-model MODEL]
               [--profile {fast,balanced,accurate}]
               [--chunk-size N] [--chunk-overlap N]
               [--top-k K] [--metric {cosine,l2}]
               [--index-type {auto,flat,hnsw}]
               [--no-rerank] [--no-bm25]
               [--filter KEY=PATTERN]
               [--cache-dir DIR] [--rebuild] [--no-cache]
               [--debug-cache]
               [--max-chars N] [--answer]
               [--answer-sentences N]
               [--answer-format {paragraph,lines}]
               [--show-passages] [--verbose]
```

### Key options

| Flag | Description |
|---|---|
| `-q` / `--query` | Query text (repeatable) |
| `--chat` | Interactive REPL (default when no queries given) |
| `--answer` | Generate local extractive answer with citations |
| `--profile fast\|balanced\|accurate` | Chunking preset (speed vs. granularity) |
| `--index-type auto\|flat\|hnsw` | FAISS index type (`auto` picks HNSW when >5 000 chunks) |
| `--no-bm25` | Disable BM25 hybrid search (vector-only) |
| `--filter key=pattern` | Glob-style metadata filter (e.g., `source=ISO*`) |
| `--rebuild` | Force rebuild the index cache |
| `--verbose` / `-v` | Enable debug logging |

## Project Structure

```
main.py                   # CLI entry point
web.py                    # Flask web interface (upload, query, manage)
templates/
  index.html              # Dark-themed SPA (Ask, Documents, Upload tabs)
rag/
  __init__.py             # Package init
  utils.py                # Logging, hashing, path helpers
  embedder.py             # Embedding backend wrapper
  chunker.py              # Paragraph/bullet-aware text chunking
  loaders.py              # Multi-format document loaders (PDF, TXT, MD, DOCX, HTML)
  reranker.py             # BM25 search, RRF fusion, metadata filtering
  answerer.py             # Local extractive answer generation + digest
  index.py                # FAISS index management, caching, HNSW support
tests/
  test_chunker.py         # Chunking unit tests
  test_answerer.py        # Answer generation tests
  test_index.py           # Index/cache tests
  test_reranker.py        # Reranker/tokenizer tests
scripts/
  generate_sample_docs.py # Generate sample PDFs for testing
documents/                # Place your documents here
```

## Architecture

```
Documents (PDF/TXT/MD/DOCX/HTML)
        │
        ▼
   ┌──────────┐
   │  Loaders  │  ─── pluggable per file type
   └────┬─────┘
        │ raw text + page metadata
        ▼
   ┌──────────┐
   │  Chunker  │  ─── paragraph/bullet-aware, configurable overlap
   └────┬─────┘
        │ chunks + metadata
        ▼
   ┌──────────┐
   │ Embedder  │  ─── fastembed (ONNX) or sentence-transformers
   └────┬─────┘
        │ float32 vectors
        ▼
   ┌───────────────┐
   │  FAISS Index   │  ─── Flat (exact) or HNSW (approximate)
   │  + SHA-256     │      with disk cache & auto-invalidation
   │    cache       │
   └───────┬───────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
 Vector        BM25 keyword
 search         search
    │              │
    └──────┬───────┘
           │ Reciprocal Rank Fusion
           ▼
    Token-overlap rerank
           │
           ▼
    Metadata filter
           │
           ▼
    Extractive answerer
     (sentence selection
      + citations)
           │
           ▼
    Digest / per-source
      key-point summary
```

## Web Interface

Run `python web.py` and open **http://localhost:5000**.

| Tab | Function |
|---|---|
| **Ask** | Enter a question, get extractive answer + digest + sources + passages |
| **Documents** | View all indexed files, delete individual files |
| **Upload** | Drag & drop (or browse) to upload PDF, TXT, MD, DOCX, HTML files |

The header shows a live index status badge (building / ready / chunks count)
and a **Rebuild Index** button to force re-indexing after uploads.

The index builds in a background thread on startup — the server is responsive
immediately, even while embedding thousands of chunks.

## Extending

### Add a custom document loader

```python
from rag.loaders import register_loader

def load_csv(filepath: str) -> list[tuple[str, int]]:
    import csv
    with open(filepath) as f:
        text = "\n".join(", ".join(row) for row in csv.reader(f))
    return [(text, 1)]

register_loader("csv", load_csv)
```

## Running Tests

```bash
pytest
```

## Notes

- **First run** builds the embedding index and caches it to `.rag_cache/`.
  Subsequent runs load from cache in seconds.
- The cache auto-invalidates when documents change (SHA-256 fingerprints),
  or when chunking / embedding settings change.
- Pin dependency versions in `requirements.txt` for reproducible builds.
