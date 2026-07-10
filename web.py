#!/usr/bin/env python3
"""Flask web interface for the local RAG system.

Provides:
  - File upload to documents/
  - Query interface with answer + digest + sources
  - Document listing & deletion
  - Index rebuild trigger
"""

from __future__ import annotations

import logging
import os
import threading

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from rag.answerer import (
    build_digest,
    filter_boilerplate_results,
    format_digest,
    format_sources,
    local_extractive_answer,
)
from rag.embedder import pick_default_backend
from rag.index import RagConfig, RagIndex, RagIndexBundle
from rag.utils import setup_logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "docx", "html", "htm"}

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ---------------------------------------------------------------------------
# Index state (singleton, rebuilt on demand)
# ---------------------------------------------------------------------------

_bundle: RagIndexBundle | None = None
_bundle_lock = threading.Lock()
_building = False
_build_error: str | None = None


def _make_config() -> RagConfig:
    backend = pick_default_backend()
    model = "BAAI/bge-small-en-v1.5" if backend == "fastembed" else "all-MiniLM-L6-v2"
    return RagConfig(
        documents_dir=DOCUMENTS_DIR,
        embedding_backend=backend,
        embedding_model=model,
        chunk_size=900,
        chunk_overlap=100,
        cache_dir=os.path.join(BASE_DIR, ".rag_cache"),
        index_metric="cosine",
        index_type="auto",
        rerank=True,
        use_bm25=True,
    )


def _build_in_background(*, rebuild: bool = False) -> None:
    """Build/load the index in a background thread."""
    global _bundle, _building, _build_error
    try:
        config = _make_config()
        logger.info("Building / loading index (background)...")
        bundle = RagIndex.load_or_build(config=config, rebuild=rebuild, no_cache=False)
        with _bundle_lock:
            _bundle = bundle
            _building = False
            _build_error = None
        logger.info("Index ready (%d chunks).", len(bundle.chunks))
    except Exception as e:
        logger.error("Index build failed: %s", e)
        with _bundle_lock:
            _building = False
            _build_error = str(e)


def _get_bundle(*, rebuild: bool = False) -> RagIndexBundle | None:
    """Return the current bundle, triggering a background build if needed."""
    global _building
    with _bundle_lock:
        if _building:
            return _bundle  # return stale (or None) while building
        if _bundle is not None and not rebuild:
            return _bundle
        _building = True

    # Kick off background build
    t = threading.Thread(target=_build_in_background, kwargs={"rebuild": rebuild}, daemon=True)
    t.start()

    if rebuild:
        # For explicit rebuild requests, wait for completion
        t.join()
        return _bundle

    return _bundle  # None on first call — UI shows "building" state


def _invalidate_bundle() -> None:
    """Mark the bundle as stale so next query triggers a rebuild."""
    global _bundle
    with _bundle_lock:
        _bundle = None


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _list_documents() -> list[dict[str, object]]:
    """List all documents in the documents directory."""
    docs: list[dict[str, object]] = []
    if not os.path.isdir(DOCUMENTS_DIR):
        return docs
    for f in sorted(os.listdir(DOCUMENTS_DIR)):
        filepath = os.path.join(DOCUMENTS_DIR, f)
        if os.path.isfile(filepath):
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            size_kb = os.path.getsize(filepath) / 1024
            docs.append({"name": f, "ext": ext, "size_kb": round(size_kb, 1)})
    return docs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index_page():
    documents = _list_documents()
    return render_template("index.html", documents=documents)


@app.route("/upload", methods=["POST"])
def upload_files():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    files = request.files.getlist("files")
    uploaded = []
    errors = []

    for file in files:
        if not file or not file.filename:
            continue
        if not _allowed_file(file.filename):
            errors.append(f"'{file.filename}' — unsupported type (allowed: {', '.join(ALLOWED_EXTENSIONS)})")
            continue
        filename = secure_filename(file.filename)
        dest = os.path.join(DOCUMENTS_DIR, filename)
        file.save(dest)
        uploaded.append(filename)
        logger.info("Uploaded: %s", filename)

    if uploaded:
        _invalidate_bundle()  # force re-index on next query

    return jsonify({"uploaded": uploaded, "errors": errors})


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename: str):
    safe_name = secure_filename(filename)
    filepath = os.path.join(DOCUMENTS_DIR, safe_name)
    if os.path.isfile(filepath):
        os.remove(filepath)
        _invalidate_bundle()
        logger.info("Deleted: %s", safe_name)
        return jsonify({"deleted": safe_name})
    return jsonify({"error": "File not found"}), 404


@app.route("/status")
def status():
    """Return the current index build status."""
    with _bundle_lock:
        building = _building
        ready = _bundle is not None
        chunks = len(_bundle.chunks) if _bundle else 0
        error = _build_error
    return jsonify({
        "building": building,
        "ready": ready,
        "chunks": chunks,
        "error": error,
    })


@app.route("/rebuild", methods=["POST"])
def rebuild_index():
    bundle = _get_bundle(rebuild=True)
    if bundle:
        return jsonify({"status": "ok", "chunks": len(bundle.chunks)})
    return jsonify({"status": "error", "message": "No documents found"}), 400


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    top_k = int(data.get("top_k", 5))
    bundle = _get_bundle()
    if bundle is None:
        with _bundle_lock:
            building = _building
        if building:
            return jsonify({"error": "Index is still building. Please wait a moment and try again."}), 503
        return jsonify({"error": "No index available. Upload documents first."}), 400

    # Over-fetch to compensate for boilerplate filtering
    fetch_k = top_k * 3
    raw_results = RagIndex.search(bundle, question, fetch_k)
    results = filter_boilerplate_results(raw_results)[:top_k]

    if not results:
        # Fallback: if everything was boilerplate, use original results
        results = raw_results[:top_k]

    # Extractive answer
    answer = local_extractive_answer(
        question=question,
        results=results,
        max_sentences=4,
        format_style="paragraph",
    )

    # Digest
    digest_data = build_digest(question=question, results=results)
    digest_text = format_digest(digest_data)

    # Sources
    sources_text = format_sources(results)

    # Raw passages for the UI
    passages = []
    for r in results:
        passages.append({
            "rank": r.get("rank"),
            "source": r.get("source"),
            "page": r.get("page"),
            "content": str(r.get("content", ""))[:500],
        })

    return jsonify({
        "question": question,
        "answer": answer,
        "digest": digest_data,
        "digest_text": digest_text,
        "sources": sources_text,
        "passages": passages,
    })


@app.route("/documents")
def documents_list():
    return jsonify(_list_documents())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Kick off background index build so the server starts immediately
    if os.path.isdir(DOCUMENTS_DIR) and os.listdir(DOCUMENTS_DIR):
        _get_bundle()  # non-blocking — builds in background thread
        logger.info("Index build started in background. Server is ready for requests.")
    else:
        logger.info("No documents found; upload files via the web UI to get started.")

    print("\n  RAG Web Interface running at:  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
