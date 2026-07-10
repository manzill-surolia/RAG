"""Shared utilities: logging, filesystem helpers, content hashing."""

import hashlib
import logging
import os
import sys


def setup_logging(*, verbose: bool = False) -> None:
    """Configure root logger for the RAG application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def file_content_hash(filepath: str, head_bytes: int = 65_536) -> str:
    """SHA-256 of the first *head_bytes* of a file — fast, reliable fingerprint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def safe_mkdir(path: str) -> None:
    """Create directory (and parents) if missing."""
    os.makedirs(path, exist_ok=True)


def norm_path(path: str) -> str:
    """Normalize a path for stable cross-platform cache validation."""
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/")
