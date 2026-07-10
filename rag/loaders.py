"""Multi-format document loaders with a pluggable registry."""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Callable

from .chunker import chunk_text

logger = logging.getLogger(__name__)

# A loader returns a list of (text, page_number) tuples per file.
LoaderFunc = Callable[[str], list[tuple[str, int]]]

_LOADERS: dict[str, LoaderFunc] = {}

# Regex patterns matching common page boilerplate (copyright footers, headers)
_PAGE_BOILERPLATE_RE = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"\u00a9\s*\d{4}[^\n]{0,200}(?:all rights reserved|reserved)[^\n]*"
    r"|[^\n]*all rights reserved[^\n]*(?:prohibited|permission|reproduced)[^\n]*"
    r"|[^\n]*no part of this publication[^\n]*"
    r"|[^\n]*may not be reproduced[^\n]*"
    r"|[^\n]*prior written (?:permission|consent)[^\n]*"
    r"|[^\n]*proprietary and confidential[^\n]*"
    r"|page\s+\d+\s+of\s+\d+"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_page_boilerplate(text: str) -> str:
    """Remove copyright footers and other recurring boilerplate from page text."""
    return _PAGE_BOILERPLATE_RE.sub("", text).strip()


def register_loader(ext: str, func: LoaderFunc) -> None:
    """Register a document loader for a file extension (without dot)."""
    _LOADERS[ext.lower().lstrip(".")] = func


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------

def _load_pdf(filepath: str) -> list[tuple[str, int]]:
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages: list[tuple[str, int]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((text, i + 1))
    return pages


def _load_text(filepath: str) -> list[tuple[str, int]]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [(text, 1)] if text.strip() else []


def _load_docx(filepath: str) -> list[tuple[str, int]]:
    try:
        import docx  # python-docx
    except ImportError:
        logger.warning("python-docx not installed; skipping %s", filepath)
        return []
    doc = docx.Document(filepath)
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(text, 1)] if text.strip() else []


def _load_html(filepath: str) -> list[tuple[str, int]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; skipping %s", filepath)
        return []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text(separator="\n")
    return [(text, 1)] if text.strip() else []


# Register default loaders
register_loader("pdf", _load_pdf)
register_loader("txt", _load_text)
register_loader("md", _load_text)
register_loader("docx", _load_docx)
register_loader("html", _load_html)
register_loader("htm", _load_html)

SUPPORTED_EXTENSIONS = tuple(_LOADERS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(
    documents_dir: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[dict[str, object]]]:
    """Load all supported documents from *documents_dir* and chunk them.

    Returns ``(chunks, metadatas)`` where each metadata dict has
    ``source`` and ``page`` keys.
    """
    logger.info("Loading documents from '%s'...", documents_dir)
    all_chunks: list[str] = []
    metadatas: list[dict[str, object]] = []
    file_count = 0

    for ext, loader in _LOADERS.items():
        pattern = os.path.join(documents_dir, f"*.{ext}")
        for filepath in sorted(glob.glob(pattern)):
            try:
                pages = loader(filepath)
                fname = os.path.basename(filepath)
                for text, page_num in pages:
                    # Strip recurring copyright footers / boilerplate
                    text = _strip_page_boilerplate(text)
                    if not text.strip():
                        continue
                    page_chunks = chunk_text(
                        text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    )
                    for chunk in page_chunks:
                        all_chunks.append(chunk)
                        metadatas.append({"source": fname, "page": page_num})
                file_count += 1
            except Exception as e:
                logger.error("Error reading %s: %s", filepath, e)

    logger.info(
        "Processed %d files (%s). Total chunks: %d",
        file_count,
        ", ".join(SUPPORTED_EXTENSIONS),
        len(all_chunks),
    )
    return all_chunks, metadatas
