"""Text chunking — paragraph-aware, bullet-aware, with configurable overlap."""

from __future__ import annotations

import re

# Detect list markers so bullet lists keep structure during chunking.
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\).\]]|[A-Za-z][\).\]]|\u2022)\s+")

# Coarse sentence boundary splitter.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_extracted_text(text: str) -> str:
    """Clean PDF/document extraction artifacts for stable downstream processing."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_units(text: str) -> list[str]:
    """Split text into paragraph / bullet units without destroying structure."""
    text = normalize_extracted_text(text)
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    units: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue

        bulletish_count = sum(1 for ln in lines if _BULLET_RE.match(ln))
        if bulletish_count >= 2:
            # Preserve each bullet as its own unit.
            units.extend(lines)
        else:
            # Treat hard line breaks as spaces.
            units.append(" ".join(lines))

    return units


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> list[str]:
    """Robust chunking: paragraph-aware, bullet-aware, with overlap."""
    units = _split_into_units(text)
    if not units:
        return []

    if chunk_size <= 0:
        return units

    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))
    step = max(1, chunk_size - chunk_overlap)

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        # --- oversized unit ---
        if len(unit) > chunk_size:
            # Try sentence split first.
            sentences = SENTENCE_SPLIT_RE.split(unit)
            if len(sentences) > 1:
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = f"{current} {sent}".strip()
                    else:
                        flush()
                        if len(sent) <= chunk_size:
                            current = sent
                        else:
                            for start in range(0, len(sent), step):
                                chunks.append(sent[start : start + chunk_size].strip())
                            current = ""
                continue

            # Hard fallback: sliding window on the entire unit.
            flush()
            for start in range(0, len(unit), step):
                chunks.append(unit[start : start + chunk_size].strip())
            continue

        # --- normal accumulation ---
        if not current:
            current = unit
            continue

        if len(current) + 1 + len(unit) <= chunk_size:
            current = f"{current} {unit}"
        else:
            prev = current
            flush()
            if chunk_overlap > 0:
                tail = prev[-chunk_overlap:]
                current = f"{tail} {unit}".strip()
            else:
                current = unit

    flush()
    return chunks
