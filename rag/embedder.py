"""Embedding backend wrapper — unifies fastembed and sentence-transformers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Unified interface for *fastembed* and *sentence-transformers* backends."""

    def __init__(self, backend: str, model_name: str) -> None:
        self.backend = backend
        self.model_name = model_name

        if backend == "fastembed":
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name)
        elif backend in {"sentence-transformers", "sentence_transformers"}:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    def encode(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed *texts* → ``float32`` numpy array suitable for FAISS."""
        if self.backend == "fastembed":
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None  # type: ignore[assignment]

            vectors: list[object] = []
            gen = self._model.embed(texts)
            if show_progress and tqdm is not None:
                gen = tqdm(gen, total=len(texts), desc="Embedding", unit="chunk")
            elif show_progress:
                total = len(texts)
                for i, vec in enumerate(gen, start=1):
                    vectors.append(vec)
                    if total > 0 and (i % 256 == 0 or i == total):
                        logger.info("Embedded %d/%d chunks...", i, total)
                return np.asarray(vectors, dtype=np.float32)

            for vec in gen:
                vectors.append(vec)
            return np.asarray(vectors, dtype=np.float32)

        # sentence-transformers has a built-in progress bar.
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)


def pick_default_backend() -> str:
    """Auto-select fastembed (lighter) if available, else sentence-transformers."""
    try:
        import fastembed  # noqa: F401

        return "fastembed"
    except Exception:
        return "sentence-transformers"
