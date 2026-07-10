"""Tests for rag.index — config, cache helpers, fingerprinting."""

import json
import os
import tempfile

import pytest

from rag.index import RagConfig, _cache_paths, _collect_file_fingerprints
from rag.utils import norm_path, safe_mkdir, file_content_hash


# ---------------------------------------------------------------------------
# _cache_paths
# ---------------------------------------------------------------------------

class TestCachePaths:
    def test_returns_expected_keys(self):
        paths = _cache_paths("/tmp/test_cache")
        assert "index" in paths
        assert "chunks" in paths
        assert "meta" in paths
        assert "info" in paths
        for p in paths.values():
            assert "test_cache" in p

    def test_paths_are_under_cache_dir(self):
        paths = _cache_paths("my_cache")
        for p in paths.values():
            assert p.startswith("my_cache")


# ---------------------------------------------------------------------------
# norm_path
# ---------------------------------------------------------------------------

class TestNormPath:
    def test_normalizes_separators(self):
        p = norm_path("./documents")
        assert "/" in p
        assert "\\\\" not in p  # no double backslashes

    def test_absolute(self):
        p = norm_path("relative/path")
        assert os.path.isabs(p.replace("/", os.sep))

    def test_consistent(self):
        assert norm_path("./a/../b") == norm_path("b")


# ---------------------------------------------------------------------------
# safe_mkdir
# ---------------------------------------------------------------------------

class TestSafeMkdir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "sub", "dir")
            safe_mkdir(target)
            assert os.path.isdir(target)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "sub")
            safe_mkdir(target)
            safe_mkdir(target)  # should not raise
            assert os.path.isdir(target)


# ---------------------------------------------------------------------------
# file_content_hash
# ---------------------------------------------------------------------------

class TestFileContentHash:
    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            f.flush()
            path = f.name
        try:
            h1 = file_content_hash(path)
            h2 = file_content_hash(path)
            assert h1 == h2
            assert len(h1) == 64  # SHA-256 hex
        finally:
            os.unlink(path)

    def test_different_content(self):
        paths = []
        for content in [b"aaa", b"bbb"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(content)
                f.flush()
                paths.append(f.name)
        try:
            assert file_content_hash(paths[0]) != file_content_hash(paths[1])
        finally:
            for p in paths:
                os.unlink(p)


# ---------------------------------------------------------------------------
# RagConfig
# ---------------------------------------------------------------------------

class TestRagConfig:
    def test_defaults(self):
        config = RagConfig()
        assert config.documents_dir == "documents"
        assert config.index_metric == "cosine"
        assert config.rerank is True
        assert config.use_bm25 is True
        assert config.index_type == "auto"

    def test_frozen(self):
        config = RagConfig()
        with pytest.raises(Exception):
            config.chunk_size = 999  # type: ignore[misc]

    def test_custom_values(self):
        config = RagConfig(
            documents_dir="my_docs",
            chunk_size=1000,
            embedding_model="custom-model",
            use_bm25=False,
        )
        assert config.documents_dir == "my_docs"
        assert config.chunk_size == 1000
        assert config.embedding_model == "custom-model"
        assert config.use_bm25 is False


# ---------------------------------------------------------------------------
# _collect_file_fingerprints
# ---------------------------------------------------------------------------

class TestCollectFileFingerprints:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fps = _collect_file_fingerprints(tmpdir)
            assert fps == []

    def test_detects_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("hello")
            fps = _collect_file_fingerprints(tmpdir)
            assert len(fps) == 1
            assert fps[0]["path"] == "test.txt"
            assert "hash" in fps[0]
            assert "size" in fps[0]

    def test_fingerprint_changes_on_content_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")

            with open(path, "w") as f:
                f.write("version 1")
            fps1 = _collect_file_fingerprints(tmpdir)

            with open(path, "w") as f:
                f.write("version 2")
            fps2 = _collect_file_fingerprints(tmpdir)

            assert fps1[0]["hash"] != fps2[0]["hash"]
