"""Tests for rag.chunker — text normalization and chunking."""

import pytest

from rag.chunker import chunk_text, normalize_extracted_text, _split_into_units


# ---------------------------------------------------------------------------
# normalize_extracted_text
# ---------------------------------------------------------------------------

class TestNormalizeExtractedText:
    def test_collapses_whitespace(self):
        assert normalize_extracted_text("hello   world") == "hello world"

    def test_normalizes_newlines(self):
        assert normalize_extracted_text("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_triple_newlines(self):
        assert normalize_extracted_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_null_bytes(self):
        assert normalize_extracted_text("he\x00llo") == "he llo"

    def test_empty_input(self):
        assert normalize_extracted_text("") == ""
        assert normalize_extracted_text("   ") == ""

    def test_strips_tabs_and_form_feeds(self):
        assert normalize_extracted_text("a\t\fb") == "a b"


# ---------------------------------------------------------------------------
# _split_into_units
# ---------------------------------------------------------------------------

class TestSplitIntoUnits:
    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph."
        units = _split_into_units(text)
        assert len(units) == 2
        assert units[0] == "First paragraph."
        assert units[1] == "Second paragraph."

    def test_bullet_preservation(self):
        text = "- Item one\n- Item two\n- Item three"
        units = _split_into_units(text)
        assert len(units) == 3
        assert all(u.startswith("- ") for u in units)

    def test_numbered_list(self):
        text = "1) First\n2) Second\n3) Third"
        units = _split_into_units(text)
        assert len(units) == 3

    def test_hard_linebreaks_joined_in_non_list(self):
        text = "This is a long sentence\nthat wraps across lines."
        units = _split_into_units(text)
        assert len(units) == 1
        assert "sentence that" in units[0]

    def test_empty_text(self):
        assert _split_into_units("") == []
        assert _split_into_units("   ") == []

    def test_only_whitespace_paragraphs(self):
        assert _split_into_units("\n\n\n") == []


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_basic_chunking(self):
        text = "A " * 500  # ~1000 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            # Allow some tolerance for boundary effects.
            assert len(c) <= 250

    def test_overlap_carries_context(self):
        text = "word " * 200
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_small_text_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=800, chunk_overlap=80)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_zero_chunk_size_returns_units(self):
        text = "Para one.\n\nPara two."
        chunks = chunk_text(text, chunk_size=0, chunk_overlap=0)
        assert len(chunks) == 2

    def test_negative_overlap_clamped(self):
        text = "Hello world. " * 50
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=-10)
        assert len(chunks) > 0

    def test_oversized_unit_sliding_window(self):
        # A single "word" longer than chunk_size → sliding window.
        text = "A" * 300
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 3
        assert all(len(c) <= 100 for c in chunks)

    def test_sentence_split_in_oversized_unit(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth very long sentence here."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(chunks) >= 2
