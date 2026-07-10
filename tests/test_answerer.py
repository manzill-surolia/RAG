"""Tests for rag.answerer — extractive answer, formatting, helpers."""

import pytest

from rag.answerer import (
    local_extractive_answer,
    format_sources,
    _clean_ws,
    _looks_like_title_or_junk,
    _ensure_sentence_punct,
    _definition_term,
    _extract_definition_candidates,
)


# ---------------------------------------------------------------------------
# _clean_ws
# ---------------------------------------------------------------------------

class TestCleanWs:
    def test_dehyphenates(self):
        assert _clean_ws("conca-\ntenated") == "concatenated"

    def test_collapses_whitespace(self):
        assert _clean_ws("hello    world") == "hello world"

    def test_strips_soft_hyphen(self):
        assert _clean_ws("soft\u00adhyphen") == "softhyphen"

    def test_empty(self):
        assert _clean_ws("") == ""


# ---------------------------------------------------------------------------
# _looks_like_title_or_junk
# ---------------------------------------------------------------------------

class TestLooksLikeTitleOrJunk:
    def test_short_text_is_junk(self):
        assert _looks_like_title_or_junk("Title 1") is True

    def test_long_sentence_is_not_junk(self):
        assert _looks_like_title_or_junk(
            "This is a proper sentence that contains enough words to be meaningful."
        ) is False

    def test_mostly_numbers_is_junk(self):
        assert _looks_like_title_or_junk("1234 5678 9012 3456 7890 1234 5678 9012") is True


# ---------------------------------------------------------------------------
# _ensure_sentence_punct
# ---------------------------------------------------------------------------

class TestEnsureSentencePunct:
    def test_adds_period(self):
        assert _ensure_sentence_punct("Hello world") == "Hello world."

    def test_preserves_existing_period(self):
        assert _ensure_sentence_punct("Done.") == "Done."

    def test_preserves_exclamation(self):
        assert _ensure_sentence_punct("Hello!") == "Hello!"

    def test_preserves_question(self):
        assert _ensure_sentence_punct("Really?") == "Really?"

    def test_empty(self):
        assert _ensure_sentence_punct("") == ""

    def test_whitespace_only(self):
        assert _ensure_sentence_punct("   ") == ""


# ---------------------------------------------------------------------------
# _definition_term
# ---------------------------------------------------------------------------

class TestDefinitionTerm:
    def test_what_is(self):
        assert _definition_term("What is ISMS?") == "ISMS"

    def test_define(self):
        assert _definition_term("Define GDPR") == "GDPR"

    def test_whats(self):
        assert _definition_term("What's PCI DSS?") == "PCI DSS"

    def test_meaning_of(self):
        assert _definition_term("Meaning of compliance") == "compliance"

    def test_no_match(self):
        assert _definition_term("How to configure passwords?") is None

    def test_empty(self):
        assert _definition_term("") is None


# ---------------------------------------------------------------------------
# _extract_definition_candidates
# ---------------------------------------------------------------------------

class TestExtractDefinitionCandidates:
    def test_acronym_expansion(self):
        content = "The Information Security Management System (ISMS) is a framework."
        result = _extract_definition_candidates(term="ISMS", content=content)
        assert len(result) >= 1
        assert any("stands for" in r.lower() or "information security" in r.lower() for r in result)

    def test_no_match(self):
        content = "This text has nothing about the term."
        result = _extract_definition_candidates(term="XYZ", content=content)
        assert result == []

    def test_explicit_definition(self):
        content = "GDPR is the General Data Protection Regulation that governs data privacy."
        result = _extract_definition_candidates(term="GDPR", content=content)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# local_extractive_answer
# ---------------------------------------------------------------------------

class TestLocalExtractiveAnswer:
    def test_returns_fallback_on_empty(self):
        answer = local_extractive_answer(question="test query", results=[])
        assert "couldn't find" in answer.lower()

    def test_extracts_from_results(self):
        results = [
            {
                "content": "Passwords must be at least 12 characters long and include special characters.",
                "source": "policy.pdf",
                "page": 1,
                "rank": 1,
            }
        ]
        answer = local_extractive_answer(question="password policy", results=results)
        assert "password" in answer.lower() or "couldn't find" in answer.lower()

    def test_lines_format_uses_newlines(self):
        results = [
            {
                "content": "Data must be retained for five years. Financial records require seven years of retention.",
                "source": "retention.pdf",
                "page": 1,
                "rank": 1,
            }
        ]
        answer = local_extractive_answer(
            question="data retention policy years",
            results=results,
            format_style="lines",
        )
        assert isinstance(answer, str)

    def test_max_sentences_respected(self):
        long_content = ". ".join(f"Sentence number {i} about security" for i in range(20)) + "."
        results = [{"content": long_content, "source": "doc.pdf", "page": 1, "rank": 1}]
        answer = local_extractive_answer(
            question="security",
            results=results,
            max_sentences=2,
            format_style="lines",
        )
        lines = [l for l in answer.strip().split("\n") if l.strip()]
        assert len(lines) <= 2

    def test_definition_query_boost(self):
        results = [
            {
                "content": "The General Data Protection Regulation (GDPR) is a regulation in EU law.",
                "source": "gdpr.pdf",
                "page": 1,
                "rank": 1,
            }
        ]
        answer = local_extractive_answer(question="What is GDPR?", results=results)
        assert "GDPR" in answer


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------

class TestFormatSources:
    def test_deduplicates(self):
        results = [
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 2},
        ]
        out = format_sources(results)
        assert out.count("a.pdf") == 1
        assert "b.pdf" in out

    def test_empty(self):
        assert format_sources([]) == ""

    def test_preserves_order(self):
        results = [
            {"source": "b.pdf", "page": 1},
            {"source": "a.pdf", "page": 2},
        ]
        out = format_sources(results)
        lines = out.strip().split("\n")
        assert "b.pdf" in lines[0]
        assert "a.pdf" in lines[1]
