"""Local extractive answer generation — no API calls, no network."""

from __future__ import annotations

import re

from .chunker import SENTENCE_SPLIT_RE
from .reranker import tokenize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEF_QUESTION_RE = re.compile(
    r"^\s*(?:what\s+is|what's|define|definition\s+of|meaning\s+of|stands\s+for)"
    r"\s+(?P<term>.+?)\s*(?:\?|\.|$)",
    re.IGNORECASE,
)


def _clean_ws(text: str) -> str:
    """Normalize whitespace artifacts from PDF extraction."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[\u2028\u2029]", "\n", text)  # unicode line separators
    text = re.sub(r"-\s*\n\s*", "", text)  # de-hyphenate across breaks
    text = re.sub(r"(?<=\w)\s*\n\s*(?=\w)", "", text)  # join word breaks
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Patterns that flag boilerplate / legal / copyright / disclaimer text
_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"no part of this publication|all rights reserved"
    r"|may not be reproduced|may be reproduced"
    r"|prior written (?:permission|consent)"
    r"|without the (?:express|prior|written)"
    r"|copyright \xa9|\u00a9\s*\d{4}|\u00a9[^\n]{0,40}(?:19|20)\d{2}"
    r"|disclaimer|proprietary and confidential"
    r"|trademark|registered trademark"
    r"|terms and conditions|terms of use|license agreement"
    # single-user / distribution licensing notices (e.g. ISO/IEC cover pages)
    r"|licen[sc]ed to\b|single[- ]user licen[sc]e|copying and networking prohibited"
    r"|(?:iso )?store order|downloaded:\s*\d{4}-\d{2}-\d{2}"
    r"|table of contents|list of (?:figures|tables)"
    r"|page \d+ of \d+|printed on|document control"
    r"|this document is|for internal use only"
    r")",
    re.IGNORECASE,
)

# Explanatory verbs for "what is X" style questions — much broader coverage
_EXPLANATORY_RE = re.compile(
    r"\b(?:"
    r"is a|is an|is the|are the|refers to|defined as|stands for"
    r"|means|provides|enables|helps|supports|includes|covers"
    r"|ensures|establishes|addresses|designed to|aims to"
    r"|focuses on|developed (?:by|to|for)|created (?:by|to|for)"
    r"|offers|delivers|manages|framework|standard|certification"
    r"|organization|set of|collection of|approach to"
    r")\b",
    re.IGNORECASE,
)


def _is_boilerplate(sentence: str) -> bool:
    """Return True for copyright, legal, or boilerplate text."""
    return bool(_BOILERPLATE_RE.search(sentence))


def _looks_like_title_or_junk(sentence: str) -> bool:
    """Return True for headings / extraction fragments that hurt answer quality."""
    s = sentence.strip()
    if len(s) < 30:
        return True
    # Boilerplate / legal text
    if _is_boilerplate(s):
        return True
    # Document title / version headers (e.g., "Version 11.0.0 HITRUST CSF PDF...")
    if re.match(r"(?:version\s+\d|v\d+\.\d+)", s, re.IGNORECASE):
        return True
    # Heavy ALL-CAPS (>40% uppercase letters = likely a title/header)
    alpha_chars = [c for c in s if c.isalpha()]
    if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.4:
        return True
    if (
        re.fullmatch(r"[A-Za-z0-9 .,:;()\-]{1,60}", s)
        and re.search(r"\b\d+\b", s)
        and not re.search(r"[.!?]", s[2:])
    ):
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    if letters < max(10, len(s) // 4):
        return True
    # Regulatory/standard reference lists (dense acronyms, section numbers, no verbs)
    if _is_reference_list(s):
        return True
    return False


def _is_reference_list(sentence: str) -> bool:
    """Detect dense regulatory/standard reference lists that lack explanatory value.

    Examples of junk:
      "Level 1 Regulatory Factors: DirectTrust HITRUST De-ID Framework NY OHIP ..."
      "FedRAMP MA-2 FedRAMP PE-2 FedRAMP PE-3 HIPAA Security Rule § 164.308..."
    """
    s = sentence.strip()
    # Count section/clause references like "§ 164.308(a)", "MA-2", "v3.1", "5.1(c)"
    ref_tokens = re.findall(
        r"§\s*\d+|v\d+\.\d+|\b[A-Z]{2,}-\d+|\b\d+\.\d+\.\d+|\b\d+\s*CFR\s*\d+"
        r"|\b\d+\.\d+\([a-z]\)|\b\d{4,5}:\d{4}",
        s,
    )
    words = re.findall(r"\b[A-Za-z]+\b", s)
    if not words:
        return True
    # If >30% of content is reference tokens, it's a reference list
    ref_ratio = len(ref_tokens) / max(1, len(words))
    if ref_ratio > 0.15 and len(ref_tokens) >= 3:
        return True
    # Detect long runs of proper nouns / acronyms with no verbs
    # Common verbs that indicate explanatory text
    has_verb = bool(re.search(
        r"\b(?:is|are|was|were|has|have|shall|should|must|can|will|may|"
        r"provide|ensure|require|implement|maintain|establish|include|"
        r"define|describe|apply|use|manage|protect|control|verify)\b",
        s, re.IGNORECASE,
    ))
    # Count capitalized word sequences (proper nouns / titles)
    caps_words = len(re.findall(r"\b[A-Z][A-Za-z]*\b", s))
    caps_ratio = caps_words / max(1, len(words))
    if not has_verb and caps_ratio > 0.5 and len(words) > 8:
        return True
    return False


def _information_density(sentence: str) -> float:
    """Score 0-1 for how information-rich a sentence is.

    Penalizes very short, very long, or stopword-heavy sentences.
    """
    words = re.findall(r"[A-Za-z0-9]+", sentence.lower())
    if len(words) < 5:
        return 0.2
    from .reranker import STOPWORDS
    content_words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    ratio = len(content_words) / max(1, len(words))
    # Sweet spot is 10-40 words; penalize extremes
    length_factor = 1.0
    if len(words) > 60:
        length_factor = 0.7
    elif len(words) < 8:
        length_factor = 0.6
    return min(1.0, ratio * 1.3) * length_factor


def _ensure_sentence_punct(s: str) -> str:
    """Append a period when terminal punctuation is missing."""
    s = s.strip()
    if not s:
        return s
    return s if s[-1] in ".!?" else s + "."


def _definition_term(question: str) -> str | None:
    m = _DEF_QUESTION_RE.match(question.strip())
    if not m:
        return None
    term = str(m.group("term") or "").strip()
    term = term.strip("\"'""''` ")
    term = re.sub(r"\s+", " ", term)
    term = term.strip(" ?.!,:;()[]{}")
    return term or None


def _extract_definition_candidates(*, term: str, content: str) -> list[str]:
    """High-precision extraction of definition-like sentences for *term*."""
    if not term:
        return []

    term_re = re.escape(term)
    exp1 = re.compile(
        rf"\b(?P<long>[A-Za-z][A-Za-z0-9 /\-&]{{3,80}}?)\s*\(\s*(?P<acro>{term_re})\s*\)",
        re.IGNORECASE,
    )
    exp2 = re.compile(
        rf"\b(?P<acro>{term_re})\s*\(\s*(?P<long>[A-Za-z][A-Za-z0-9 /\-&]{{3,80}}?)\s*\)",
        re.IGNORECASE,
    )

    picked: list[str] = []
    cleaned = _clean_ws(content)

    for m in exp1.finditer(cleaned):
        long_form = _clean_ws(m.group("long") or "")
        acro = _clean_ws(m.group("acro") or term)
        if long_form and acro:
            picked.append(f"{acro.upper()} stands for {long_form}.")

    for m in exp2.finditer(cleaned):
        long_form = _clean_ws(m.group("long") or "")
        acro = _clean_ws(m.group("acro") or term)
        if long_form and acro:
            picked.append(f"{acro.upper()} stands for {long_form}.")

    definers = re.compile(
        rf"\b{term_re}\b\s*(?:is|means|refers\s+to|stands\s+for|defined\s+as)\b",
        re.IGNORECASE,
    )
    for sent in SENTENCE_SPLIT_RE.split(cleaned):
        sent = _clean_ws(sent)
        if not sent:
            continue
        if definers.search(sent):
            picked.append(_ensure_sentence_punct(sent))

    out: list[str] = []
    seen: set[str] = set()
    for s in picked:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_boilerplate_results(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove results whose content is predominantly boilerplate/copyright text.

    Call this *before* passing results to ``local_extractive_answer`` or
    ``build_digest`` so they have high-quality passages to work with.
    """
    cleaned: list[dict[str, object]] = []
    for r in results:
        content = str(r.get("content", ""))
        # Check if the majority of meaningful sentences are boilerplate
        sentences = SENTENCE_SPLIT_RE.split(_clean_ws(content))
        total = 0
        boilerplate_count = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            total += 1
            if _is_boilerplate(sent):
                boilerplate_count += 1
        # Skip if >50% of sentences are boilerplate
        if total > 0 and boilerplate_count / total > 0.5:
            continue
        cleaned.append(r)
    # Re-number ranks
    for i, r in enumerate(cleaned, start=1):
        r["rank"] = i
    return cleaned


def local_extractive_answer(
    *,
    question: str,
    results: list[dict[str, object]],
    max_sentences: int = 4,
    format_style: str = "paragraph",
) -> str:
    """Build a local-only answer by selecting best-matching sentences.

    Returns formatted text with inline ``(source pN)`` citations.
    """
    q_tokens = set(tokenize(question))
    if not q_tokens:
        q_tokens = set(re.findall(r"[A-Za-z0-9]+", question.lower()))

    def_term = _definition_term(question)

    scored: list[tuple[float, str, str]] = []
    is_definitional = def_term is not None

    for r in results:
        content = str(r.get("content", ""))
        content = _clean_ws(content)
        source = r.get("source", "?")
        page = r.get("page", "?")
        citation = f"({source} p{page})"
        rank = int(r.get("rank", 999))
        rank_bonus = 1.0 / max(1, rank)

        # --- definition candidates (big boost) ---
        if def_term:
            for cand in _extract_definition_candidates(term=def_term, content=content):
                scored.append(
                    (1000.0 + rank_bonus, _ensure_sentence_punct(_clean_ws(cand)), citation)
                )

        # --- sentence-level scoring ---
        sentences = SENTENCE_SPLIT_RE.split(content)
        n_sentences = len(sentences)
        for idx, sent in enumerate(sentences):
            sent = _clean_ws(sent)
            if _looks_like_title_or_junk(sent):
                continue
            s_tokens = set(tokenize(sent))
            overlap = len(q_tokens & s_tokens)
            if overlap == 0:
                continue

            # --- base overlap score ---
            score = overlap * 1.5 + rank_bonus

            # --- information density bonus ---
            density = _information_density(sent)
            score *= (0.5 + density)  # range 0.5-1.5x

            # --- position bonus: earlier sentences in a chunk are usually more informative ---
            position_bonus = 1.0 + 0.3 * max(0.0, 1.0 - idx / max(1, n_sentences))
            score *= position_bonus

            # --- definitional question bonuses ---
            if is_definitional and def_term:
                term_pat = re.escape(def_term)
                # Sentence mentions the term
                if re.search(rf"\b{term_pat}\b", sent, re.IGNORECASE):
                    score += 1.5
                    # Sentence *explains* the term (much stronger signal)
                    if _EXPLANATORY_RE.search(sent):
                        score += 4.0
                    # Even stronger: "TERM is a ..." pattern
                    if re.search(
                        rf"\b{term_pat}\b\s+(?:is|are|was|provides|enables|helps)",
                        sent, re.IGNORECASE,
                    ):
                        score += 3.0
            elif _EXPLANATORY_RE.search(sent):
                # Non-definitional questions still benefit slightly from explanatory sentences
                score += 0.5

            scored.append((score, sent, citation))

    if not scored:
        # For definitional questions, synthesize from source metadata
        if def_term and results:
            sources = set()
            for r in results:
                s = str(r.get("source", ""))
                if s:
                    sources.add(s)
            if sources:
                src_list = ", ".join(sorted(sources))
                return (
                    f"The documents don't contain an explicit definition of "
                    f"\"{def_term}\", but it is referenced extensively in: {src_list}. "
                    f"Try asking a more specific question about {def_term} "
                    f"(e.g., requirements, controls, or implementation)."
                )
        return "I couldn't find enough information in the documents to answer that."

    scored.sort(key=lambda x: x[0], reverse=True)

    picked: list[str] = []
    seen: set[str] = set()
    for _, sent, cit in scored:
        sent = _ensure_sentence_punct(_clean_ws(sent))
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(f"{sent} {cit}")
        if len(picked) >= max_sentences:
            break

    if format_style == "lines":
        return "\n".join(picked)
    return " ".join(picked)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_sources(results: list[dict[str, object]]) -> str:
    """Deduplicate sources for compact display."""
    seen: set[tuple[object, object]] = set()
    parts: list[str] = []
    for r in results:
        key = (r.get("source"), r.get("page"))
        if key not in seen:
            seen.add(key)
            parts.append(f"- {key[0]} (Page {key[1]})")
    return "\n".join(parts)


def print_results(*, results: list[dict[str, object]], max_chars: int) -> None:
    """Print retrieved passages for human inspection."""
    for item in results:
        rank = item.get("rank", "?")
        source = item.get("source", "?")
        page = item.get("page", "?")
        content = str(item.get("content", ""))

        print(f"[{rank}] Source: {source} (Page {page})")
        if max_chars <= 0 or len(content) <= max_chars:
            print(f"    Content: {content}")
        else:
            print(f"    Content: {content[:max_chars]}...")


# ---------------------------------------------------------------------------
# Summary / Digest
# ---------------------------------------------------------------------------

def build_digest(
    *,
    question: str,
    results: list[dict[str, object]],
    max_sentences_per_source: int = 3,
    max_total_sentences: int = 8,
) -> list[dict[str, object]]:
    """Build a per-source digest of the most relevant sentences.

    Returns a list of dicts, each with:
      - ``source``: filename
      - ``pages``: sorted list of page numbers
      - ``key_points``: list of the best sentences (strings)
    """
    from collections import defaultdict

    q_tokens = set(tokenize(question))
    if not q_tokens:
        q_tokens = set(re.findall(r"[A-Za-z0-9]+", question.lower()))

    # Group chunks by source file.
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in results:
        source = str(r.get("source", "unknown"))
        by_source[source].append(r)

    total_picked = 0
    digest_entries: list[dict[str, object]] = []

    for source, chunks in by_source.items():
        pages: set[int] = set()
        scored_sents: list[tuple[float, str]] = []

        for chunk in chunks:
            pages.add(int(chunk.get("page", 0)))
            content = _clean_ws(str(chunk.get("content", "")))
            rank = int(chunk.get("rank", 999))
            rank_bonus = 1.0 / max(1, rank)

            sentences = SENTENCE_SPLIT_RE.split(content)
            n_sentences = len(sentences)
            for idx, sent in enumerate(sentences):
                sent = _clean_ws(sent)
                if _looks_like_title_or_junk(sent):
                    continue
                s_tokens = set(tokenize(sent))
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    continue
                density = _information_density(sent)
                position_bonus = 1.0 + 0.3 * max(0.0, 1.0 - idx / max(1, n_sentences))
                score = (overlap * 1.5 + rank_bonus) * (0.5 + density) * position_bonus
                if _EXPLANATORY_RE.search(sent):
                    score += 1.0
                scored_sents.append((score, _ensure_sentence_punct(sent)))

        if not scored_sents:
            continue

        scored_sents.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate and pick top sentences for this source.
        seen: set[str] = set()
        key_points: list[str] = []
        for _, sent in scored_sents:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            key_points.append(sent)
            total_picked += 1
            if len(key_points) >= max_sentences_per_source:
                break
            if total_picked >= max_total_sentences:
                break

        digest_entries.append({
            "source": source,
            "pages": sorted(pages - {0}),
            "key_points": key_points,
        })

        if total_picked >= max_total_sentences:
            break

    return digest_entries


def format_digest(digest: list[dict[str, object]]) -> str:
    """Format digest entries into a human-readable string."""
    if not digest:
        return "No relevant information found."

    parts: list[str] = []
    for entry in digest:
        source = entry["source"]
        pages = entry.get("pages", [])
        key_points = entry.get("key_points", [])
        page_str = ", ".join(str(p) for p in pages) if pages else "?"
        parts.append(f"[{source} — p.{page_str}]")
        for point in key_points:
            parts.append(f"  • {point}")
    return "\n".join(parts)
