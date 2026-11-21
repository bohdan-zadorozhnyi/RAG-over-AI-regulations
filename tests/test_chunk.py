"""Unit tests for chunk. Pure functions use a word-count counter; the tests of
the real tokenizer are skipped when it cannot be loaded (offline, not cached)."""

from dataclasses import replace

import pytest

from regrag.chunk import _tokenizer, build_header, count_tokens, recursive_baseline, section_aware
from regrag.config import settings
from regrag.models import Section, SectionKind


def tokenizer_available() -> bool:
    try:
        _tokenizer(settings.embedding_model)
    except OSError:  # no network and nothing in the Hugging Face cache
        return False
    return True


needs_tokenizer = pytest.mark.skipif(
    not tokenizer_available(), reason="embedding tokenizer not downloadable or cached"
)


@needs_tokenizer
def test_empty_text_has_no_tokens() -> None:
    assert count_tokens("") == 0  # special tokens are excluded


@needs_tokenizer
def test_counts_are_additive_over_words() -> None:
    assert count_tokens("risk management system") > count_tokens("risk")


@needs_tokenizer
def test_legal_text_is_denser_than_words() -> None:
    # Citations split into many pieces: tokens outnumber whitespace words,
    # which is why a word count would understate chunk size.
    text = "Regulation (EU) 2016/679 and Directive (EU) 2022/2555, Article 6(1)(f)."
    assert count_tokens(text) > len(text.split())


# --- shared helpers -------------------------------------------------------

AI_ACT = "32024R1689"


def words(text: str) -> int:
    """Word-count stand-in for the tokenizer: deterministic, no download."""
    return len(text.split())


def article(
    path: str, text: str, heading: str | None = "Duties", para: str | None = "2"
) -> Section:
    article_no = path.split("article-")[1].split("/")[0]
    return Section(path, SectionKind.ARTICLE, article_no, para, heading, text)


# --- build_header ---------------------------------------------------------


def test_header_article_paragraph_with_chapter_and_section() -> None:
    s = article("chapter-III/section-2/article-9/paragraph-2", "x", "Risk management system")
    assert build_header(s, AI_ACT) == (
        "AI Act, Chapter III, Section 2, Article 9: Risk management system, paragraph 2"
    )


def test_header_definition_point_keeps_parentheses() -> None:
    s = Section("chapter-I/article-3/point-1", SectionKind.ARTICLE, "3", "(1)", "Definitions", "x")
    assert build_header(s, AI_ACT) == "AI Act, Chapter I, Article 3: Definitions, point (1)"


def test_header_article_without_heading_or_paragraphs() -> None:
    s = Section("chapter-I/article-4", SectionKind.ARTICLE, "4", None, None, "x")
    assert build_header(s, AI_ACT) == "AI Act, Chapter I, Article 4"


def test_header_recital() -> None:
    s = Section("recital-12", SectionKind.RECITAL, None, None, None, "x")
    assert build_header(s, AI_ACT) == "AI Act, Recital 12"


def test_header_annex_point_drops_trailing_dot() -> None:
    s = Section("annex-III/point-5", SectionKind.ANNEX, "Annex III", "5.", "High-risk systems", "x")
    assert build_header(s, AI_ACT) == "AI Act, Annex III: High-risk systems, point 5"


def test_header_annex_group_point_and_whole_annex() -> None:
    grouped = Section("annex-I/group-2/point-13", SectionKind.ANNEX, "Annex I", "13.", "List", "x")
    whole = Section("annex-VI", SectionKind.ANNEX, "Annex VI", None, "Internal control", "x")
    assert build_header(grouped, AI_ACT) == "AI Act, Annex I: List, point 13"
    assert build_header(whole, AI_ACT) == "AI Act, Annex VI: Internal control"


def test_header_uses_document_title() -> None:
    s = Section("chapter-II/article-9/paragraph-1", SectionKind.ARTICLE, "9", "1", "Special", "x")
    assert build_header(s, "32016R0679").startswith("GDPR, Chapter II, Article 9")


# --- section_aware --------------------------------------------------------

LONG_PARAGRAPH = "\n".join(
    [
        "1. The following practices shall be prohibited:",
        "(a) " + "alpha " * 20,
        "  (i) " + "nested " * 5,
        "(b) " + "beta " * 20,
        "(c) " + "gamma " * 20,
        "(d) " + "delta " * 20,
    ]
)


def test_short_section_is_one_chunk() -> None:
    s = article("chapter-I/article-2/paragraph-2", "2. Providers shall comply.")
    [chunk] = section_aware([s], AI_ACT, words, max_tokens=100)
    assert chunk.text == s.text
    assert chunk.header.startswith("AI Act, Chapter I, Article 2")
    assert chunk.token_count == words(f"{chunk.header}\n{chunk.text}")
    assert (chunk.section_path, chunk.ordinal, chunk.covers) == (s.path, 0, (s.path,))


def test_long_section_splits_between_points_and_repeats_lead_in() -> None:
    s = article("chapter-II/article-5/paragraph-1", LONG_PARAGRAPH)
    chunks = section_aware([s], AI_ACT, words, max_tokens=70)

    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        lines = chunk.text.split("\n")
        assert lines[0] == "1. The following practices shall be prohibited:"
        assert lines[1].startswith("(")  # never starts on an orphaned sub-line
        assert chunk.token_count <= 70
    joined = "\n".join(c.text for c in chunks)
    for point in ("(a)", "(b)", "(c)", "(d)"):
        assert joined.count(point) == 1  # nothing lost, nothing duplicated


def test_sub_lines_stay_with_their_point() -> None:
    s = article("chapter-II/article-5/paragraph-1", LONG_PARAGRAPH)
    chunks = section_aware([s], AI_ACT, words, max_tokens=70)
    holder = next(c for c in chunks if "(a)" in c.text)
    assert "(i) nested" in holder.text


def test_overlong_line_splits_into_sentences() -> None:
    line = " ".join(f"Sentence {n} has five words." for n in range(20))
    s = article("chapter-I/article-7/paragraph-1", f"1. Intro:\n{line}")
    chunks = section_aware([s], AI_ACT, words, max_tokens=40)
    assert all(c.token_count <= 40 for c in chunks)
    assert all(c.text.split("\n")[-1].endswith(".") for c in chunks)  # sentence breaks


def test_overlong_sentence_falls_back_to_words() -> None:
    s = article("chapter-I/article-8/paragraph-1", "word " * 200)
    chunks = section_aware([s], AI_ACT, words, max_tokens=40)
    assert len(chunks) > 1
    assert all(c.token_count <= 40 for c in chunks)


def test_hash_is_stable_and_content_sensitive() -> None:
    s = article("chapter-I/article-2/paragraph-2", "2. Providers shall comply.")
    [first] = section_aware([s], AI_ACT, words, 100)
    [again] = section_aware([s], AI_ACT, words, 100)
    edited = replace(s, text="2. Providers shall not comply.")
    [changed] = section_aware([edited], AI_ACT, words, 100)
    assert first.content_hash == again.content_hash
    assert first.content_hash != changed.content_hash


# --- recursive_baseline ---------------------------------------------------


def doc_sections() -> list[Section]:
    return [
        Section("recital-1", SectionKind.RECITAL, None, None, None, "(1) " + "why " * 30),
        article("chapter-I/article-1/paragraph-1", "1. " + "scope " * 30, "Scope", "1"),
        article("chapter-I/article-2/paragraph-1", "1. " + "duty " * 30, "Duties", "1"),
    ]


def test_baseline_chunks_fit_and_have_no_header() -> None:
    chunks = recursive_baseline(doc_sections(), words, chunk_size=25, overlap=5)
    assert len(chunks) > 3
    assert all(c.header == "" and c.token_count == words(c.text) <= 25 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_baseline_covers_every_overlapped_section() -> None:
    # Short sections, as most real ones are: the splitter merges several into
    # one chunk, so chunks straddle section boundaries.
    sections = [
        article(f"chapter-I/article-{n}/paragraph-1", f"{n}. " + f"text{n} " * 8, "T", "1")
        for n in range(1, 7)
    ]
    texts = {s.path: s.text for s in sections}
    chunks = recursive_baseline(sections, words, chunk_size=20, overlap=3)

    for chunk in chunks:
        assert chunk.section_path == chunk.covers[0]
        covered = "\n\n".join(texts[p] for p in chunk.covers)
        assert chunk.text in covered  # the text really comes from those sections
    assert any(len(c.covers) > 1 for c in chunks)  # some chunks straddle sections
    assert {p for c in chunks for p in c.covers} == set(texts)


def test_baseline_positions_survive_token_overlap() -> None:
    # Regression: with a token length function, LangChain's add_start_index
    # mislocated chunks (-1) and credited them to the first section.
    sections = [
        Section(f"recital-{n}", SectionKind.RECITAL, None, None, None, f"({n}) " + f"r{n} " * 40)
        for n in range(1, 8)
    ]
    chunks = recursive_baseline(sections, words, chunk_size=30, overlap=10)
    last = chunks[-1]
    assert "recital-7" in last.covers and "recital-1" not in last.covers
