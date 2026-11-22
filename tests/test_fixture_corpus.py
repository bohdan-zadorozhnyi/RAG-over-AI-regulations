"""The checked-in fixture corpus still covers the eval set (no database needed)."""

from regrag.config import settings
from regrag.evaluate import load_questions
from tests.fixtures.build_index import CHECKSUM_PREFIX, load_corpus


def test_fixture_holds_every_expected_provision() -> None:
    corpus = load_corpus()
    present = {
        f"{document.celex}:{section.article_no}"
        for document, sections in corpus
        for section in sections
        if section.article_no is not None
    }
    expected = {
        a for q in load_questions(settings.eval_questions_path) for a in q.expected_articles
    }
    assert expected <= present, f"re-run export_corpus.py; missing {sorted(expected - present)}"
    assert {document.celex for document, _ in corpus} == set(settings.celex_ids)
    assert all(document.checksum.startswith(CHECKSUM_PREFIX) for document, _ in corpus)
