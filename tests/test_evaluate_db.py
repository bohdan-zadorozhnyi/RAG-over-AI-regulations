"""Eval end to end on the topic corpus: question storage, the matrix, the
eval_* rows and both output files. Skipped without Postgres."""

import json
from pathlib import Path

import pytest

from regrag.config import settings
from regrag.db import Connection
from regrag.encoder import Vectors
from regrag.evaluate import Question, run, store_questions
from tests.topic_corpus import TEST_CELEX, index_corpus, topic_encoder, use_test_settings

QUESTIONS = f"""
- id: t-dl
  category: direct_lookup
  question: What does Article 3 say about gamma?
  expected_articles: ["{TEST_CELEX}:3"]
  expected_behaviour: answer
- id: t-ob
  category: obligation
  question: What primary obligations apply?
  expected_articles: ["{TEST_CELEX}:1"]
  expected_behaviour: answer
- id: t-cd
  category: cross_document
  question: How do alpha and delta interact?
  expected_articles: ["{TEST_CELEX}:1", "{TEST_CELEX}:4"]
  expected_behaviour: answer
- id: t-oos
  category: out_of_scope
  question: What is the corporate tax rate in Poland?
  expected_articles: []
  expected_behaviour: refuse
  tier: A
"""

COUNT_ROWS = """
SELECT (SELECT count(*) FROM eval_runs r WHERE r.id = ANY(%(ids)s)),
       (SELECT count(*) FROM eval_results e WHERE e.run_id = ANY(%(ids)s))
"""


@pytest.fixture(autouse=True)
def test_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_test_settings(monkeypatch)
    path = tmp_path / "questions.yaml"
    path.write_text(QUESTIONS)
    monkeypatch.setattr(settings, "eval_questions_path", path)
    monkeypatch.setattr(settings, "eval_questions_per_category", 1)


@pytest.fixture
def indexed(conn: Connection) -> Connection:
    return index_corpus(conn)


def test_store_questions_upserts(indexed: Connection) -> None:
    q = Question("t-x", "direct_lookup", "First wording?", (f"{TEST_CELEX}:2",))
    store_questions(indexed, [q])
    store_questions(
        indexed, [Question("t-x", "direct_lookup", "Second wording?", q.expected_articles)]
    )
    rows = indexed.execute("SELECT question FROM eval_questions WHERE id = 't-x'").fetchall()
    assert rows == [("Second wording?",)]


def test_store_questions_rejects_articles_not_in_the_corpus(indexed: Connection) -> None:
    typo = Question("t-typo", "direct_lookup", "Q?", (f"{TEST_CELEX}:31",))
    recital = Question("t-rec", "direct_lookup", "Q?", (f"{TEST_CELEX}:Recital 1",))
    with pytest.raises(ValueError) as error:
        store_questions(indexed, [typo, recital])
    assert f"t-typo: {TEST_CELEX}:31" in str(error.value)
    assert f"t-rec: {TEST_CELEX}:Recital 1" in str(error.value)


def test_run_scores_the_matrix_and_records_it(indexed: Connection, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def counting_encoder(texts: list[str]) -> Vectors:
        calls.append(texts)
        return topic_encoder(texts)

    report, metrics = tmp_path / "eval-report.md", tmp_path / "eval-metrics.json"
    runs = run(report_path=report, metrics_path=metrics, conn=indexed, encoder=counting_encoder)

    assert len(runs) == 12
    assert len(calls) == 1 and len(calls[0]) == 4  # the questions, encoded once for all 12

    shipped = next(r for r in runs if r.params.label == "section_aware, recitals, norm 0")
    by_id = {r.question.id: r for r in shipped.results}
    assert by_id["t-dl"].hit_at_5 and by_id["t-ob"].hit_at_5
    assert by_id["t-cd"].recall_at_5 == 1.0
    assert by_id["t-oos"].first_hit_rank is None

    ids = indexed.execute("SELECT id FROM eval_runs ORDER BY id DESC LIMIT 12").fetchall()
    row = indexed.execute(COUNT_ROWS, {"ids": [i for (i,) in ids]}).fetchone()
    assert row == (12, 12 * 4)

    data = json.loads(metrics.read_text())
    assert data["n_questions"] == 4 == data["expected_questions"]
    assert data["config"] == "section_aware, recitals, norm 0"
    assert len(data["by_config"]) == 12
    assert "## hit@5 across the matrix" in report.read_text()


def test_run_refuses_an_incomplete_set_unless_allowed(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "eval_questions_per_category", 10)
    with pytest.raises(ValueError, match="10 questions per category"):
        run(conn=indexed, encoder=topic_encoder)

    runs = run(matrix=False, allow_incomplete=True, conn=indexed, encoder=topic_encoder)
    assert len(runs) == 1
