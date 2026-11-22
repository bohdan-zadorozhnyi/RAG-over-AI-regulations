"""Eval: question validation, metrics, matrix, outputs. No database."""

import json
from pathlib import Path

import pytest

from regrag.config import config_hash, settings
from regrag.evaluate import (
    CATEGORIES,
    CategoryMetrics,
    Metrics,
    Question,
    QuestionResult,
    RunResult,
    best_run,
    cached_encoder,
    check_complete,
    first_hit_rank,
    load_questions,
    matrix_params,
    recall_at,
    reciprocal_rank,
    score_question,
    separation_auc,
    spread,
    summarise,
    write_metrics_json,
    write_report,
)
from regrag.models import RECURSIVE, SECTION_AWARE, Retrieved, SearchResult, SectionKind
from regrag.retrieve import SearchParams

AI_ACT = "32024R1689"


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text)
    return path


# --- questions ------------------------------------------------------------


def test_load_questions(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
- id: dl-01
  category: direct_lookup
  question: What does Article 9 require?
  expected_articles: ["32024R1689:9", "32024R1689:Annex III"]
  expected_behaviour: answer
  collision: true
  notes: free text, ignored
- id: oos-01
  category: out_of_scope
  question: What is the corporate tax rate in Poland?
  expected_articles: []
  expected_behaviour: scoped
  tier: B
""",
    )
    dl, oos = load_questions(path)
    assert dl == Question(
        "dl-01",
        "direct_lookup",
        "What does Article 9 require?",
        (f"{AI_ACT}:9", f"{AI_ACT}:Annex III"),
        expected_behaviour="answer",
        collision=True,
    )
    assert oos.expected_behaviour == "scoped" and oos.tier == "B" and not oos.collision
    assert dl.answerable and not oos.answerable


def test_load_questions_lists_every_problem(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
- id: a
  category: lookup
  question: Q?
  expected_articles: ["32024R1689:9"]
- id: b
  category: obligation
  question: Q?
  expected_articles: []
- id: c
  category: out_of_scope
  question: Q?
  expected_articles: ["32024R1689:9"]
- id: d
  category: obligation
  question: Q?
  expected_articles: ["32024R1689:Recital 12", "9"]
- id: d
  category: obligation
  question: ""
  expected_articles: ["32024R1689:9", "32024R1689:9"]
- question: no id
- id: e
  category: obligation
  question: Q?
  expected_articles: ["32024R1689:9"]
  expected_behaviour: refuse
  tier: A
  collision: true
- id: f
  category: out_of_scope
  question: Q?
  expected_articles: []
  expected_behaviour: answer
- id: g
  category: out_of_scope
  question: Q?
  expected_articles: []
  expected_behaviour: maybe
  tier: C
- id: h
  category: direct_lookup
  question: Q?
  expected_articles: ["32024R1689:9"]
  expected_behaviour: answer
  collision: "yes"
""",
    )
    with pytest.raises(ValueError) as error:
        load_questions(path)
    message = str(error.value)
    for expected in [
        "a: unknown category 'lookup'",
        "b: expected_articles is empty",
        "c: out_of_scope questions expect no articles",
        "d: '32024R1689:Recital 12': recitals are not binding",
        "d: '9' is not <celex>:<article_no>",
        "d: empty question",
        "d: expected_articles has duplicates",
        "d: duplicate id",
        "item 6: needs id",
        "e: only out_of_scope questions are scoped or refused",
        "e: tier applies to out_of_scope questions only",
        "e: collision applies to direct_lookup questions only",
        "f: out_of_scope questions are scoped or refused, not answered",
        "f: out_of_scope questions need a tier",
        "g: expected_behaviour 'maybe' must be one of",
        "g: out_of_scope questions need a tier",
        "h: collision must be true or false",
    ]:
        assert expected in message


def test_the_real_questions_file_is_valid_and_complete() -> None:
    questions = load_questions(settings.eval_questions_path)
    check_complete(questions, settings.eval_questions_per_category)


def question(
    category: str = "obligation",
    qid: str = "q",
    expected: tuple[str, ...] = (),
    collision: bool = False,
    tier: str | None = None,
) -> Question:
    behaviour = "answer" if category != "out_of_scope" else "refuse"
    return Question(qid, category, "Q?", expected, behaviour, tier, collision)


def test_check_complete_requires_the_count_per_category() -> None:
    full = [question(c, f"{c}-{i}") for c in CATEGORIES for i in range(2)]
    check_complete(full, per_category=2)
    with pytest.raises(ValueError, match="direct_lookup 1"):
        check_complete(full[1:], per_category=2)
    with pytest.raises(ValueError, match="found"):
        check_complete([], per_category=2)  # the vacuous pass


# --- metrics --------------------------------------------------------------


def test_first_hit_rank_credits_any_cited_provision() -> None:
    citations = [("X:1",), ("X:7", "X:8", "X:9"), ("X:9",)]
    assert first_hit_rank(citations, ["X:9"]) == 2  # a baseline chunk spanning 7 to 9
    assert first_hit_rank(citations, ["X:2"]) is None
    assert first_hit_rank([], ["X:9"]) is None


def test_recall_at_counts_expected_articles_in_the_top_k() -> None:
    citations = [("X:1",), ("Y:5",), ("X:1",), ("X:1",), ("X:1",), ("Z:19",)]
    assert recall_at(citations, ["X:1", "Y:5", "Z:19"], k=5) == pytest.approx(2 / 3)
    assert recall_at(citations, ["X:1", "Y:5", "Z:19"], k=6) == 1.0
    with pytest.raises(ValueError):
        recall_at(citations, [], k=5)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(1) == 1.0 and reciprocal_rank(4) == 0.25 and reciprocal_rank(None) == 0.0


def test_separation_auc() -> None:
    assert separation_auc([0.9, 0.8], [0.3, 0.2]) == 1.0
    assert separation_auc([0.3], [0.9]) == 0.0
    assert separation_auc([0.5], [0.5]) == 0.5
    assert separation_auc([0.9, 0.4], [0.5]) == 0.5
    assert separation_auc([0.9], []) is None


def test_spread() -> None:
    s = spread([0.2, 0.9, 0.5])
    assert s is not None and (s.min, s.median, s.max) == (0.2, 0.5, 0.9)
    assert spread([]) is None


def chunk(*citations: str, chunk_id: int = 1) -> Retrieved:
    return Retrieved(
        chunk_id=chunk_id,
        celex=AI_ACT,
        article_no=None,
        header="",
        text="",
        vector_rank=1,
        fts_rank=None,
        reference_rank=None,
        pinned=False,
        coverage=False,
        fused_score=0.01,
        similarity=0.5,
        strategy=SECTION_AWARE,
        kind=SectionKind.ARTICLE,
        section_path="p",
        citations=citations,
    )


def result(q: Question, ranked: list[tuple[str, ...]], nearest: float = 0.6) -> QuestionResult:
    hits = [chunk(*c, chunk_id=i) for i, c in enumerate(ranked)]
    return score_question(q, SearchResult(hits, nearest), latency_ms=2.0)


def test_score_question() -> None:
    q = question("cross_document", expected=("X:5", "Y:10"))
    r = result(q, [("Z:1",), ("Y:10",), ("Z:2",)])
    assert r.first_hit_rank == 2 and r.hit_at_5 and r.recall_at_5 == 0.5

    late = result(q, [("Z:1",)] * 5 + [("X:5",)])
    assert late.first_hit_rank == 6 and not late.hit_at_5
    assert late.recall_at_5 == 0.0 and late.context_recall == 0.5  # rank 6 is in the context

    oos = result(question("out_of_scope"), [("Z:1",)])
    assert oos.first_hit_rank is None and oos.recall_at_5 is None and not oos.hit_at_5


def test_summarise_averages_over_answerable_questions_only() -> None:
    results = [
        result(question("direct_lookup", "a", ("X:9",)), [("X:9",)], nearest=0.9),  # rank 1
        result(question("obligation", "b", ("X:5",)), [("Z:1",), ("Z:2",), ("X:5",)], nearest=0.8),
        result(question("obligation", "c", ("X:4", "X:6")), [("Z:1",)], nearest=0.4),  # miss
        result(question("out_of_scope", "d"), [("X:9", "X:10")], nearest=0.5),
    ]
    m = summarise(results)
    assert m.n_questions == 4 and m.n_answerable == 3
    assert m.hit_at_5 == pytest.approx(2 / 3)
    assert m.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert m.recall_at_5 == pytest.approx((1 + 1 + 0) / 3)
    assert m.by_category["obligation"] == CategoryMetrics(
        n=2, hit_at_5=0.5, recall_at_5=0.5, context_recall=0.5, mrr=pytest.approx(1 / 6)
    )
    assert "out_of_scope" not in m.by_category
    assert m.separation_auc == pytest.approx(2 / 3)  # 0.9 and 0.8 beat 0.5; 0.4 does not
    assert m.provisions_per_chunk == pytest.approx(7 / 6)  # six chunks, one cites two


def test_direct_lookup_is_also_scored_without_collision_questions() -> None:
    results = [
        result(question("direct_lookup", "dl-1", ("X:9",)), [("X:9",)]),
        result(question("direct_lookup", "dl-2", ("X:5",)), [("Z:1",)]),
        result(question("direct_lookup", "dl-7", ("X:17",), collision=True), [("Y:17",)]),
        result(question("obligation", "ob-1", ("X:5",)), [("X:5",)]),
    ]
    m = summarise(results)
    assert m.by_category["direct_lookup"].hit_at_5 == pytest.approx(1 / 3)
    without = m.direct_lookup_without_collisions
    assert without is not None and without.n == 2 and without.hit_at_5 == 0.5

    no_flags = [r for r in results if not r.question.collision]
    assert summarise(no_flags).direct_lookup_without_collisions is None  # nothing to compare


def test_summarise_refuses_a_vacuous_set() -> None:
    with pytest.raises(ValueError, match="vacuous"):
        summarise([result(question("out_of_scope"), [])])


# --- matrix ---------------------------------------------------------------


def test_matrix_has_every_combination_including_the_shipped_one() -> None:
    combos = matrix_params()
    assert len(combos) == 2 * 2 * len(settings.eval_fts_rank_normalizations) == 12
    assert len(set(combos)) == 12
    assert SearchParams.from_settings() in combos
    assert {p.strategy for p in matrix_params(RECURSIVE)} == {RECURSIVE}
    assert matrix_params(matrix=False) == [SearchParams.from_settings()]


def test_matrix_adds_a_shipped_value_outside_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fts_rank_normalization", 8)
    combos = matrix_params()
    assert combos[0] == SearchParams.from_settings() and len(combos) == 13


def test_cached_encoder_encodes_once() -> None:
    calls: list[list[str]] = []

    def encode(texts: list[str]) -> object:
        calls.append(texts)
        import numpy as np

        return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)

    lookup = cached_encoder(encode, ["a", "b"])  # type: ignore[arg-type]
    assert lookup(["b"]).tolist() == [[2.0, 3.0]]
    assert lookup(["a", "b"]).shape == (2, 2)
    assert calls == [["a", "b"]]


# --- config hash ----------------------------------------------------------


def test_config_hash_tracks_retrieval_settings_only(monkeypatch: pytest.MonkeyPatch) -> None:
    base = config_hash()
    assert config_hash() == base
    assert config_hash({"fts_rank_normalization": 2}) != base
    assert config_hash({"exclude_recitals": not settings.exclude_recitals}) != base

    monkeypatch.setattr(settings, "embedding_batch_size", 7)  # throughput, not results
    monkeypatch.setattr(settings, "database_url", "postgresql://elsewhere/x")
    assert config_hash() == base

    monkeypatch.setattr(settings, "rrf_k", 30)
    assert config_hash() != base


def test_config_hash_rejects_unknown_overrides() -> None:
    with pytest.raises(ValueError, match="not hashed"):
        config_hash({"database_url": "x"})


# --- outputs --------------------------------------------------------------


def metrics(hit: float, n: int = 40) -> Metrics:
    return Metrics(
        hit_at_5=hit,
        recall_at_5=hit - 0.1,
        context_recall=hit - 0.05,
        mrr=hit - 0.2,
        separation_auc=0.8,
        similarity_in_scope=spread([0.6, 0.7]),
        similarity_out_of_scope=spread([0.4, 0.5]),
        provisions_per_chunk=1.0,
        mean_latency_ms=3.0,
        n_questions=n,
        n_answerable=30,
        by_category={"direct_lookup": CategoryMetrics(10, hit, hit, hit, hit)},
        direct_lookup_without_collisions=CategoryMetrics(8, hit, hit, hit, hit),
    )


def run_result(
    params: SearchParams, hit: float, results: tuple[QuestionResult, ...] = ()
) -> RunResult:
    return RunResult(params, f"hash-{hit}", "corpus", 100, metrics(hit), results)


SHIPPED = SearchParams(SECTION_AWARE, False, 0)
BEST = SearchParams(SECTION_AWARE, True, 1)
BASELINE = SearchParams(RECURSIVE, False, 0)


def test_best_run_breaks_ties_on_recall_then_mrr() -> None:
    a, b = run_result(SHIPPED, 0.7), run_result(BEST, 0.8)
    assert best_run([a, b]) is b


def test_metrics_json_gates_on_the_shipped_combination(tmp_path: Path) -> None:
    runs = [run_result(SHIPPED, 0.7), run_result(BEST, 0.9)]
    path = tmp_path / "eval-metrics.json"
    write_metrics_json(runs, SHIPPED, path, expected_questions=40, sha="abc123")

    data = json.loads(path.read_text())
    assert data["hit_at_5"] == 0.7  # shipped, not best
    assert data["n_questions"] == 40 and data["expected_questions"] == 40
    assert data["config"] == SHIPPED.label and data["git_sha"] == "abc123"
    assert set(data["by_config"]) == {SHIPPED.label, BEST.label}


def test_report_leads_with_the_matrix_and_details_three_runs(tmp_path: Path) -> None:
    miss = result(question("obligation", "ob-09", ("X:5",)), [("Z:1",), ("Z:2",)])
    partial = result(question("cross_document", "cd-01", ("X:5", "Y:10")), [("X:5",)])
    hit_ = result(question("obligation", "ob-01", ("X:5",)), [("X:5",)], nearest=0.70)
    near_miss = result(question("out_of_scope", "oos-06", tier="B"), [("X:1",)], nearest=0.65)
    clean = result(question("out_of_scope", "oos-09", tier="A"), [("X:1",)], nearest=0.40)
    runs = [
        run_result(SHIPPED, 0.7, (miss, partial, hit_, near_miss, clean)),
        run_result(BEST, 0.9),
        run_result(BASELINE, 0.5),
        run_result(SearchParams(RECURSIVE, True, 2), 0.4),
    ]
    path = tmp_path / "eval-report.md"
    write_report(runs, SHIPPED, path, expected_questions=40, sha="abc123")
    text = path.read_text()

    assert text.index("## hit@5 across the matrix") < text.index("(shipped)\n")
    assert "| section_aware | included | 0 | 0.700 | shipped |" in text
    assert "| section_aware | excluded | 1 | 0.900 | best |" in text
    assert text.count("| category | n |") == 3  # shipped, best, recursive baseline
    assert "same questions it is scored on" in text
    assert "**ob-09** (obligation, miss, not retrieved)" in text
    assert "**cd-01** (cross_document, partial, recall@5 0.50)" in text
    assert "Incomplete" not in text
    assert "| direct_lookup, collision questions excluded | 8 |" in text
    assert "recall@8 (context)" in text
    # out-of-scope listing: per question, tier shown, highest similarity first
    assert text.index("| oos-06 | B | refuse | 0.650 | 2 of 3 |") < text.index(
        "| oos-09 | A | refuse | 0.400 | 0 of 3 |"
    )


def test_report_flags_an_incomplete_question_set(tmp_path: Path) -> None:
    runs = [run_result(SHIPPED, 0.7)]
    path = tmp_path / "eval-report.md"
    write_report(runs, SHIPPED, path, expected_questions=80, sha=None)
    assert "Incomplete question set: 40 of 80" in path.read_text()
