"""Answer-eval scoring and the reply cache, without a model or a database."""

import json
from pathlib import Path

import pytest

from regrag.answer import Generation, compose
from regrag.answer_eval import AnswerResult, CachedGenerator, summarise, write_report
from regrag.evaluate import Question
from tests.test_answer import HITS, SOURCES, reply


def q(
    qid: str,
    category: str,
    expected: tuple[str, ...] = (),
    behaviour: str = "answer",
    also: tuple[str, ...] = (),
) -> Question:
    tier = None if category != "out_of_scope" else "B"
    return Question(qid, category, f"question {qid}", expected, behaviour, tier, False, also)


def result(question: Question, generation: Generation, cached: bool = False) -> AnswerResult:
    answer = compose(question.question, HITS, SOURCES, generation, 1000)
    return AnswerResult(question, answer, cached)


def test_an_answer_citing_half_the_expected_articles() -> None:
    r = result(
        q("cd-1", "cross_document", ("32016R0679:32", "32016R0679:9")),
        reply("Security [1]. Training data [2]."),
    )
    assert r.behaviour_ok and r.refusal_ok
    assert r.citation_recall == 0.5
    assert r.citation_precision == 0.5  # cited 32 and AI Act 10; only 32 expected


def test_refusing_an_answerable_question_is_a_false_refusal() -> None:
    r = result(q("ob-1", "obligation", ("32016R0679:32",)), reply("", status="refuse"))
    assert not r.behaviour_ok and not r.refusal_ok
    assert r.citation_recall == 0.0 and r.citation_precision is None


def test_scoped_instead_of_answer_is_a_behaviour_miss_but_not_a_refusal_error() -> None:
    r = result(q("ob-2", "obligation", ("32016R0679:32",)), reply("Partly [1].", status="scoped"))
    assert not r.retrieval_limited
    assert not r.behaviour_ok and r.refusal_ok


def test_scoped_is_credited_when_retrieval_missed_an_expected_provision() -> None:
    # GDPR 9 is not among HITS: the sources cannot answer the whole question.
    missed = q("cd-2", "cross_document", ("32024R1689:10", "32016R0679:9"))
    r = result(missed, reply("AI Act 10 allows it [2].", status="scoped"))
    assert r.retrieval_limited and r.accepted == ("answer", "scoped")
    assert r.behaviour_ok


def test_also_accept_credits_a_scoped_reply_to_a_refuse_question() -> None:
    liability = q("oos-1", "out_of_scope", behaviour="refuse", also=("scoped",))
    r = result(liability, reply("Only unlawful processing [1].", status="scoped"))
    assert r.behaviour_ok and r.refusal_ok


def test_summary_counts_errors_and_spend() -> None:
    results = [
        result(q("dl-1", "direct_lookup", ("32016R0679:32",)), reply("Security [1].")),
        result(q("ob-1", "obligation", ("32024R1689:10",)), reply("", status="refuse")),
        result(q("oos-1", "out_of_scope", behaviour="refuse"), reply("It says [1].")),
        result(
            q("oos-2", "out_of_scope", behaviour="refuse"),
            reply("", status="refuse"),
            cached=True,
        ),
    ]
    m = summarise(results)
    assert m.behaviour_accuracy == 0.5 and m.refusal_accuracy == 0.5
    assert (m.false_refusals, m.missed_refusals) == (1, 1)
    assert m.citation_recall == 0.5  # 1.0 and 0.0
    assert m.api_calls == 3
    one_call = (4000 * 0.10 + 600 * 0.50) / 1e6
    assert m.cost_usd == pytest.approx(3 * one_call)
    assert m.cost_usd_uncached == pytest.approx(4 * one_call)
    assert m.by_category["out_of_scope"]["refusal_accuracy"] == 0.5


def test_the_cache_answers_repeats_without_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def generate(system: str, user: str) -> Generation:
        calls.append(user)
        return reply(f"About {user} [1].")

    cached = CachedGenerator(generate, tmp_path)
    first = cached("system", "a")
    assert not cached.last_cached
    assert cached("system", "a") == first and cached.last_cached
    cached("system", "b")
    assert calls == ["a", "b"] and cached.api_calls == 2

    # Anything that shapes the reply is part of the key.
    monkeypatch.setattr("regrag.answer_eval.settings.answer_effort", "high")
    cached("system", "a")
    assert calls == ["a", "b", "a"]
    assert json.loads(next(tmp_path.iterdir()).read_text())["model"] == "claude-haiku-5-5"


def test_report_lists_failures_first_with_their_replies(tmp_path: Path) -> None:
    results = [
        result(q("dl-1", "direct_lookup", ("32016R0679:32",)), reply("Security [1].")),
        result(q("ob-1", "obligation", ("32024R1689:10",)), reply("", status="refuse")),
    ]
    path = tmp_path / "report.md"
    write_report(results, summarise(results), path, expected_questions=40, sha="abc")
    text = path.read_text()

    assert "Incomplete question set: 2 of 40" in text
    assert text.index("| ob-1 | answer | refuse | ✗ | complete") < text.index(
        "| dl-1 | answer | answer | ✓ | complete"
    )
    assert "| answer | 1 | 0 | 1 |" in text  # expected answer: 1 answered, 1 refused
    assert "### ob-1 · accepted answer, returned refuse ✗" in text
    assert "Security [1]." in text
