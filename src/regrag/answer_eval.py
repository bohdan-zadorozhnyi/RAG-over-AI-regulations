"""Step 7 eval: does the answer step refuse, scope and cite as it should?

Scored deterministically against the question file, with no grader model:

- behaviour accuracy: status is the question's expected_behaviour (or one of
  its also_accept). Three-way: answer, scoped, refuse. When retrieval did not
  return every expected provision, "scoped" is credited too: answering only
  the part the sources cover is then the honest reply, and the miss is
  retrieval's (counted as retrieval_limited, so it stays visible).
- refusal accuracy: the binary that matters most. Refused exactly when it
  should; a credited alternative (also_accept) counts as correct.
- citation recall: share of expected_articles the answer cites (answerable
  questions; a refusal cites nothing, so scores 0).
- citation precision: share of cited provisions in expected_articles. The
  expected sets are minimal, so citing a relevant extra article lowers it:
  read it as a drift signal, not a correctness score.

Retrieval is fixed at the shipped configuration and is not re-measured here.
Replies are cached by everything that shapes them (model, effort, prompts),
so changing only the scoring or report costs nothing; the run reports what
it actually spent.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from regrag.answer import (
    REPLY_SCHEMA,
    Answer,
    Generation,
    Generator,
    answer,
    load_generator,
    provision_label,
)
from regrag.config import config_hash, settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder, load_encoder
from regrag.evaluate import CATEGORIES, Question, check_complete, git_sha, load_questions


class CachedGenerator:
    """Wraps a Generator with an on-disk cache; counts calls that hit the API."""

    def __init__(self, generate: Generator, cache_dir: Path) -> None:
        self._generate = generate
        self._dir = cache_dir
        self.api_calls = 0
        self.last_cached = False

    def key(self, system: str, user: str) -> str:
        shaping = {
            "model": settings.answer_model,
            "effort": settings.answer_effort,
            "max_tokens": settings.answer_max_tokens,
            "schema": REPLY_SCHEMA,
            "system": system,
            "user": user,
        }
        return hashlib.sha256(json.dumps(shaping, sort_keys=True).encode()).hexdigest()

    def __call__(self, system: str, user: str) -> Generation:
        path = self._dir / f"{self.key(system, user)}.json"
        if path.exists():
            self.last_cached = True
            return Generation(**json.loads(path.read_text()))
        generation = self._generate(system, user)
        self.api_calls += 1
        self.last_cached = False
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(generation)))
        return generation


@dataclass(frozen=True)
class AnswerResult:
    question: Question
    answer: Answer
    cached: bool

    @property
    def retrieval_limited(self) -> bool:
        """An expected provision is not among the retrieved chunks."""
        retrieved = {c for h in self.answer.hits for c in h.citations}
        return not set(self.question.expected_articles) <= retrieved

    @property
    def accepted(self) -> tuple[str, ...]:
        accepted = self.question.accepted_behaviours
        if self.retrieval_limited and "answer" in accepted and "scoped" not in accepted:
            return (*accepted, "scoped")
        return accepted

    @property
    def behaviour_ok(self) -> bool:
        return self.answer.status in self.accepted

    @property
    def refusal_ok(self) -> bool:
        should_refuse = self.question.expected_behaviour == "refuse"
        return self.behaviour_ok or self.answer.refused == should_refuse

    @property
    def citation_recall(self) -> float | None:
        expected = self.question.expected_articles
        if not expected:
            return None
        return len(set(expected) & set(self.answer.citations)) / len(expected)

    @property
    def citation_precision(self) -> float | None:
        if not self.question.expected_articles or not self.answer.citations:
            return None
        hits = set(self.question.expected_articles) & set(self.answer.citations)
        return len(hits) / len(self.answer.citations)


@dataclass(frozen=True)
class AnswerMetrics:
    n_questions: int
    behaviour_accuracy: float
    refusal_accuracy: float
    retrieval_limited: int  # an expected provision was not retrieved
    false_refusals: int  # should answer (fully or in part), refused
    missed_refusals: int  # should refuse, answered
    citation_recall: float | None
    citation_precision: float | None
    downgraded: int  # answered without a valid citation; refused by the code
    invalid_markers: int  # source numbers that name no retrieved chunk
    cost_usd: float  # spent in this run (cached replies are free)
    cost_usd_uncached: float  # what the whole run costs without the cache
    api_calls: int
    mean_latency_ms: float | None  # over replies generated in this run
    by_category: dict[str, dict[str, float | None]]


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarise(results: Sequence[AnswerResult]) -> AnswerMetrics:
    if not results:
        raise ValueError("no results to summarise")
    fresh = [r for r in results if not r.cached]
    by_category: dict[str, dict[str, float | None]] = {}
    for category in CATEGORIES:
        rs = [r for r in results if r.question.category == category]
        if rs:
            by_category[category] = {
                "n": len(rs),
                "behaviour_accuracy": sum(r.behaviour_ok for r in rs) / len(rs),
                "refusal_accuracy": sum(r.refusal_ok for r in rs) / len(rs),
                "citation_recall": _mean([r.citation_recall for r in rs]),
            }
    return AnswerMetrics(
        n_questions=len(results),
        behaviour_accuracy=sum(r.behaviour_ok for r in results) / len(results),
        refusal_accuracy=sum(r.refusal_ok for r in results) / len(results),
        retrieval_limited=sum(r.retrieval_limited for r in results),
        false_refusals=sum(
            r.answer.refused and r.question.expected_behaviour != "refuse" for r in results
        ),
        missed_refusals=sum(
            not r.refusal_ok and r.question.expected_behaviour == "refuse" for r in results
        ),
        citation_recall=_mean([r.citation_recall for r in results]),
        citation_precision=_mean([r.citation_precision for r in results]),
        downgraded=sum(r.answer.downgraded for r in results),
        invalid_markers=sum(len(r.answer.invalid_sources) for r in results),
        cost_usd=sum(r.answer.cost_usd or 0.0 for r in fresh),
        cost_usd_uncached=sum(r.answer.cost_usd or 0.0 for r in results),
        api_calls=len(fresh),
        mean_latency_ms=_mean([float(r.answer.latency_ms) for r in fresh]),
        by_category=by_category,
    )


def run(
    report_path: Path | None = None,
    metrics_path: Path | None = None,
    allow_incomplete: bool = False,
    conn: Connection | None = None,
    encoder: Encoder | None = None,
    generate: Generator | None = None,
) -> tuple[list[AnswerResult], AnswerMetrics]:
    """Answer every question with the configured model and score the replies.

    A failed call stops the run; replies already generated stay cached, so a
    rerun resumes where it stopped at no extra cost.
    """
    questions = load_questions(settings.eval_questions_path)
    expected = settings.eval_questions_per_category * len(CATEGORIES)
    if not allow_incomplete:
        check_complete(questions, settings.eval_questions_per_category)
    cached = CachedGenerator(generate or load_generator(), settings.answer_cache_dir)
    encode = encoder or load_encoder()

    owns_conn = conn is None
    db = conn or connect()
    results = []
    try:
        for q in questions:
            reply = answer(q.question, conn=db, encoder=encode, generate=cached, log=False)
            results.append(AnswerResult(q, reply, cached.last_cached and reply.model is not None))
    finally:
        if owns_conn:
            db.close()

    metrics = summarise(results)
    sha = git_sha()
    if metrics_path:
        write_metrics_json(metrics, metrics_path, expected, sha)
    if report_path:
        write_report(results, metrics, report_path, expected, sha)
    return results, metrics


# --- outputs ------------------------------------------------------------------


def write_metrics_json(
    metrics: AnswerMetrics, path: Path, expected_questions: int, sha: str | None
) -> None:
    payload = {
        **asdict(metrics),
        "expected_questions": expected_questions,
        "question_set": settings.eval_questions_path.name,
        "model": settings.answer_model,
        "effort": settings.answer_effort,
        "retrieval_config_hash": config_hash(),
        "git_sha": sha,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def write_report(
    results: Sequence[AnswerResult],
    metrics: AnswerMetrics,
    path: Path,
    expected_questions: int,
    sha: str | None,
) -> None:
    m = metrics
    lines = [
        "# Answer eval",
        "",
        f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC · git {sha or 'unknown'} · "
        f"{m.n_questions} questions ({settings.eval_questions_path.name}) · "
        f"model {settings.answer_model}, effort {settings.answer_effort}",
        "",
    ]
    if m.n_questions != expected_questions:
        lines += [
            f"> **Incomplete question set: {m.n_questions} of {expected_questions}.**",
            "",
        ]
    latency = "-" if m.mean_latency_ms is None else f"{m.mean_latency_ms / 1000:.1f} s"
    lines += [
        "| metric | value |",
        "|---|---|",
        f"| behaviour accuracy (answer / scoped / refuse) | {_pct(m.behaviour_accuracy)} |",
        f"| refusal accuracy | {_pct(m.refusal_accuracy)} |",
        f"| false refusals · missed refusals | {m.false_refusals} · {m.missed_refusals} |",
        f"| questions limited by retrieval (scoped also credited) | {m.retrieval_limited} |",
        f"| citation recall (expected articles cited) | {_pct(m.citation_recall)} |",
        f"| citation precision (cited within the minimal set) | {_pct(m.citation_precision)} |",
        f"| uncited answers withheld · invalid source markers "
        f"| {m.downgraded} · {m.invalid_markers} |",
        f"| spent this run · full run uncached | ${m.cost_usd:.4f} · ${m.cost_usd_uncached:.4f} |",
        f"| API calls · mean latency (generated replies) | {m.api_calls} · {latency} |",
        "",
        "## By category",
        "",
        "| category | n | behaviour | refusal | citation recall |",
        "|---|---|---|---|---|",
    ]
    for category, c in m.by_category.items():
        lines.append(
            f"| {category} | {c['n']:.0f} | {_pct(c['behaviour_accuracy'])} "
            f"| {_pct(c['refusal_accuracy'])} | {_pct(c['citation_recall'])} |"
        )
    confusion = Counter((r.question.expected_behaviour, r.answer.status) for r in results)
    lines += [
        "",
        "## Expected vs returned status",
        "",
        "| expected \\ returned | answer | scoped | refuse |",
        "|---|---|---|---|",
    ]
    for expected in ("answer", "scoped", "refuse"):
        row = [str(confusion[(expected, got)]) for got in ("answer", "scoped", "refuse")]
        lines.append(f"| {expected} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Questions (failures first)",
        "",
        "| id | accepted | returned | ok | retrieval | citation recall | cited |",
        "|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(results, key=lambda r: (r.behaviour_ok, r.question.id))
    for r in ordered:
        cited = ", ".join(provision_label(c) for c in r.answer.citations) or "-"
        retrieval = "missed one" if r.retrieval_limited else "complete"
        lines.append(
            f"| {r.question.id} | {'/'.join(r.accepted)} | {r.answer.status} "
            f"| {'✓' if r.behaviour_ok else '✗'} | {retrieval} "
            f"| {_pct(r.citation_recall)} | {cited} |"
        )
    lines += ["", "## Replies", ""]
    for r in ordered:
        lines += _reply_lines(r)
    path.write_text("\n".join(lines) + "\n")


def _reply_lines(r: AnswerResult) -> list[str]:
    a = r.answer
    expected = ", ".join(provision_label(e) for e in r.question.expected_articles) or "none"
    lines = [
        f"### {r.question.id} · accepted {'/'.join(r.accepted)}, "
        f"returned {a.status}{'' if r.behaviour_ok else ' ✗'}",
        "",
        f"**Q:** {r.question.question}",
        "",
        f"Expected provisions: {expected}. "
        f"Retrieved: {', '.join(provision_label(c) for h in a.hits for c in h.citations)}.",
        "",
    ]
    if a.text:
        lines += [a.text, ""]
    if a.missing:
        lines += [f"*Missing:* {a.missing}", ""]
    if a.invalid_sources:
        lines += [f"*Invalid source markers dropped:* {a.invalid_sources}", ""]
    return lines
