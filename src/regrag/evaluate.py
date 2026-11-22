"""Step 6. The component that makes this engineering rather than a demo.

Scores the eval questions against a matrix of retrieval settings (strategy x
exclude_recitals x fts_rank_normalization). Metrics come in three tiers, kept
apart so the report stays readable:

  gate         hit_at_5: any expected article in the top 5 chunks, on the
               shipped configuration. The only number CI checks.
  quality      recall_at_5 (share of a question's expected articles in the
               top 5; hit_at_5 is too generous for cross-document questions),
               context_recall (the same over every retrieved chunk: the
               fusion_top_k chunks step 7 answers from), mrr, and
               separation_auc: whether the nearest-chunk similarity of
               in-scope questions is higher than of out-of-scope ones.
  descriptive  provisions_per_chunk, latency, chunk counts. No target value;
               they explain results and never go near the gate.

Refusal accuracy is a generation behaviour and waits for step 7. Separation
is measurable now; no threshold is chosen from it, since tuning a cutoff on
ten out-of-scope questions memorises them.

A chunk hits when its citations include an expected article. Citations come
from chunk_sections, so a baseline chunk is credited with every provision it
spans; provisions_per_chunk shows that cost. Recitals never count: expected
articles are binding provisions.

The question embeddings do not depend on any matrix dimension, so the
questions are encoded once and reused by every combination.

Eval consumes retrieval, as `ask` does; it writes only eval_* tables and
its two output files.
"""

import json
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from psycopg.types.json import Jsonb

from regrag.config import config_hash, settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder, Vectors, load_encoder, query_input
from regrag.models import RECURSIVE, Retrieved, SearchResult, select_strategies
from regrag.reranker import Reranker, load_reranker
from regrag.retrieve import SearchParams, search

HIT_AT = 5  # the metrics' definition: hit_at_5, recall_at_5
OUT_OF_SCOPE = "out_of_scope"
DIRECT_LOOKUP = "direct_lookup"
CATEGORIES = (DIRECT_LOOKUP, "obligation", "cross_document", OUT_OF_SCOPE)
# What a correct answer does; step 7 scores it. "scoped": answer the part the
# corpus supports and say plainly what it does not.
BEHAVIOURS = ("answer", "scoped", "refuse")
# Out-of-scope only. A: the instrument is absent from the corpus. B: it is
# named there (lexical search matches) but its substance is not.
TIERS = ("A", "B")

UPSERT_QUESTION = """
INSERT INTO eval_questions (id, question, expected_articles, category)
VALUES (%(id)s, %(question)s, %(expected_articles)s, %(category)s)
ON CONFLICT (id) DO UPDATE
SET question = EXCLUDED.question,
    expected_articles = EXCLUDED.expected_articles,
    category = EXCLUDED.category
"""

# Every citable binding provision, in the citation format search() returns.
KNOWN_PROVISIONS = """
SELECT DISTINCT d.celex || ':' || s.article_no
FROM sections s JOIN documents d ON d.id = s.document_id
WHERE s.kind <> 'recital' AND s.article_no IS NOT NULL AND d.celex = ANY(%(celex_ids)s)
"""

# Identifies the chunks a run searched: re-chunking changes results under the
# same config_hash, so the fingerprint is recorded with every run.
CORPUS_FINGERPRINT = """
SELECT count(*), md5(coalesce(string_agg(c.content_hash, '' ORDER BY c.content_hash), ''))
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.strategy = %(strategy)s AND d.celex = ANY(%(celex_ids)s)
"""

INSERT_RUN = """
INSERT INTO eval_runs (git_sha, config_hash, strategy, metrics)
VALUES (%(git_sha)s, %(config_hash)s, %(strategy)s, %(metrics)s)
RETURNING id
"""

INSERT_RESULT = """
INSERT INTO eval_results (run_id, question_id, first_hit_rank, hit_at_5, retrieved)
VALUES (%(run_id)s, %(question_id)s, %(first_hit_rank)s, %(hit_at_5)s, %(retrieved)s)
"""


# --- questions ------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    expected_articles: tuple[str, ...]  # "<celex>:<article_no>"; empty: should refuse
    expected_behaviour: str = "answer"
    tier: str | None = None  # out_of_scope only
    # direct_lookup questions written to fail until a structural article
    # lookup exists; direct_lookup is also reported without them
    collision: bool = False

    @property
    def answerable(self) -> bool:
        return self.category != OUT_OF_SCOPE


def load_questions(path: Path) -> list[Question]:
    """Parse and validate the questions file. Raises listing every problem."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of questions")

    questions: list[Question] = []
    problems: list[str] = []
    for i, item in enumerate(raw, start=1):
        where = f"item {i}"
        if not isinstance(item, dict) or not {"id", "category", "question"} <= item.keys():
            problems.append(f"{where}: needs id, category, question, expected_articles")
            continue
        where = str(item["id"])
        expected = item.get("expected_articles") or []
        if not isinstance(expected, list):
            problems.append(f"{where}: expected_articles must be a list")
            continue
        collision = item.get("collision", False)
        if not isinstance(collision, bool):
            problems.append(f"{where}: collision must be true or false")
            continue
        tier = item.get("tier")
        question = Question(
            id=where,
            category=str(item["category"]),
            question=str(item["question"]).strip(),
            expected_articles=tuple(str(e) for e in expected),
            expected_behaviour=str(item.get("expected_behaviour", "")),
            tier=None if tier is None else str(tier),
            collision=collision,
        )
        problems.extend(f"{where}: {p}" for p in _question_problems(question))
        questions.append(question)

    duplicates = [i for i, n in Counter(q.id for q in questions).items() if n > 1]
    problems.extend(f"{i}: duplicate id" for i in duplicates)
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return questions


def _question_problems(q: Question) -> list[str]:
    problems = []
    if q.category not in CATEGORIES:
        problems.append(f"unknown category {q.category!r}; use one of {CATEGORIES}")
    if not q.question:
        problems.append("empty question")
    if q.answerable and not q.expected_articles:
        problems.append("expected_articles is empty; only out_of_scope questions refuse")
    if not q.answerable and q.expected_articles:
        problems.append("out_of_scope questions expect no articles")
    if q.expected_behaviour not in BEHAVIOURS:
        problems.append(f"expected_behaviour {q.expected_behaviour!r} must be one of {BEHAVIOURS}")
    elif q.answerable and q.expected_behaviour != "answer":
        problems.append("only out_of_scope questions are scoped or refused")
    elif not q.answerable and q.expected_behaviour == "answer":
        problems.append("out_of_scope questions are scoped or refused, not answered")
    if not q.answerable and q.tier not in TIERS:
        problems.append(f"out_of_scope questions need a tier, one of {TIERS}")
    if q.answerable and q.tier is not None:
        problems.append("tier applies to out_of_scope questions only")
    if q.collision and q.category != DIRECT_LOOKUP:
        problems.append("collision applies to direct_lookup questions only")
    if len(set(q.expected_articles)) != len(q.expected_articles):
        problems.append("expected_articles has duplicates")
    for article in q.expected_articles:
        celex, _, number = article.partition(":")
        if not celex or not number:
            problems.append(f"{article!r} is not <celex>:<article_no>")
        elif number.startswith("Recital"):
            problems.append(f"{article!r}: recitals are not binding; expect articles")
    return problems


def check_complete(questions: Sequence[Question], per_category: int) -> None:
    """Fail unless every category has exactly `per_category` questions.

    A partial question set otherwise passes vacuously: hit@5 over too few
    questions, or over none, with CI green while the eval measures nothing.
    """
    counts = Counter(q.category for q in questions)
    wrong = {c: counts[c] for c in CATEGORIES if counts[c] != per_category}
    if wrong:
        found = ", ".join(f"{c} {n}" for c, n in wrong.items())
        raise ValueError(
            f"eval needs {per_category} questions per category; found {found}. "
            "Use --allow-incomplete while writing questions."
        )


def store_questions(conn: Connection, questions: Sequence[Question]) -> None:
    """Upsert into eval_questions after checking every expected article exists.

    A typo such as 32024R1689:91 for :9 would otherwise score 0 forever with
    no error. Questions removed from the file stay in the table: past
    eval_results reference them.
    """
    known = {row[0] for row in conn.execute(KNOWN_PROVISIONS, {"celex_ids": settings.celex_ids})}
    unknown = [
        f"{q.id}: {article}"
        for q in questions
        for article in q.expected_articles
        if article not in known
    ]
    if unknown:
        raise ValueError("expected articles not in the parsed corpus:\n  " + "\n  ".join(unknown))
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            UPSERT_QUESTION,
            [{**asdict(q), "expected_articles": list(q.expected_articles)} for q in questions],
        )


# --- metrics (pure) -------------------------------------------------------


def first_hit_rank(citations: Sequence[Sequence[str]], expected: Iterable[str]) -> int | None:
    """1-based rank of the first chunk citing any expected article."""
    wanted = set(expected)
    for rank, cited in enumerate(citations, start=1):
        if wanted.intersection(cited):
            return rank
    return None


def recall_at(citations: Sequence[Sequence[str]], expected: Sequence[str], k: int) -> float:
    """Share of expected articles cited by the top k chunks."""
    if not expected:
        raise ValueError("recall is undefined without expected articles")
    found = {c for cited in citations[:k] for c in cited}
    return sum(article in found for article in expected) / len(expected)


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def separation_auc(in_scope: Sequence[float], out_of_scope: Sequence[float]) -> float | None:
    """Probability that a random in-scope question's nearest similarity beats
    a random out-of-scope one's (ties count half). Threshold-free: 1.0 means a
    cutoff exists that separates them perfectly, 0.5 means no signal."""
    if not in_scope or not out_of_scope:
        return None
    wins = sum(1.0 if i > o else 0.5 if i == o else 0.0 for i in in_scope for o in out_of_scope)
    return wins / (len(in_scope) * len(out_of_scope))


@dataclass(frozen=True)
class Spread:
    min: float
    median: float
    max: float


def spread(values: Sequence[float]) -> Spread | None:
    if not values:
        return None
    return Spread(min(values), statistics.median(values), max(values))


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    hits: tuple[Retrieved, ...]
    first_hit_rank: int | None
    recall_at_5: float | None  # None for out_of_scope
    context_recall: float | None  # over all retrieved chunks; None for out_of_scope
    nearest_similarity: float | None
    latency_ms: float

    @property
    def hit_at_5(self) -> bool:
        return self.first_hit_rank is not None and self.first_hit_rank <= HIT_AT


def score_question(question: Question, result: SearchResult, latency_ms: float) -> QuestionResult:
    citations = [hit.citations for hit in result.hits]
    answerable = question.answerable
    return QuestionResult(
        question=question,
        hits=tuple(result.hits),
        first_hit_rank=(
            first_hit_rank(citations, question.expected_articles) if answerable else None
        ),
        recall_at_5=(
            recall_at(citations, question.expected_articles, HIT_AT) if answerable else None
        ),
        context_recall=(
            recall_at(citations, question.expected_articles, len(citations)) if answerable else None
        ),
        nearest_similarity=result.nearest_similarity,
        latency_ms=latency_ms,
    )


@dataclass(frozen=True)
class CategoryMetrics:
    n: int
    hit_at_5: float
    recall_at_5: float
    context_recall: float
    mrr: float


@dataclass(frozen=True)
class Metrics:
    # gate
    hit_at_5: float
    # quality
    recall_at_5: float
    context_recall: float
    mrr: float
    separation_auc: float | None
    similarity_in_scope: Spread | None
    similarity_out_of_scope: Spread | None
    # descriptive
    provisions_per_chunk: float
    mean_latency_ms: float
    n_questions: int
    n_answerable: int
    by_category: dict[str, CategoryMetrics]
    # direct_lookup minus its collision questions, so a gain from an article
    # lookup can be judged with and without the questions written for it
    direct_lookup_without_collisions: CategoryMetrics | None


def summarise(results: Sequence[QuestionResult]) -> Metrics:
    """Retrieval metrics average over answerable questions only."""
    answerable = [r for r in results if r.question.answerable]
    if not answerable:
        raise ValueError("no answerable questions: every retrieval metric would be vacuous")

    def category(rs: Sequence[QuestionResult]) -> CategoryMetrics:
        return CategoryMetrics(
            n=len(rs),
            hit_at_5=statistics.mean(r.hit_at_5 for r in rs),
            recall_at_5=statistics.mean(r.recall_at_5 or 0.0 for r in rs),
            context_recall=statistics.mean(r.context_recall or 0.0 for r in rs),
            mrr=statistics.mean(reciprocal_rank(r.first_hit_rank) for r in rs),
        )

    overall = category(answerable)
    in_scope = [r.nearest_similarity for r in answerable if r.nearest_similarity is not None]
    out_of_scope = [
        r.nearest_similarity
        for r in results
        if not r.question.answerable and r.nearest_similarity is not None
    ]
    chunks = [hit for r in results for hit in r.hits]
    return Metrics(
        hit_at_5=overall.hit_at_5,
        recall_at_5=overall.recall_at_5,
        context_recall=overall.context_recall,
        mrr=overall.mrr,
        separation_auc=separation_auc(in_scope, out_of_scope),
        similarity_in_scope=spread(in_scope),
        similarity_out_of_scope=spread(out_of_scope),
        provisions_per_chunk=(statistics.mean(len(h.citations) for h in chunks) if chunks else 0.0),
        mean_latency_ms=statistics.mean(r.latency_ms for r in results),
        n_questions=len(results),
        n_answerable=len(answerable),
        by_category={
            c: category([r for r in answerable if r.question.category == c])
            for c in CATEGORIES
            if any(r.question.category == c for r in answerable)
        },
        direct_lookup_without_collisions=_without_collisions(answerable, category),
    )


def _without_collisions(
    answerable: Sequence[QuestionResult],
    category: Callable[[Sequence[QuestionResult]], CategoryMetrics],
) -> CategoryMetrics | None:
    lookups = [r for r in answerable if r.question.category == DIRECT_LOOKUP]
    kept = [r for r in lookups if not r.question.collision]
    if not kept or len(kept) == len(lookups):
        return None  # nothing to compare
    return category(kept)


# --- running the matrix ---------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    params: SearchParams
    config_hash: str
    corpus_hash: str
    chunk_count: int
    metrics: Metrics
    results: tuple[QuestionResult, ...]


def matrix_params(strategy: str = "all", matrix: bool = True) -> list[SearchParams]:
    """Every combination to score, the shipped one first.

    First matters with a reranker: its scores are cached across combinations,
    so only the first combination's latency includes every reranker call.
    """
    shipped = SearchParams.from_settings(None if strategy == "all" else strategy)
    if not matrix:
        return [shipped]
    combos = [
        SearchParams(s, exclude, norm)
        for s in select_strategies(strategy)
        for exclude in (False, True)
        for norm in settings.eval_fts_rank_normalizations
    ]
    return [shipped, *(c for c in combos if c != shipped)]


def cached_encoder(encode: Encoder, texts: Sequence[str]) -> Encoder:
    """Encode `texts` once, in one batch; later calls are dictionary lookups."""
    vectors = encode(list(texts))
    table = dict(zip(texts, vectors, strict=True))

    def lookup(batch: list[str]) -> Vectors:
        return np.asarray([table[t] for t in batch], dtype=np.float32)

    return lookup


def cached_reranker(rerank: Reranker) -> Reranker:
    """Reranker scores depend only on (question, candidate text), so they are
    reused across combinations; the first combination pays for every call."""
    table: dict[tuple[str, str], float] = {}

    def lookup(question: str, candidates: Sequence[str]) -> list[float]:
        missing = list(dict.fromkeys(c for c in candidates if (question, c) not in table))
        for text, score in zip(missing, rerank(question, missing), strict=True):
            table[(question, text)] = score
        return [table[(question, c)] for c in candidates]

    return lookup


def run_matrix(
    questions: Sequence[Question],
    params_list: Sequence[SearchParams],
    conn: Connection,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> list[RunResult]:
    encode = cached_encoder(encoder or load_encoder(), [query_input(q.question) for q in questions])
    raw_rerank = (reranker or load_reranker()) if settings.rerank_model else None
    rerank = cached_reranker(raw_rerank) if raw_rerank else None
    if questions and params_list:  # untimed: the first timed search would pay for cold caches
        # uncached reranker: the first timed search must still pay for its scores
        search(
            questions[0].question, params_list[0], conn=conn, encoder=encode, reranker=raw_rerank
        )
    fingerprints: dict[str, tuple[int, str]] = {}
    runs = []
    for params in params_list:
        if params.strategy not in fingerprints:
            row = conn.execute(
                CORPUS_FINGERPRINT,
                {"strategy": params.strategy, "celex_ids": settings.celex_ids},
            ).fetchone()
            fingerprints[params.strategy] = (int(row[0]), str(row[1])) if row else (0, "")
        count, corpus = fingerprints[params.strategy]

        results = []
        for q in questions:
            started = time.perf_counter()
            found = search(q.question, params, conn=conn, encoder=encode, reranker=rerank)
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(score_question(q, found, elapsed_ms))
        runs.append(
            RunResult(
                params=params,
                config_hash=config_hash(params.overrides()),
                corpus_hash=corpus,
                chunk_count=count,
                metrics=summarise(results),
                results=tuple(results),
            )
        )
    return runs


def record_runs(conn: Connection, runs: Sequence[RunResult], git_sha: str | None) -> list[int]:
    """One eval_runs row per combination, one eval_results row per question."""
    ids = []
    with conn.transaction():
        for run_result in runs:
            row = conn.execute(
                INSERT_RUN,
                {
                    "git_sha": git_sha,
                    "config_hash": run_result.config_hash,
                    "strategy": run_result.params.strategy,
                    "metrics": Jsonb(
                        {
                            **asdict(run_result.metrics),
                            "params": asdict(run_result.params),
                            # main set or held-out: eval_runs has no column for it
                            "question_set": settings.eval_questions_path.name,
                            "corpus_hash": run_result.corpus_hash,
                            "chunk_count": run_result.chunk_count,
                        }
                    ),
                },
            ).fetchone()
            if row is None:
                raise RuntimeError("eval_runs insert returned no id")
            run_id = int(row[0])
            ids.append(run_id)
            with conn.cursor() as cur:
                cur.executemany(
                    INSERT_RESULT,
                    [
                        {
                            "run_id": run_id,
                            "question_id": r.question.id,
                            "first_hit_rank": r.first_hit_rank,
                            "hit_at_5": r.hit_at_5,
                            "retrieved": Jsonb(_retrieved_json(r)),
                        }
                        for r in run_result.results
                    ],
                )
    return ids


def _retrieved_json(r: QuestionResult) -> dict[str, Any]:
    return {
        "nearest_similarity": r.nearest_similarity,
        "recall_at_5": r.recall_at_5,
        "context_recall": r.context_recall,
        "latency_ms": round(r.latency_ms, 2),
        "chunks": [
            {
                "rank": rank,
                "chunk_id": h.chunk_id,
                "citations": list(h.citations),
                "vector_rank": h.vector_rank,
                "fts_rank": h.fts_rank,
                "reference_rank": h.reference_rank,
                "pinned": h.pinned,
                "coverage": h.coverage,
                "rerank_score": h.rerank_score,
                "similarity": h.similarity,
                "fused_score": h.fused_score,
            }
            for rank, h in enumerate(r.hits, start=1)
        ],
    }


def git_sha() -> str | None:
    """HEAD, with -dirty when the working tree has changes; None outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha[:12]}-dirty" if dirty else sha[:12]


def run(
    strategy: str = "all",
    matrix: bool = True,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
    allow_incomplete: bool = False,
    conn: Connection | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> list[RunResult]:
    """Load the questions, score every combination, record and write outputs."""
    questions = load_questions(settings.eval_questions_path)
    expected = settings.eval_questions_per_category * len(CATEGORIES)
    if not allow_incomplete:
        check_complete(questions, settings.eval_questions_per_category)
    params_list = matrix_params(strategy, matrix)  # always includes the shipped one
    shipped = SearchParams.from_settings(None if strategy == "all" else strategy)

    owns_conn = conn is None
    db = conn or connect()
    try:
        store_questions(db, questions)
        runs = run_matrix(questions, params_list, db, encoder, reranker)
        sha = git_sha()
        record_runs(db, runs, sha)
    finally:
        if owns_conn:
            db.close()

    if metrics_path:
        write_metrics_json(runs, shipped, metrics_path, expected, sha)
    if report_path:
        write_report(runs, shipped, report_path, expected, sha)
    return runs


# --- outputs --------------------------------------------------------------


def best_run(runs: Sequence[RunResult]) -> RunResult:
    """Highest hit@5; ties broken by recall@5, then MRR."""
    return max(runs, key=lambda r: (r.metrics.hit_at_5, r.metrics.recall_at_5, r.metrics.mrr))


def find_run(runs: Sequence[RunResult], params: SearchParams) -> RunResult:
    for r in runs:
        if r.params == params:
            return r
    raise ValueError(f"no run for {params.label}")


def write_metrics_json(
    runs: Sequence[RunResult],
    shipped: SearchParams,
    path: Path,
    expected_questions: int,
    sha: str | None,
) -> None:
    """Flat JSON: the shipped combination at the top level (what CI reads),
    every combination under `by_config`."""
    gated = find_run(runs, shipped)
    m = gated.metrics
    payload = {
        "hit_at_5": m.hit_at_5,
        "recall_at_5": m.recall_at_5,
        "context_recall": m.context_recall,
        "mrr": m.mrr,
        "separation_auc": m.separation_auc,
        "n_questions": m.n_questions,
        "expected_questions": expected_questions,
        "question_set": settings.eval_questions_path.name,
        "config": shipped.label,
        "config_hash": gated.config_hash,
        "git_sha": sha,
        "by_config": {
            r.params.label: {
                "hit_at_5": r.metrics.hit_at_5,
                "recall_at_5": r.metrics.recall_at_5,
                "context_recall": r.metrics.context_recall,
                "mrr": r.metrics.mrr,
                "separation_auc": r.metrics.separation_auc,
                "config_hash": r.config_hash,
            }
            for r in runs
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


SELECTION_CAVEAT = (
    "The best combination is chosen on the same questions it is scored on, so "
    "its number is optimistic. data/eval/heldout.yaml estimates by how much: "
    "run it once (make eval-heldout) with the configuration frozen. "
    "One question moves hit@5 by 1/n_answerable: trust differences that hold "
    "across combinations over the best single number."
)


def write_report(
    runs: Sequence[RunResult],
    shipped: SearchParams,
    path: Path,
    expected_questions: int,
    sha: str | None,
) -> None:
    """Leads with one hit@5 table; detail only for shipped, best and baseline."""
    gated = find_run(runs, shipped)
    best = best_run(runs)
    baseline_params = SearchParams(
        RECURSIVE, shipped.exclude_recitals, shipped.fts_rank_normalization
    )
    baseline = next((r for r in runs if r.params == baseline_params), None)
    n = gated.metrics.n_questions

    lines = [
        "# Retrieval eval",
        "",
        f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC · git {sha or 'unknown'} · "
        f"{n} questions ({gated.metrics.n_answerable} answerable, "
        f"{settings.eval_questions_path.name}) · "
        f"model {settings.embedding_model}",
        "",
    ]
    if n != expected_questions:
        lines += [
            f"> **Incomplete question set: {n} of {expected_questions}.** "
            "Numbers are not comparable with complete runs.",
            "",
        ]
    lines += [
        f"## hit@{HIT_AT} across the matrix",
        "",
        "| strategy | recitals | ts_rank norm | hit@5 | |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        marks = [m for m, on in (("shipped", r is gated), ("best", r is best)) if on]
        recitals = "excluded" if r.params.exclude_recitals else "included"
        lines.append(
            f"| {r.params.strategy} | {recitals} | {r.params.fts_rank_normalization} "
            f"| {r.metrics.hit_at_5:.3f} | {', '.join(marks)} |"
        )
    lines += ["", SELECTION_CAVEAT, ""]

    detailed: list[tuple[str, RunResult]] = [("shipped", gated)]
    if best is not gated:
        detailed.append(("best", best))
    if baseline is not None and baseline not in (gated, best):
        detailed.append(("recursive baseline", baseline))
    for role, r in detailed:
        lines += _detail(role, r)

    lines += _out_of_scope_similarities(gated)
    lines += _misses(gated)
    path.write_text("\n".join(lines) + "\n")


def _detail(role: str, r: RunResult) -> list[str]:
    m = r.metrics
    lines = [
        f"## {r.params.label} ({role})",
        "",
        f"| category | n | hit@5 | recall@5 | recall@{settings.fusion_top_k} (context) | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in m.by_category.items():
        lines.append(
            f"| {name} | {c.n} | {c.hit_at_5:.3f} | {c.recall_at_5:.3f} "
            f"| {c.context_recall:.3f} | {c.mrr:.3f} |"
        )
        no_collisions = m.direct_lookup_without_collisions
        if name == DIRECT_LOOKUP and no_collisions is not None:
            c = no_collisions
            lines.append(
                f"| {name}, collision questions excluded | {c.n} | {c.hit_at_5:.3f} "
                f"| {c.recall_at_5:.3f} | {c.context_recall:.3f} | {c.mrr:.3f} |"
            )
    lines.append(
        f"| **all answerable** | {m.n_answerable} | {m.hit_at_5:.3f} "
        f"| {m.recall_at_5:.3f} | {m.context_recall:.3f} | {m.mrr:.3f} |"
    )
    auc = "n/a" if m.separation_auc is None else f"{m.separation_auc:.3f}"
    lines += [
        "",
        f"- **Separation AUC** (nearest-chunk similarity, in-scope vs out-of-scope): {auc}",
        f"  - in-scope {_spread(m.similarity_in_scope)}",
        f"  - out-of-scope {_spread(m.similarity_out_of_scope)}",
        "  - Few out-of-scope questions, so wide error bars: evidence of signal, "
        "not a refusal threshold.",
        f"- Provisions cited per retrieved chunk: {m.provisions_per_chunk:.2f}",
        f"- Mean search latency: {m.mean_latency_ms:.1f} ms ({_latency_note(role)})",
        f"- Chunks searched: {r.chunk_count:,} · config {r.config_hash} · "
        f"corpus {r.corpus_hash[:12]}",
        "",
    ]
    return lines


def _out_of_scope_similarities(r: RunResult) -> list[str]:
    """Per question rather than per-tier AUCs: with this few out-of-scope
    questions, the reader eyeballing the grouping is more honest than a
    statistic implying resolution the set does not have."""
    in_scope = [
        q.nearest_similarity
        for q in r.results
        if q.question.answerable and q.nearest_similarity is not None
    ]
    rows = sorted(
        (q for q in r.results if not q.question.answerable),
        key=lambda q: -(q.nearest_similarity or 0.0),
    )
    if not rows:
        return []
    lines = [
        f"## Out-of-scope questions: nearest similarity ({r.params.label})",
        "",
        "| question | tier | expected | similarity | in-scope questions below it |",
        "|---|---|---|---|---|",
    ]
    for q in rows:
        sim = q.nearest_similarity
        below = "-" if sim is None else f"{sum(s < sim for s in in_scope)} of {len(in_scope)}"
        shown = "-" if sim is None else f"{sim:.3f}"
        lines.append(
            f"| {q.question.id} | {q.question.tier or '-'} | {q.question.expected_behaviour} "
            f"| {shown} | {below} |"
        )
    return [*lines, ""]


def _latency_note(role: str) -> str:
    if not settings.rerank_model:
        return "SQL only; question embeddings cached"
    if role == "shipped":
        return "SQL and reranking; question embeddings cached"
    return "reranker scores cached from the shipped run, so not a latency figure"


def _spread(s: Spread | None) -> str:
    if s is None:
        return "n/a"
    return f"min {s.min:.3f} · median {s.median:.3f} · max {s.max:.3f}"


def _misses(r: RunResult) -> list[str]:
    misses = [q for q in r.results if q.question.answerable and not q.hit_at_5]
    partial = [
        q for q in r.results if q.hit_at_5 and q.recall_at_5 is not None and q.recall_at_5 < 1.0
    ]
    lines = [f"## Misses ({r.params.label})", ""]
    if not misses and not partial:
        return [*lines, "None.", ""]
    for q in misses:
        where = (
            "not retrieved" if q.first_hit_rank is None else f"first hit at rank {q.first_hit_rank}"
        )
        lines += _question_lines(q, f"miss, {where}")
    for q in partial:
        lines += _question_lines(q, f"partial, recall@5 {q.recall_at_5:.2f}")
    return lines


def _question_lines(q: QuestionResult, verdict: str) -> list[str]:
    top = "; ".join(", ".join(h.citations) for h in q.hits[:3]) or "nothing"
    return [
        f"- **{q.question.id}** ({q.question.category}, {verdict}): {q.question.question}",
        f"  - expected {', '.join(q.question.expected_articles)}",
        f"  - top 3: {top}",
    ]
