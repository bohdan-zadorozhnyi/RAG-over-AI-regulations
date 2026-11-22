"""Step 5. Hybrid retrieval.

Two independent rankings fused by reciprocal rank fusion:

  score(d) = sum over rankings r of 1 / (rrf_k + rank_r(d))

Both paths exist because legal queries are bimodal. "What does Article 9
require" is lexical — an embedding blurs the article number. "What are my
obligations around training data" is semantic — full-text misses paraphrase.
Pure vector search fails badly on the first kind.

The fusion runs in SQL, one round trip; reciprocal_rank_fusion() below is the
reference implementation the tests hold the SQL to.

Full-text matching uses OR, not AND: requiring every word of a natural
question to appear in one chunk matched nothing for 4 of 5 eval questions.
The OR query is built from Postgres's own lexemes (same stemmer and stopwords
as chunks.tsv), so the FTS path ranks chunks by query-term overlap. RRF only
consumes ranks, so it tolerates that noisier ranking.

A third ranking handles explicit references. Neither path above can resolve
"Article 12 of DORA": the number matches every chunk containing a 12, and
the embedding barely registers it. parse_references() finds "Article N",
"Art. N" and "Annex <roman>", each paired with the nearest regulation named
(configured aliases); unpaired references apply to every regulation in
scope. Chunks of the referenced provisions are ranked by vector distance and
fused like the other two lists, and the best chunk of each referenced
provision is pinned to the top: naming a provision is explicit intent.

Regulation coverage (with_coverage) keeps the tail of the results for
regulations the head leaves out, so cross-regulation questions reach the
answer step with every regulation they span.
"""

import re
from dataclasses import dataclass, replace
from typing import Any

from pgvector.psycopg import register_vector

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder, check_model_indexed, chunk_input, load_encoder, query_input
from regrag.models import Retrieved, SearchResult, SectionKind, select_strategies
from regrag.reranker import Reranker, load_reranker

ALL_KINDS = [k.value for k in SectionKind]
NON_RECITAL_KINDS = [k.value for k in SectionKind if k is not SectionKind.RECITAL]


@dataclass(frozen=True)
class SearchParams:
    """The settings an eval matrix varies. Everything else comes from config."""

    strategy: str
    exclude_recitals: bool
    fts_rank_normalization: int

    @classmethod
    def from_settings(
        cls,
        strategy: str | None = None,
        exclude_recitals: bool | None = None,
        fts_rank_normalization: int | None = None,
    ) -> "SearchParams":
        """The shipped configuration, with any given field overridden."""
        [chosen] = select_strategies(strategy or settings.chunk_strategy)  # not "all"
        norm = (
            settings.fts_rank_normalization
            if fts_rank_normalization is None
            else fts_rank_normalization
        )
        if norm < 0:
            raise ValueError(f"fts_rank_normalization is a bitmask, got {norm}")
        return cls(
            strategy=chosen,
            exclude_recitals=(
                settings.exclude_recitals if exclude_recitals is None else exclude_recitals
            ),
            fts_rank_normalization=norm,
        )

    def overrides(self) -> dict[str, object]:
        """As setting names, for config_hash."""
        return {
            "chunk_strategy": self.strategy,
            "exclude_recitals": self.exclude_recitals,
            "fts_rank_normalization": self.fts_rank_normalization,
        }

    @property
    def label(self) -> str:
        recitals = "no recitals" if self.exclude_recitals else "recitals"
        return f"{self.strategy}, {recitals}, norm {self.fts_rank_normalization}"


_REFERENCE = re.compile(
    r"\b(?:articles?|art\.?)\s*(?P<article>\d+[a-z]?)\b"
    r"|\bannex\s+(?P<annex>[ivxlc]+)\b",
    re.IGNORECASE,
)


def parse_references(question: str) -> list[str]:
    """Provisions the question names, as "<celex>:<article_no>" citations.

    'What does Article 12 of DORA require?' -> ['32022R2554:12']
    'GDPR Art. 35 ...' -> ['32016R0679:35']
    'What does Annex III of the AI Act list?' -> ['32024R1689:Annex III']
    'What does Article 9 say?' -> Article 9 of every regulation in scope

    Each reference takes the regulation written next to it ("Article 12 of
    DORA", "DORA's Article 30", "GDPR Art. 35"), else the nearest one
    mentioned, else every regulation in scope.
    """
    aliases = [
        (celex, re.escape(alias))
        for celex in settings.celex_ids
        for alias in settings.document_aliases.get(celex, [])
    ]
    mentions = [
        (match.start(), celex)
        for celex, alias in aliases
        for match in re.finditer(rf"\b{alias}\b", question, re.IGNORECASE)
    ]
    found: list[str] = []
    for ref in _REFERENCE.finditer(question):
        article = ref["article"]
        number = article.lower() if article else f"Annex {ref['annex'].upper()}"
        after, before = question[ref.end() :], question[: ref.start()]
        adjacent = [
            celex
            for celex, alias in aliases
            if re.match(rf"(?:\(\w+\))*\s*(?:of\s+(?:the\s+)?)?{alias}\b", after, re.I)
            or re.search(rf"\b{alias}(?:'s)?\s*$", before, re.I)
        ]
        if adjacent:
            targets = adjacent[:1]
        elif mentions:
            targets = [min(mentions, key=lambda m: abs(m[0] - ref.start()))[1]]
        else:
            targets = list(settings.celex_ids)
        found.extend(f"{celex}:{number}" for celex in targets)
    return list(dict.fromkeys(found))  # unique, in order


# Session-local pgvector settings, applied inside the search transaction.
# iterative_scan keeps the HNSW scan going when WHERE filters (model,
# strategy, kind, documents) discard candidates, so a filtered search still
# returns vector_top_k rows; strict_order keeps results in distance order.
SET_HNSW = """
SELECT set_config('hnsw.ef_search', %(ef_search)s, true),
       set_config('hnsw.iterative_scan', 'strict_order', true)
"""

# Shared by both queries below. Filters apply to the section a chunk starts
# in (chunks.section_id).
_QUERY_CTE = """
query AS (
    SELECT (
        SELECT string_agg(quote_literal(lexeme), ' | ')
        FROM unnest(to_tsvector('english', %(question)s))
    )::tsquery AS tsq  -- NULL when the question is all stopwords
)"""

# Reads a CTE `fused` (chunk_id, vector_rank, fts_rank, ref_rank, pinned,
# score, distance) and returns one hit row per chunk, with its citations.
_HITS = """
SELECT f.chunk_id, d.celex, s.article_no, c.header, c.text,
       f.vector_rank, f.fts_rank, f.ref_rank, f.pinned, f.score,
       1 - f.distance AS similarity,
       c.strategy, s.kind, s.path, cited.citations,
       {nearest} AS nearest_similarity
FROM fused f
JOIN chunks c ON c.id = f.chunk_id
JOIN sections s ON s.id = c.section_id
JOIN documents d ON d.id = c.document_id
CROSS JOIN LATERAL (
    SELECT array_agg(citation ORDER BY first_section) AS citations
    FROM (
        SELECT CASE WHEN s2.kind = 'recital'
                    THEN d.celex || ':Recital ' || substr(s2.path, length('recital-') + 1)
                    ELSE d.celex || ':' || s2.article_no
               END AS citation,
               min(s2.id) AS first_section
        FROM chunk_sections cs
        JOIN sections s2 ON s2.id = cs.section_id
        WHERE cs.chunk_id = f.chunk_id
        GROUP BY 1
    ) provisions
) cited
"""

FUSED_SEARCH = (
    "WITH"
    + _QUERY_CTE
    + """,
vec AS (
    SELECT chunk_id, distance, row_number() OVER (ORDER BY distance, chunk_id) AS rank
    FROM (
        SELECT e.chunk_id, e.vector <=> %(qvec)s AS distance
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN sections s ON s.id = c.section_id
        JOIN documents d ON d.id = c.document_id
        WHERE e.model = %(model)s
          AND c.strategy = %(strategy)s
          AND s.kind = ANY(%(kinds)s)
          AND d.celex = ANY(%(celex_ids)s)
        ORDER BY e.vector <=> %(qvec)s
        LIMIT %(vector_k)s
    ) nearest
),
fts AS (
    SELECT chunk_id, row_number() OVER (ORDER BY score DESC, chunk_id) AS rank
    FROM (
        SELECT c.id AS chunk_id, ts_rank(c.tsv, query.tsq, %(norm)s) AS score
        FROM chunks c
        JOIN sections s ON s.id = c.section_id
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN query
        WHERE c.tsv @@ query.tsq
          AND c.strategy = %(strategy)s
          AND s.kind = ANY(%(kinds)s)
          AND d.celex = ANY(%(celex_ids)s)
        ORDER BY score DESC, c.id
        LIMIT %(fts_k)s
    ) matched
),
-- Chunks of the provisions the question names. A chunk counts for every
-- provision it overlaps (chunk_sections), so baseline chunks qualify too.
ref AS (
    SELECT chunk_id, distance, rank, provision_rank = 1 AS pinned
    FROM (
        SELECT chunk_id, distance,
               row_number() OVER (ORDER BY distance, chunk_id) AS rank,
               row_number() OVER (PARTITION BY provision ORDER BY distance, chunk_id)
                   AS provision_rank
        FROM (
            SELECT c.id AS chunk_id, min(d.celex || ':' || s2.article_no) AS provision,
                   min(e.vector <=> %(qvec)s) AS distance
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id AND e.model = %(model)s
            JOIN sections s ON s.id = c.section_id
            JOIN documents d ON d.id = c.document_id
            JOIN chunk_sections cs ON cs.chunk_id = c.id
            JOIN sections s2 ON s2.id = cs.section_id
            WHERE c.strategy = %(strategy)s
              AND s.kind = ANY(%(kinds)s)
              AND d.celex = ANY(%(celex_ids)s)
              AND d.celex || ':' || s2.article_no = ANY(%(references)s)
            GROUP BY c.id
        ) referenced
    ) ranked
    WHERE rank <= %(ref_k)s
),
fused AS (
    SELECT chunk_id,
           min(vector_rank) AS vector_rank,
           min(fts_rank) AS fts_rank,
           min(ref_rank) AS ref_rank,
           min(distance) AS distance,
           bool_or(pinned) AS pinned,
           sum(1.0::float8 / (%(rrf_k)s + rank)) AS score
    FROM (
        SELECT chunk_id, rank, rank AS vector_rank, NULL::bigint AS fts_rank,
               NULL::bigint AS ref_rank, distance, false AS pinned FROM vec
        UNION ALL
        SELECT chunk_id, rank, NULL, rank, NULL, NULL, false FROM fts
        UNION ALL
        SELECT chunk_id, rank, NULL, NULL, rank, distance, pinned FROM ref
    ) ranked
    GROUP BY chunk_id
    ORDER BY pinned DESC, score DESC, chunk_id
    LIMIT %(top_n)s
)"""
    + _HITS.format(nearest="1 - (SELECT min(distance) FROM vec)")
    + """
-- Pinned first (the best chunk of each referenced provision, nearest first),
-- then by fused score.
ORDER BY f.pinned DESC, CASE WHEN f.pinned THEN f.ref_rank END, f.score DESC, f.chunk_id
"""
)

# The best chunks of each regulation in scope (per_regulation of them), by the
# same vector + full-text fusion run within that regulation alone. Supplies
# regulation coverage.
COVERAGE_SEARCH = (
    "WITH"
    + _QUERY_CTE
    + """,
doc_vec AS (
    SELECT r.celex, n.chunk_id, n.distance,
           row_number() OVER (PARTITION BY r.celex ORDER BY n.distance, n.chunk_id) AS rank
    FROM unnest(%(celex_ids)s::text[]) AS r(celex)
    CROSS JOIN LATERAL (
        SELECT e.chunk_id, e.vector <=> %(qvec)s AS distance
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN sections s ON s.id = c.section_id
        JOIN documents d ON d.id = c.document_id
        WHERE e.model = %(model)s
          AND c.strategy = %(strategy)s
          AND s.kind = ANY(%(kinds)s)
          AND d.celex = r.celex
        ORDER BY e.vector <=> %(qvec)s
        LIMIT %(vector_k)s
    ) n
),
doc_fts AS (
    SELECT celex, chunk_id, rank
    FROM (
        SELECT d.celex, c.id AS chunk_id,
               row_number() OVER (
                   PARTITION BY d.celex ORDER BY ts_rank(c.tsv, query.tsq, %(norm)s) DESC, c.id
               ) AS rank
        FROM chunks c
        JOIN sections s ON s.id = c.section_id
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN query
        WHERE c.tsv @@ query.tsq
          AND c.strategy = %(strategy)s
          AND s.kind = ANY(%(kinds)s)
          AND d.celex = ANY(%(celex_ids)s)
    ) ranked
    WHERE rank <= %(fts_k)s
),
doc_fused AS (
    SELECT celex, chunk_id, min(distance) AS distance,
           sum(1.0::float8 / (%(rrf_k)s + rank)) AS score
    FROM (
        SELECT celex, chunk_id, rank, distance FROM doc_vec
        UNION ALL
        SELECT celex, chunk_id, rank, NULL FROM doc_fts
    ) ranked
    GROUP BY celex, chunk_id
),
fused AS (
    SELECT chunk_id, NULL::bigint AS vector_rank, NULL::bigint AS fts_rank,
           NULL::bigint AS ref_rank, false AS pinned, score, distance
    FROM (
        SELECT chunk_id, score, distance,
               row_number() OVER (PARTITION BY celex ORDER BY score DESC, chunk_id) AS place
        FROM doc_fused
    ) per_regulation
    WHERE place <= %(per_regulation)s
)"""
    + _HITS.format(nearest="NULL::float8")
    + """
ORDER BY f.score DESC, f.chunk_id
"""
)


def search(
    question: str,
    params: SearchParams | None = None,
    conn: Connection | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
) -> SearchResult:
    """Top `fusion_top_k` chunks for a question, fused from vector and full-text.

    `params` defaults to the shipped configuration. Refuses to run if the
    configured model has no embeddings for the strategy: a question vector
    compared against another model's vectors returns confident nonsense with
    no error. `conn`, `encoder` and `reranker` are injectable.
    """
    if not question.strip():
        raise ValueError("empty question")
    chosen = params or SearchParams.from_settings()
    reranking = bool(settings.rerank_model)

    owns_conn = conn is None
    db = conn or connect()
    try:
        register_vector(db)
        check_model_indexed(db, settings.embedding_model, chosen.strategy)
        encode = encoder or load_encoder()
        query_vector = encode([query_input(question)])[0]
        sql_params = {
            "question": question,
            "qvec": query_vector,
            "model": settings.embedding_model,
            "strategy": chosen.strategy,
            "kinds": NON_RECITAL_KINDS if chosen.exclude_recitals else ALL_KINDS,
            "celex_ids": settings.celex_ids,
            "vector_k": settings.vector_top_k,
            "fts_k": settings.fts_top_k,
            "norm": chosen.fts_rank_normalization,
            "references": parse_references(question),
            "ref_k": settings.reference_top_k,
            "rrf_k": settings.rrf_k,
            "top_n": (
                max(settings.fusion_top_k, settings.rerank_candidates)
                if reranking
                else settings.fusion_top_k
            ),
            "per_regulation": settings.coverage_candidates if reranking else 1,
        }
        covering = settings.regulation_coverage and len(settings.celex_ids) > 1
        with db.transaction():
            db.execute(SET_HNSW, {"ef_search": str(settings.hnsw_ef_search)})
            rows = db.execute(FUSED_SEARCH, sql_params).fetchall()
            coverage_rows = db.execute(COVERAGE_SEARCH, sql_params).fetchall() if covering else []
    finally:
        if owns_conn:
            db.close()

    hits = [_hit(row) for row in rows]
    best_per_regulation = [_hit(row) for row in coverage_rows]
    if reranking:
        hits, best_per_regulation = rerank(
            question, hits, best_per_regulation, reranker or load_reranker()
        )
    if covering:
        hits = with_coverage(
            hits,
            best_per_regulation,
            top_n=settings.fusion_top_k,
            regulations=len(settings.celex_ids),
        )
    hits = hits[: settings.fusion_top_k]
    # Every row carries the same value, NULL when the vector side is empty. No
    # rows at all means both sides were empty.
    nearest = rows[0][-1] if rows else None
    return SearchResult(hits, None if nearest is None else float(nearest))


def with_coverage(
    hits: list[Retrieved], best_per_regulation: list[Retrieved], top_n: int, regulations: int
) -> list[Retrieved]:
    """Reserve the tail of the results for regulations the head leaves out.

    The first top_n - (regulations - 1) hits stay as fused, so hit@5 is
    untouched by construction while top_n >= regulations + 4. Each remaining
    slot goes to the best chunk (ranked within its own regulation) of a
    regulation the head does not cite, best first; leftover slots go back to
    the fused order. A whole-question embedding is dominated by whichever
    regulation's vocabulary the question leans on, but within each regulation
    the right provision usually ranks first, so a cross-regulation question
    then has every regulation in the context step 7 answers from.
    """
    head = hits[: max(top_n - (regulations - 1), 0)]
    covered = {h.celex for h in head}
    taken = {h.chunk_id for h in head}
    fused = {h.chunk_id: h for h in hits}
    tail = []
    for best in best_per_regulation:
        if best.celex not in covered and best.chunk_id not in taken:
            tail.append(replace(fused.get(best.chunk_id, best), coverage=True))
            covered.add(best.celex)
            taken.add(best.chunk_id)
    rest = [h for h in hits if h.chunk_id not in taken]
    return (head + tail + rest)[:top_n]


def rerank(
    question: str,
    hits: list[Retrieved],
    coverage_candidates: list[Retrieved],
    reranker: Reranker,
) -> tuple[list[Retrieved], list[Retrieved]]:
    """Rescore every candidate with the cross-encoder.

    Returns the fused candidates reordered (pinned references stay first, in
    reference order; the rest by reranker score), and the best candidate of
    each regulation by reranker score, best first, for coverage. Every
    candidate of a regulation competes for its coverage slot, whether fusion
    or the per-regulation search found it.
    """
    # A chunk found by both keeps its fused version: ranks, pin and all.
    pool = {h.chunk_id: h for h in coverage_candidates} | {h.chunk_id: h for h in hits}
    scores = reranker(query_input(question), [chunk_input(h.header, h.text) for h in pool.values()])
    scored = {
        chunk_id: replace(h, rerank_score=score)
        for (chunk_id, h), score in zip(pool.items(), scores, strict=True)
    }

    def order(h: Retrieved) -> tuple[bool, int, float, int]:
        reference_order = h.reference_rank or 0 if h.pinned else 0
        return (not h.pinned, reference_order, -(h.rerank_score or 0.0), h.chunk_id)

    ranked = sorted((scored[h.chunk_id] for h in hits), key=order)
    best: dict[str, Retrieved] = {}
    for h in scored.values():
        current = best.get(h.celex)
        if current is None or (h.rerank_score or 0.0) > (current.rerank_score or 0.0):
            best[h.celex] = h
    by_score = sorted(best.values(), key=lambda h: (-(h.rerank_score or 0.0), h.chunk_id))
    return ranked, by_score


def _hit(row: tuple[Any, ...]) -> Retrieved:
    (
        chunk_id,
        celex,
        article_no,
        header,
        text,
        vector_rank,
        fts_rank,
        ref_rank,
        pinned,
        score,
        similarity,
        strategy,
        kind,
        path,
        citations,
        _nearest,
    ) = row
    return Retrieved(
        chunk_id=chunk_id,
        celex=celex,
        article_no=article_no,
        header=header,
        text=text,
        vector_rank=vector_rank,
        fts_rank=fts_rank,
        reference_rank=ref_rank,
        pinned=bool(pinned),
        coverage=False,
        rerank_score=None,
        fused_score=float(score),
        similarity=None if similarity is None else float(similarity),
        strategy=strategy,
        kind=SectionKind(kind),
        section_path=path,
        citations=tuple(citations or ()),
    )


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, int]]], k: int, top_n: int
) -> list[tuple[int, float]]:
    """Fuse rankings of (chunk_id, rank) into the top_n (chunk_id, score).

    score = sum over rankings containing the chunk of 1 / (k + rank), ranks
    starting at 1. A chunk absent from a ranking gets nothing from it. Ties
    break by chunk_id so the order is deterministic.

    The reference implementation: search() fuses in SQL, and the tests check
    the SQL's order against this function on the same two rankings.
    """
    if k < 0:
        raise ValueError(f"rrf k must be non-negative, got {k}")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for chunk_id, rank in ranking:
            if rank < 1:
                raise ValueError(f"ranks start at 1, got {rank} for chunk {chunk_id}")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_n]
