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
"""

from dataclasses import dataclass

from pgvector.psycopg import register_vector

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder, check_model_indexed, load_encoder, query_input
from regrag.models import Retrieved, SearchResult, SectionKind, select_strategies

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


# Session-local pgvector settings, applied inside the search transaction.
# iterative_scan keeps the HNSW scan going when WHERE filters (model,
# strategy, kind, documents) discard candidates, so a filtered search still
# returns vector_top_k rows; strict_order keeps results in distance order.
SET_HNSW = """
SELECT set_config('hnsw.ef_search', %(ef_search)s, true),
       set_config('hnsw.iterative_scan', 'strict_order', true)
"""

# Filters apply to the section a chunk starts in (chunks.section_id).
FUSED_SEARCH = """
WITH
query AS (
    SELECT (
        SELECT string_agg(quote_literal(lexeme), ' | ')
        FROM unnest(to_tsvector('english', %(question)s))
    )::tsquery AS tsq  -- NULL when the question is all stopwords
),
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
fused AS (
    SELECT coalesce(vec.chunk_id, fts.chunk_id) AS chunk_id,
           vec.rank AS vector_rank,
           vec.distance,
           fts.rank AS fts_rank,
           coalesce(1.0::float8 / (%(rrf_k)s + vec.rank), 0)
         + coalesce(1.0::float8 / (%(rrf_k)s + fts.rank), 0) AS score
    FROM vec FULL OUTER JOIN fts ON fts.chunk_id = vec.chunk_id
    ORDER BY score DESC, chunk_id
    LIMIT %(top_n)s
)
SELECT f.chunk_id, d.celex, s.article_no, c.header, c.text,
       f.vector_rank, f.fts_rank, f.score, 1 - f.distance AS similarity,
       c.strategy, s.kind, s.path, cited.citations,
       1 - (SELECT min(distance) FROM vec) AS nearest_similarity
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
ORDER BY f.score DESC, f.chunk_id
"""


def search(
    question: str,
    params: SearchParams | None = None,
    conn: Connection | None = None,
    encoder: Encoder | None = None,
) -> SearchResult:
    """Top `fusion_top_k` chunks for a question, fused from vector and full-text.

    `params` defaults to the shipped configuration. Refuses to run if the
    configured model has no embeddings for the strategy: a question vector
    compared against another model's vectors returns confident nonsense with
    no error. `conn` and `encoder` are injectable.
    """
    if not question.strip():
        raise ValueError("empty question")
    chosen = params or SearchParams.from_settings()

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
            "rrf_k": settings.rrf_k,
            "top_n": settings.fusion_top_k,
        }
        with db.transaction():
            db.execute(SET_HNSW, {"ef_search": str(settings.hnsw_ef_search)})
            rows = db.execute(FUSED_SEARCH, sql_params).fetchall()
    finally:
        if owns_conn:
            db.close()

    hits = [
        Retrieved(
            chunk_id=chunk_id,
            celex=celex,
            article_no=article_no,
            header=header,
            text=text,
            vector_rank=vector_rank,
            fts_rank=fts_rank,
            fused_score=float(score),
            similarity=None if similarity is None else float(similarity),
            strategy=row_strategy,
            kind=SectionKind(kind),
            section_path=path,
            citations=tuple(citations or ()),
        )
        for (
            chunk_id,
            celex,
            article_no,
            header,
            text,
            vector_rank,
            fts_rank,
            score,
            similarity,
            row_strategy,
            kind,
            path,
            citations,
            _nearest,
        ) in rows
    ]
    # Every row carries the same value, NULL when the vector side is empty. No
    # rows at all means both sides were empty.
    nearest = rows[0][-1] if rows else None
    return SearchResult(hits, None if nearest is None else float(nearest))


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
