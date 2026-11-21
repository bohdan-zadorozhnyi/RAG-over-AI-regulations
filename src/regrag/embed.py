"""Step 4a. Generate and store embeddings.

Model name and dimension are stored on every row, so re-embedding with a
different model of the same dimension is a new row and both can be evaluated
side by side. Idempotent on (chunk_id, model).

The model, text preparation and stored-state checks live in encoder.py, shared
with retrieval so questions and chunks are always encoded the same way.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pgvector.psycopg import register_vector

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder, check_fits, check_vector_column, chunk_input, load_encoder
from regrag.models import select_strategies

# --- stage entry point ----------------------------------------------------

# Scoped to settings.celex_ids, like parse and chunk, so a subset run
# (REGRAG_CELEX_IDS) embeds only those documents.
SELECT_PENDING = """
SELECT c.id, c.strategy, c.header, c.text, c.token_count
FROM chunks c
JOIN documents d ON d.id = c.document_id
LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = %(model)s
WHERE e.chunk_id IS NULL
  AND c.strategy = ANY(%(strategies)s)
  AND d.celex = ANY(%(celex_ids)s)
ORDER BY c.id
"""

COUNT_CHUNKS = """
SELECT c.strategy, count(*), count(e.chunk_id)
FROM chunks c
JOIN documents d ON d.id = c.document_id
LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = %(model)s
WHERE c.strategy = ANY(%(strategies)s) AND d.celex = ANY(%(celex_ids)s)
GROUP BY c.strategy
"""

INSERT_EMBEDDING = """
INSERT INTO embeddings (chunk_id, model, dim, vector)
VALUES (%(chunk_id)s, %(model)s, %(dim)s, %(vector)s)
ON CONFLICT (chunk_id, model) DO NOTHING
"""

Progress = Callable[[int, int], None]  # (embedded so far, total pending)


@dataclass(frozen=True)
class EmbedResult:
    strategy: str
    embedded: int  # this run
    already: int  # had a vector for this model before the run


def embed_pending(
    strategy: str = "all",
    conn: Connection | None = None,
    encoder: Encoder | None = None,
    progress: Progress | None = None,
) -> list[EmbedResult]:
    """Stage entry point: embed chunks lacking a vector for the configured model.

    Commits after every batch, so an interrupted run resumes where it stopped.
    The model is loaded only if something is pending, so a no-op re-run is
    instant. `conn` and `encoder` are injectable for tests.
    """
    strategies = list(select_strategies(strategy))
    model_name = settings.embedding_model
    scope = {"model": model_name, "strategies": strategies, "celex_ids": settings.celex_ids}

    owns_conn = conn is None
    db = conn or connect()
    try:
        register_vector(db)
        check_vector_column(db)

        counts = {s: (int(total), int(done)) for s, total, done in db.execute(COUNT_CHUNKS, scope)}
        if not counts:
            raise ValueError(f"no chunks for {strategies} in {settings.celex_ids}; run make chunk")
        pending = db.execute(SELECT_PENDING, scope).fetchall()

        embedded: dict[str, int] = dict.fromkeys(strategies, 0)
        if pending:
            check_fits(max(row[4] for row in pending), settings.embedding_max_seq_length)
            encode = encoder or load_encoder()
            size = settings.embedding_batch_size
            for start in range(0, len(pending), size):
                batch = pending[start : start + size]
                _store_batch(db, batch, encode)
                for row in batch:
                    embedded[row[1]] += 1
                if progress is not None:
                    progress(min(start + size, len(pending)), len(pending))

        return [
            EmbedResult(s, embedded[s], counts.get(s, (0, 0))[1]) for s in strategies
        ]
    finally:
        if owns_conn:
            db.close()


def _store_batch(conn: Connection, batch: list[Any], encode: Encoder) -> None:
    vectors = encode([chunk_input(header, text) for _, _, header, text, _ in batch])
    expected = (len(batch), settings.embedding_dim)
    if vectors.shape != expected:
        raise ValueError(f"encoder returned shape {vectors.shape}, expected {expected}")
    params = [
        {
            "chunk_id": chunk_id,
            "model": settings.embedding_model,
            "dim": settings.embedding_dim,
            "vector": vector,
        }
        for (chunk_id, *_), vector in zip(batch, vectors, strict=True)
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(INSERT_EMBEDDING, params)
