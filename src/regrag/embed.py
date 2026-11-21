"""Step 4a. Generate and store embeddings.

Model name and dimension are stored on every row, so re-embedding with a
different model of the same dimension is a new row and both can be evaluated
side by side. Idempotent on (chunk_id, model).

The encoder is injectable (`Encoder`): tests and CI's offline fixture index
use a deterministic fake, so neither downloads the 2.2 GB model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pgvector.psycopg import register_vector

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.models import select_strategies

Vectors = NDArray[np.float32]  # shape (n_texts, dim), rows L2-normalised
Encoder = Callable[[list[str]], Vectors]

SPECIAL_TOKENS = 2  # <s> and </s>, added once per input; excluded from token_count


def embedding_text(header: str, text: str) -> str:
    """What gets embedded: exactly what chunk.token_count measured."""
    return f"{header}\n{text}" if header else text


def resolve_device(setting: str) -> str:
    """'auto' picks mps (Apple GPU), then cuda, then cpu; anything else is kept."""
    if setting != "auto":
        return setting
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_dimension(model_dim: int | None, configured_dim: int) -> None:
    """The model's output size must match config (and, checked at store time,
    the VECTOR column). A different size needs a migration, not a new row."""
    if model_dim != configured_dim:
        raise ValueError(
            f"{settings.embedding_model} produces {model_dim}-dim vectors but "
            f"embedding_dim is {configured_dim}; a different dimension needs a new "
            "migration for the VECTOR column, not just a config change"
        )


def check_fits(longest_chunk_tokens: int, max_seq_length: int) -> None:
    """Fail rather than let the model silently truncate a chunk."""
    needed = longest_chunk_tokens + SPECIAL_TOKENS
    if needed > max_seq_length:
        raise ValueError(
            f"longest chunk needs {needed} tokens but embedding_max_seq_length is "
            f"{max_seq_length}; raise it or lower chunk_target_tokens"
        )


@cache
def load_encoder() -> Encoder:
    """Load the configured model once per process.

    Downloads the model (about 2.2 GB for bge-m3) on first use into the
    Hugging Face cache, outside the repo.
    """
    from sentence_transformers import SentenceTransformer  # slow import

    device = resolve_device(settings.embedding_device)
    model = SentenceTransformer(settings.embedding_model, device=device)
    check_dimension(model.get_embedding_dimension(), settings.embedding_dim)
    if settings.embedding_precision == "fp16":
        model.half()
    model.max_seq_length = settings.embedding_max_seq_length

    def encode(texts: list[str]) -> Vectors:
        vectors: Any = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,  # cosine distance == dot product
            convert_to_numpy=True,
            show_progress_bar=False,  # the stage reports progress per batch
        )
        return np.asarray(vectors, dtype=np.float32)

    return encode


# --- stage entry point ----------------------------------------------------

# pgvector stores a VECTOR(n) column's dimension in atttypmod.
SELECT_COLUMN_DIM = """
SELECT atttypmod FROM pg_attribute
WHERE attrelid = 'embeddings'::regclass AND attname = 'vector'
"""

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
        _check_column(db)

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


def _check_column(conn: Connection) -> None:
    row = conn.execute(SELECT_COLUMN_DIM).fetchone()
    column_dim = int(row[0]) if row is not None else None
    if column_dim != settings.embedding_dim:
        raise ValueError(
            f"embeddings.vector is VECTOR({column_dim}) but embedding_dim is "
            f"{settings.embedding_dim}; a different dimension needs a new migration"
        )


def _store_batch(conn: Connection, batch: list[Any], encode: Encoder) -> None:
    vectors = encode([embedding_text(header, text) for _, _, header, text, _ in batch])
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
