"""The embedding model, shared by embed (chunks) and retrieve (questions).

Questions must be encoded with the same model and the same text preparation
rules as the chunks they are compared against. A mismatch raises no error; it
just returns plausible-looking wrong results. So this module owns:

  - loading the model (one place, one configuration)
  - text preparation, with one explicitly named function per side:
    chunk_input (header prepended) and query_input (never a header)
  - the checks that the configured model matches what is stored

The encoder is injectable (`Encoder`): tests and CI's offline fixture index
use a deterministic fake, so neither downloads the 2.2 GB model. CI selects
it by name (embedding_model = HASHING_MODEL), so the unchanged `regrag eval`
encodes its questions with it too.
"""

import hashlib
import os
import re
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regrag.config import settings
from regrag.db import Connection

Vectors = NDArray[np.float32]  # shape (n_texts, dim), rows L2-normalised
Encoder = Callable[[list[str]], Vectors]

SPECIAL_TOKENS = 2  # <s> and </s>, added once per input; excluded from token_count


# --- text preparation: one function per side ------------------------------


def chunk_input(header: str, text: str) -> str:
    """Model input for a chunk: header + newline + text, exactly what
    chunk.token_count measured. Baseline chunks have no header."""
    return f"{header}\n{text}" if header else text


def query_input(question: str) -> str:
    """Model input for a question: the question alone.

    Never prepend a chunk-style header to a question. bge-m3 needs no query
    instruction prefix either, unlike older BGE English models.
    """
    return " ".join(question.split())


# --- offline encoder ------------------------------------------------------

# A model name no hub hosts: embeddings stored under it can never be mistaken
# for bge-m3's, and check_model_indexed keeps the two apart.
HASHING_MODEL = "regrag/hashing-bow"

_WORD = re.compile(r"[a-z0-9]+")
# Function words carry no topic; without IDF they would dominate the cosine.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its may of on or shall "  # noqa: SIM905
    "that the their this to under what when where which who with does do must "
    "should can not any such other".split()
)


def hashing_encoder(texts: list[str]) -> Vectors:
    """Bag of words hashed into embedding_dim buckets, log-scaled, normalised.

    Deterministic across runs and machines (md5, not Python's salted hash),
    no network, about a millisecond per chunk. Lexical only: in CI the vector
    side agrees with full-text search instead of adding meaning, so the
    fixture eval measures fusion, reference lookup and coverage logic, not
    model quality.
    """
    dim = settings.embedding_dim
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        counts: dict[int, int] = {}
        for word in _WORD.findall(text.lower()):
            if word not in _STOPWORDS:
                bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little") % dim
                counts[bucket] = counts.get(bucket, 0) + 1
        for bucket, n in counts.items():
            out[row, bucket] = 1.0 + np.log(n)
        out[row, -1] += 1e-3  # never a zero vector
        out[row] /= np.linalg.norm(out[row])
    return out


# --- model ----------------------------------------------------------------


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
    """The model's output size must match config (and the VECTOR column).
    A different size needs a migration, not a new row."""
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
    Hugging Face cache, outside the repo. Once cached it loads offline: the
    hub is not contacted for update checks. That is the effect of
    HF_HUB_OFFLINE=1, without the env var's catch that a fresh machine could
    never make the first download. HASHING_MODEL needs no model at all.
    """
    if settings.embedding_model == HASHING_MODEL:
        return hashing_encoder

    from sentence_transformers import SentenceTransformer  # slow import

    # bge-m3 ships pytorch_model.bin, and transformers then asks the hub for a
    # safetensors conversion on every load, in a background thread that
    # ignores local_files_only. Read at call time, so setting it here works.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    device = resolve_device(settings.embedding_device)
    try:
        model = SentenceTransformer(settings.embedding_model, device=device, local_files_only=True)
    except OSError:  # not cached yet
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
            show_progress_bar=False,  # callers report their own progress
        )
        return np.asarray(vectors, dtype=np.float32)

    return encode


# --- stored-state checks --------------------------------------------------

# pgvector stores a VECTOR(n) column's dimension in atttypmod.
SELECT_COLUMN_DIM = """
SELECT atttypmod FROM pg_attribute
WHERE attrelid = 'embeddings'::regclass AND attname = 'vector'
"""

SELECT_MODEL_INDEXED = """
SELECT EXISTS (
    SELECT 1 FROM embeddings e JOIN chunks c ON c.id = e.chunk_id
    WHERE e.model = %(model)s AND c.strategy = %(strategy)s
)
"""

SELECT_STORED_MODELS = "SELECT DISTINCT model FROM embeddings ORDER BY model"


def check_vector_column(conn: Connection) -> None:
    """embeddings.vector must be VECTOR(embedding_dim)."""
    row = conn.execute(SELECT_COLUMN_DIM).fetchone()
    column_dim = int(row[0]) if row is not None else None
    if column_dim != settings.embedding_dim:
        raise ValueError(
            f"embeddings.vector is VECTOR({column_dim}) but embedding_dim is "
            f"{settings.embedding_dim}; a different dimension needs a new migration"
        )


def check_model_indexed(conn: Connection, model: str, strategy: str) -> None:
    """Fail loudly unless chunks of `strategy` have embeddings from `model`.

    Comparing a question vector from one model against chunk vectors from
    another returns confident nonsense with no error, so search refuses to run.
    """
    row = conn.execute(SELECT_MODEL_INDEXED, {"model": model, "strategy": strategy}).fetchone()
    if row is not None and row[0]:
        return
    stored = [r[0] for r in conn.execute(SELECT_STORED_MODELS)]
    raise ValueError(
        f"no {strategy} embeddings for model {model!r}; stored models: "
        f"{stored or 'none'}. Run make embed, or set REGRAG_EMBEDDING_MODEL to the "
        "model the chunks were embedded with"
    )
