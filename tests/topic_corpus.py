"""A tiny indexed corpus with a topic encoder, shared by retrieval and eval tests.

The topic encoder gives words controlled meanings: each topic owns one vector
dimension, and synonyms share it. That makes it possible to build chunks that
match a question semantically but not lexically ("primary" vs "alpha"), and
the reverse.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from regrag.chunk import chunk_all
from regrag.config import settings
from regrag.db import Connection
from regrag.embed import embed_pending
from regrag.encoder import Vectors
from regrag.models import SECTION_AWARE, Document, Section, SectionKind
from regrag.parse import store_document

TEST_CELEX = "TEST00000"
MODEL = "test/topic-encoder"

TOPICS = [
    {"alpha", "primary"},  # synonyms: same dimension
    {"beta", "secondary"},
    {"gamma"},
    {"delta"},
    {"epsilon"},
]


def topic_encoder(texts: list[str]) -> Vectors:
    rows = []
    for text in texts:
        words = [w.strip(".,:;?()").lower() for w in text.split()]
        v = np.zeros(settings.embedding_dim, dtype=np.float64)
        for dim, topic in enumerate(TOPICS):
            v[dim] = sum(w in topic for w in words)
        v[-1] = 1e-3  # never a zero vector
        rows.append(v / np.linalg.norm(v))
    return np.asarray(rows, dtype=np.float32)


def article(n: int, text: str) -> Section:
    return Section(
        f"chapter-I/article-{n}/paragraph-1", SectionKind.ARTICLE, str(n), "1", f"Heading {n}", text
    )


CORPUS = [
    Section("recital-1", SectionKind.RECITAL, None, None, None, "(1) Alpha alpha alpha context."),
    article(1, "1. Alpha obligations apply."),
    article(2, "1. Beta duties apply."),
    article(3, "1. Gamma rules apply."),
    article(4, "1. Delta reporting apply."),
    article(5, "1. Epsilon notices apply."),
    article(6, "1. Alpha and beta together."),
]


def use_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "celex_ids": [TEST_CELEX],
        "document_titles": {TEST_CELEX: "Test Act"},
        "embedding_model": MODEL,
        "chunk_target_tokens": 12,
        "chunk_overlap_tokens": 2,
        "chunk_strategy": SECTION_AWARE,
        "exclude_recitals": False,
        "fts_rank_normalization": 0,
        "vector_top_k": 20,
        "fts_top_k": 20,
        "fusion_top_k": 50,  # large: every candidate is returned, for exact comparisons
    }.items():
        monkeypatch.setattr(settings, name, value)


def index_corpus(conn: Connection) -> Connection:
    """Parse, chunk (both strategies) and embed CORPUS. Needs use_test_settings."""
    store_document(
        conn,
        Document(
            TEST_CELEX,
            "Test Act",
            "https://example.test",
            "v1",
            datetime(2026, 11, 21, tzinfo=UTC),
            1,
        ),
        CORPUS,
    )
    chunk_all(conn=conn, count=lambda t: len(t.split()))
    embed_pending(conn=conn, encoder=topic_encoder)
    return conn
