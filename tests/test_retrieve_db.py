"""Database tests for search(). The rolled-back `conn` fixture comes from
conftest.py (skipped without Postgres).

A topic encoder gives words controlled meanings: each topic owns one vector
dimension, and synonyms share it. That makes it possible to build chunks that
match a question semantically but not lexically ("primary" vs "alpha"), and
the reverse, and to check the SQL fusion against the Python reference.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from pgvector.psycopg import register_vector

from regrag.chunk import chunk_all
from regrag.config import settings
from regrag.db import Connection
from regrag.embed import embed_pending
from regrag.encoder import Vectors
from regrag.models import RECURSIVE, SECTION_AWARE, Document, Section, SectionKind
from regrag.parse import store_document
from regrag.retrieve import reciprocal_rank_fusion, search

TEST_CELEX = "TEST00000"
MODEL = "test/topic-encoder"

COUNT_TEST_CHUNKS = """
SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND c.strategy = %s
"""

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


@pytest.fixture(autouse=True)
def test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "celex_ids": [TEST_CELEX],
        "document_titles": {TEST_CELEX: "Test Act"},
        "embedding_model": MODEL,
        "chunk_target_tokens": 12,
        "chunk_overlap_tokens": 2,
        "chunk_strategy": SECTION_AWARE,
        "vector_top_k": 20,
        "fts_top_k": 20,
        "fusion_top_k": 50,  # large: every candidate is returned, for exact comparisons
    }.items():
        monkeypatch.setattr(settings, name, value)


@pytest.fixture
def indexed(conn: Connection) -> Connection:
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


def test_sql_fusion_matches_the_python_reference(indexed: Connection) -> None:
    hits = search("alpha gamma obligations", conn=indexed, encoder=topic_encoder)
    vector = [(h.chunk_id, h.vector_rank) for h in hits if h.vector_rank is not None]
    fts = [(h.chunk_id, h.fts_rank) for h in hits if h.fts_rank is not None]

    expected = reciprocal_rank_fusion([vector, fts], k=settings.rrf_k, top_n=len(hits))

    assert [h.chunk_id for h in hits] == [chunk_id for chunk_id, _ in expected]
    for hit, (_, score) in zip(hits, expected, strict=True):
        assert hit.fused_score == pytest.approx(score)


def test_respects_fusion_top_k(indexed: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fusion_top_k", 3)
    assert len(search("alpha", conn=indexed, encoder=topic_encoder)) == 3


def test_semantic_only_match_comes_from_the_vector_side(indexed: Connection) -> None:
    # "primary" shares alpha's topic but appears in no chunk text
    hits = search("primary", conn=indexed, encoder=topic_encoder)
    top = hits[0]
    assert top.fts_rank is None and top.vector_rank == 1
    assert "lpha" in top.text


def test_lexical_only_match_survives_the_outer_join(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "vector_top_k", 1)  # vector side: one alpha chunk only
    hits = search("alpha epsilon", conn=indexed, encoder=topic_encoder)
    epsilon = next(h for h in hits if "Epsilon" in h.text)
    assert epsilon.fts_rank is not None


def test_all_stopword_question_falls_back_to_vectors(indexed: Connection) -> None:
    hits = search("what is it?", conn=indexed, encoder=topic_encoder)
    assert hits and all(h.fts_rank is None for h in hits)


def test_exclude_recitals(indexed: Connection) -> None:
    with_recitals = search("alpha", conn=indexed, encoder=topic_encoder)
    without = search("alpha", exclude_recitals=True, conn=indexed, encoder=topic_encoder)
    assert any(h.kind is SectionKind.RECITAL for h in with_recitals)
    assert without and all(h.kind is not SectionKind.RECITAL for h in without)


def test_strategy_filter(indexed: Connection) -> None:
    hits = search("alpha", strategy=RECURSIVE, conn=indexed, encoder=topic_encoder)
    assert hits and all(h.strategy == RECURSIVE and h.header == "" for h in hits)


def test_vector_side_returns_every_candidate_under_filters(indexed: Connection) -> None:
    # Planner-agnostic check that filters never shrink the vector side below
    # min(chunks, vector_top_k). On a corpus this small Postgres chooses an
    # exact scan, so the HNSW post-filtering problem itself is covered by
    # test_real_baseline_search_fills_vector_top_k below.
    row = indexed.execute(COUNT_TEST_CHUNKS, (TEST_CELEX, RECURSIVE)).fetchone()
    assert row is not None
    expected = min(int(row[0]), settings.vector_top_k)

    hits = search("alpha", strategy=RECURSIVE, conn=indexed, encoder=topic_encoder)

    assert sum(h.vector_rank is not None for h in hits) == expected


def test_section_aware_cites_one_provision(indexed: Connection) -> None:
    hits = search("alpha", conn=indexed, encoder=topic_encoder)
    by_path = {h.section_path: h for h in hits}
    assert by_path["chapter-I/article-1/paragraph-1"].citations == (f"{TEST_CELEX}:1",)
    assert by_path["recital-1"].citations == (f"{TEST_CELEX}:Recital 1",)


def test_baseline_chunk_cites_every_article_it_spans(indexed: Connection) -> None:
    hits = search("alpha beta", strategy=RECURSIVE, conn=indexed, encoder=topic_encoder)
    spanning = [h for h in hits if len(h.citations) > 1]
    assert spanning, "small budgets should make baseline chunks straddle articles"
    for hit in spanning:
        assert all(c.startswith(f"{TEST_CELEX}:") for c in hit.citations)
        assert len(set(hit.citations)) == len(hit.citations)


def test_refuses_when_model_has_no_embeddings(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "embedding_model", "amazon.titan-embed-text-v2")
    with pytest.raises(ValueError, match="no section_aware embeddings"):
        search("alpha", conn=indexed, encoder=topic_encoder)


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raises(conn: Connection, question: str) -> None:
    with pytest.raises(ValueError, match="empty question"):
        search(question, conn=conn, encoder=topic_encoder)


def test_all_is_not_a_search_strategy(indexed: Connection) -> None:
    with pytest.raises(ValueError):
        search("alpha", strategy="all", conn=indexed, encoder=topic_encoder)


# --- against the real corpus (skipped unless make embed has run) ----------

REAL_QUERY_VECTOR = """
SELECT e.vector FROM embeddings e
JOIN chunks c ON c.id = e.chunk_id
JOIN sections s ON s.id = c.section_id
JOIN documents d ON d.id = s.document_id
WHERE d.celex = '32024R1689' AND s.path = 'chapter-III/section-2/article-9/paragraph-1'
  AND c.strategy = 'section_aware' AND e.model = %s
"""


def test_real_baseline_search_fills_vector_top_k(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HNSW applies WHERE filters after collecting ef_search candidates, and
    baseline chunks are ~21% of rows. Measured on this corpus: without
    iterative scanning a baseline-only search for AI Act Article 9 got 4 of 20
    vector candidates at pgvector's default ef_search=40, and 0 at 10."""
    real_model = "BAAI/bge-m3"
    register_vector(conn)
    row = conn.execute(REAL_QUERY_VECTOR, (real_model,)).fetchone()
    if row is None:
        pytest.skip("real bge-m3 embeddings not present; run make embed")
    stored = np.asarray(row[0].to_numpy(), dtype=np.float32)

    for name, value in {
        "celex_ids": ["32024R1689", "32016R0679", "32022R2554"],
        "embedding_model": real_model,
        "fusion_top_k": 50,  # return every candidate so the vector side can be counted
        "hnsw_ef_search": 10,  # well below vector_top_k: only iterative scanning fills it
    }.items():
        monkeypatch.setattr(settings, name, value)

    hits = search(
        "risk management system",
        strategy=RECURSIVE,
        conn=conn,
        encoder=lambda texts: np.asarray([stored] * len(texts)),
    )

    assert sum(h.vector_rank is not None for h in hits) == settings.vector_top_k
