"""Database tests for search(). The rolled-back `conn` fixture comes from
conftest.py (skipped without Postgres); the corpus and topic encoder from
topic_corpus.py.
"""

import numpy as np
import pytest
from pgvector.psycopg import register_vector

from regrag.config import settings
from regrag.db import Connection
from regrag.models import RECURSIVE, Retrieved, SearchResult, SectionKind
from regrag.retrieve import SearchParams, reciprocal_rank_fusion, search
from tests.topic_corpus import TEST_CELEX, index_corpus, topic_encoder, use_test_settings

COUNT_TEST_CHUNKS = """
SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND c.strategy = %s
"""


@pytest.fixture(autouse=True)
def test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    use_test_settings(monkeypatch)


@pytest.fixture
def indexed(conn: Connection) -> Connection:
    return index_corpus(conn)


def run(conn: Connection, question: str, **overrides: object) -> SearchResult:
    params = SearchParams.from_settings(**overrides)  # type: ignore[arg-type]
    return search(question, params, conn=conn, encoder=topic_encoder)


def hits(conn: Connection, question: str, **overrides: object) -> list[Retrieved]:
    return run(conn, question, **overrides).hits


def test_sql_fusion_matches_the_python_reference(indexed: Connection) -> None:
    found = hits(indexed, "alpha gamma obligations")
    vector = [(h.chunk_id, h.vector_rank) for h in found if h.vector_rank is not None]
    fts = [(h.chunk_id, h.fts_rank) for h in found if h.fts_rank is not None]

    expected = reciprocal_rank_fusion([vector, fts], k=settings.rrf_k, top_n=len(found))

    assert [h.chunk_id for h in found] == [chunk_id for chunk_id, _ in expected]
    for hit, (_, score) in zip(found, expected, strict=True):
        assert hit.fused_score == pytest.approx(score)


def test_respects_fusion_top_k(indexed: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fusion_top_k", 3)
    assert len(hits(indexed, "alpha")) == 3


def test_semantic_only_match_comes_from_the_vector_side(indexed: Connection) -> None:
    # "primary" shares alpha's topic but appears in no chunk text
    top = hits(indexed, "primary")[0]
    assert top.fts_rank is None and top.vector_rank == 1
    assert "lpha" in top.text


def test_lexical_only_match_survives_the_outer_join(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "vector_top_k", 1)  # vector side: one alpha chunk only
    epsilon = next(h for h in hits(indexed, "alpha epsilon") if "Epsilon" in h.text)
    assert epsilon.fts_rank is not None
    assert epsilon.vector_rank is None and epsilon.similarity is None


def test_all_stopword_question_falls_back_to_vectors(indexed: Connection) -> None:
    found = hits(indexed, "what is it?")
    assert found and all(h.fts_rank is None for h in found)


def test_similarity_is_cosine_and_nearest_is_vector_rank_one(indexed: Connection) -> None:
    result = run(indexed, "alpha gamma obligations")
    vector_hits = sorted(
        (h for h in result.hits if h.vector_rank is not None), key=lambda h: h.vector_rank or 0
    )
    sims = [h.similarity for h in vector_hits]
    assert all(s is not None and -1.0 <= s <= 1.0 for s in sims)
    assert sims == sorted(sims, reverse=True)  # vector rank follows similarity
    assert result.nearest_similarity == pytest.approx(vector_hits[0].similarity)


def test_nearest_similarity_survives_fusion_dropping_vector_rank_one(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = run(indexed, "alpha gamma obligations")
    monkeypatch.setattr(settings, "fusion_top_k", 1)
    top_only = run(indexed, "alpha gamma obligations")
    assert top_only.nearest_similarity == pytest.approx(full.nearest_similarity)


def test_exclude_recitals(indexed: Connection) -> None:
    with_recitals = hits(indexed, "alpha")
    without = hits(indexed, "alpha", exclude_recitals=True)
    assert any(h.kind is SectionKind.RECITAL for h in with_recitals)
    assert without and all(h.kind is not SectionKind.RECITAL for h in without)


def test_exclude_recitals_default_comes_from_settings(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "exclude_recitals", True)
    assert all(h.kind is not SectionKind.RECITAL for h in hits(indexed, "alpha"))
    assert any(
        h.kind is SectionKind.RECITAL for h in hits(indexed, "alpha", exclude_recitals=False)
    )


@pytest.mark.parametrize("norm", [1, 2])
def test_rank_normalization_reaches_ts_rank(indexed: Connection, norm: int) -> None:
    # Articles 1 and 6 each contain "alpha" once, so unnormalised ts_rank ties
    # them and chunk id order puts Article 1 first. Article 6 has fewer
    # lexemes ("and" is a stopword), so dividing by length puts it ahead.
    def text_order(n: int) -> list[str]:
        found = [h for h in hits(indexed, "alpha", fts_rank_normalization=n) if h.fts_rank]
        return [h.section_path for h in sorted(found, key=lambda h: h.fts_rank or 0)]

    article_1, article_6 = "chapter-I/article-1/paragraph-1", "chapter-I/article-6/paragraph-1"
    assert text_order(0) == ["recital-1", article_1, article_6]
    assert text_order(norm) == ["recital-1", article_6, article_1]


def test_strategy_filter(indexed: Connection) -> None:
    found = hits(indexed, "alpha", strategy=RECURSIVE)
    assert found and all(h.strategy == RECURSIVE and h.header == "" for h in found)


def test_vector_side_returns_every_candidate_under_filters(indexed: Connection) -> None:
    # Planner-agnostic check that filters never shrink the vector side below
    # min(chunks, vector_top_k). On a corpus this small Postgres chooses an
    # exact scan, so the HNSW post-filtering problem itself is covered by
    # test_real_baseline_search_fills_vector_top_k below.
    row = indexed.execute(COUNT_TEST_CHUNKS, (TEST_CELEX, RECURSIVE)).fetchone()
    assert row is not None
    expected = min(int(row[0]), settings.vector_top_k)

    found = hits(indexed, "alpha", strategy=RECURSIVE)

    assert sum(h.vector_rank is not None for h in found) == expected


def test_section_aware_cites_one_provision(indexed: Connection) -> None:
    by_path = {h.section_path: h for h in hits(indexed, "alpha")}
    assert by_path["chapter-I/article-1/paragraph-1"].citations == (f"{TEST_CELEX}:1",)
    assert by_path["recital-1"].citations == (f"{TEST_CELEX}:Recital 1",)


def test_baseline_chunk_cites_every_article_it_spans(indexed: Connection) -> None:
    spanning = [h for h in hits(indexed, "alpha beta", strategy=RECURSIVE) if len(h.citations) > 1]
    assert spanning, "small budgets should make baseline chunks straddle articles"
    for hit in spanning:
        assert all(c.startswith(f"{TEST_CELEX}:") for c in hit.citations)
        assert len(set(hit.citations)) == len(hit.citations)


def test_refuses_when_model_has_no_embeddings(
    indexed: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "embedding_model", "amazon.titan-embed-text-v2")
    with pytest.raises(ValueError, match="no section_aware embeddings"):
        hits(indexed, "alpha")


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raises(conn: Connection, question: str) -> None:
    with pytest.raises(ValueError, match="empty question"):
        search(question, conn=conn, encoder=topic_encoder)


def test_params_default_to_settings_and_take_overrides() -> None:
    shipped = SearchParams.from_settings()
    assert shipped == SearchParams(settings.chunk_strategy, False, 0)
    assert SearchParams.from_settings(RECURSIVE, True, 2) == SearchParams(RECURSIVE, True, 2)
    assert shipped.label == "section_aware, recitals, norm 0"


@pytest.mark.parametrize("bad", [{"strategy": "all"}, {"fts_rank_normalization": -1}])
def test_params_reject_invalid_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchParams.from_settings(**bad)  # type: ignore[arg-type]


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

    found = search(
        "risk management system",
        SearchParams.from_settings(RECURSIVE),
        conn=conn,
        encoder=lambda texts: np.asarray([stored] * len(texts)),
    ).hits

    assert sum(h.vector_rank is not None for h in found) == settings.vector_top_k
