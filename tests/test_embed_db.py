"""Database tests for embed. A deterministic fake encoder stands in for the
model; the rolled-back `conn` fixture comes from conftest.py (skipped without
Postgres)."""

import hashlib
from datetime import UTC, datetime

import numpy as np
import pytest

from regrag import embed
from regrag.chunk import chunk_all
from regrag.config import settings
from regrag.db import Connection
from regrag.embed import Vectors, embed_pending
from regrag.models import RECURSIVE, SECTION_AWARE, Document, Section, SectionKind
from regrag.parse import store_document

TEST_CELEX = "TEST00000"

COUNT_EMBEDDINGS = """
SELECT count(*) FROM embeddings e
JOIN chunks c ON c.id = e.chunk_id JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND e.model = %s
"""
SELECT_ROWS = """
SELECT e.model, e.dim, e.vector FROM embeddings e
JOIN chunks c ON c.id = e.chunk_id JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s
"""
COUNT_CHUNKS = """
SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND c.strategy = ANY(%s)
"""


class FakeEncoder:
    """Deterministic unit vectors seeded from each text's hash. Counts calls."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.texts: list[str] = []
        self.fail_on_call = fail_on_call

    def __call__(self, texts: list[str]) -> Vectors:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated crash mid-run")
        self.texts.extend(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(settings.embedding_dim)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32)


def words(text: str) -> int:
    return len(text.split())


def sections(changed: bool = False) -> list[Section]:
    texts = {n: "1. " + f"rule{n} " * 8 for n in range(1, 6)}
    if changed:
        texts[3] = "1. " + "amended " * 8
    return [
        Section("recital-1", SectionKind.RECITAL, None, None, None, "(1) " + "why " * 8),
        *(
            Section(
                f"chapter-I/article-{n}/paragraph-1",
                SectionKind.ARTICLE,
                str(n),
                "1",
                f"Heading {n}",
                text,
            )
            for n, text in texts.items()
        ),
    ]


def document(checksum: str) -> Document:
    return Document(
        TEST_CELEX,
        "Test Act",
        "https://example.test/doc",
        checksum,
        datetime(2026, 11, 21, tzinfo=UTC),
        100,
    )


@pytest.fixture(autouse=True)
def small_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "celex_ids", [TEST_CELEX])
    monkeypatch.setattr(settings, "document_titles", {TEST_CELEX: "Test Act"})
    monkeypatch.setattr(settings, "chunk_target_tokens", 25)
    monkeypatch.setattr(settings, "chunk_overlap_tokens", 3)
    monkeypatch.setattr(settings, "embedding_model", "test/fake-encoder")
    monkeypatch.setattr(settings, "embedding_batch_size", 4)


@pytest.fixture
def chunked(conn: Connection) -> Connection:
    store_document(conn, document("v1"), sections())
    chunk_all(conn=conn, count=words)
    return conn


def n_embeddings(conn: Connection, model: str = "test/fake-encoder") -> int:
    row = conn.execute(COUNT_EMBEDDINGS, (TEST_CELEX, model)).fetchone()
    assert row is not None
    return int(row[0])


def n_chunks(conn: Connection, strategies: list[str]) -> int:
    row = conn.execute(COUNT_CHUNKS, (TEST_CELEX, strategies)).fetchone()
    assert row is not None
    return int(row[0])


def test_first_run_embeds_every_chunk(chunked: Connection) -> None:
    encoder = FakeEncoder()
    results = {r.strategy: r for r in embed_pending(conn=chunked, encoder=encoder)}

    assert results[SECTION_AWARE].embedded == n_chunks(chunked, [SECTION_AWARE]) == 6
    assert results[RECURSIVE].embedded == n_chunks(chunked, [RECURSIVE])
    assert all(r.already == 0 for r in results.values())
    assert n_embeddings(chunked) == n_chunks(chunked, [SECTION_AWARE, RECURSIVE])


def test_rows_carry_model_dim_and_unit_vectors(chunked: Connection) -> None:
    embed_pending(conn=chunked, encoder=FakeEncoder())
    for model, dim, vector in chunked.execute(SELECT_ROWS, (TEST_CELEX,)):
        assert (model, dim) == ("test/fake-encoder", settings.embedding_dim)
        assert float(np.linalg.norm(vector.to_numpy())) == pytest.approx(1.0, abs=1e-5)


def test_embeds_header_plus_text(chunked: Connection) -> None:
    encoder = FakeEncoder()
    embed_pending(SECTION_AWARE, conn=chunked, encoder=encoder)
    assert "Test Act, Chapter I, Article 2: Heading 2, paragraph 1\n1. rule2" in "\n".join(
        encoder.texts
    )


def test_rerun_embeds_nothing_and_never_loads_the_model(
    chunked: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed_pending(conn=chunked, encoder=FakeEncoder())

    def must_not_load() -> None:
        raise AssertionError("model loaded with nothing pending")

    monkeypatch.setattr(embed, "load_encoder", must_not_load)
    results = embed_pending(conn=chunked)

    assert all(r.embedded == 0 and r.already > 0 for r in results)


def test_changed_section_embeds_only_new_chunks(chunked: Connection) -> None:
    embed_pending(SECTION_AWARE, conn=chunked, encoder=FakeEncoder())
    store_document(chunked, document("v2"), sections(changed=True))
    chunk_all(SECTION_AWARE, conn=chunked, count=words)

    encoder = FakeEncoder()
    [result] = embed_pending(SECTION_AWARE, conn=chunked, encoder=encoder)

    assert (result.embedded, result.already) == (1, 5)
    assert len(encoder.texts) == 1 and "amended" in encoder.texts[0]


def test_new_model_adds_rows_alongside(
    chunked: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    embed_pending(conn=chunked, encoder=FakeEncoder())
    first = n_embeddings(chunked)

    monkeypatch.setattr(settings, "embedding_model", "test/other-encoder")
    embed_pending(conn=chunked, encoder=FakeEncoder())

    assert n_embeddings(chunked, "test/other-encoder") == first
    assert n_embeddings(chunked) == first  # the old model's rows are untouched


def test_interrupted_run_resumes(chunked: Connection) -> None:
    total = n_chunks(chunked, [SECTION_AWARE, RECURSIVE])
    with pytest.raises(RuntimeError, match="simulated crash"):
        embed_pending(conn=chunked, encoder=FakeEncoder(fail_on_call=2))
    assert n_embeddings(chunked) == settings.embedding_batch_size  # first batch committed

    results = embed_pending(conn=chunked, encoder=FakeEncoder())

    assert sum(r.embedded for r in results) == total - settings.embedding_batch_size
    assert n_embeddings(chunked) == total


def test_dimension_mismatch_fails_before_writing(
    chunked: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "embedding_dim", 768)
    with pytest.raises(ValueError, match="needs a new migration"):
        embed_pending(conn=chunked, encoder=FakeEncoder())
    assert n_embeddings(chunked) == 0


def test_encoder_with_wrong_shape_fails(chunked: Connection) -> None:
    def bad_encoder(texts: list[str]) -> Vectors:
        return np.zeros((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="encoder returned shape"):
        embed_pending(conn=chunked, encoder=bad_encoder)
    assert n_embeddings(chunked) == 0


def test_chunk_too_long_for_model_fails_before_loading(
    chunked: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "embedding_max_seq_length", 5)
    with pytest.raises(ValueError, match="embedding_max_seq_length"):
        embed_pending(conn=chunked, encoder=FakeEncoder())


def test_no_chunks_raises(conn: Connection) -> None:
    with pytest.raises(ValueError, match="run make chunk"):
        embed_pending(conn=conn, encoder=FakeEncoder())


def test_single_strategy_leaves_the_other(chunked: Connection) -> None:
    [result] = embed_pending(RECURSIVE, conn=chunked, encoder=FakeEncoder())
    assert result.strategy == RECURSIVE
    assert n_embeddings(chunked) == n_chunks(chunked, [RECURSIVE])
