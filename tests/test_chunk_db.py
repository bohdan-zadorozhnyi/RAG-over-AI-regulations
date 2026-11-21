"""Database tests for chunk storage. Use the rolled-back `conn` fixture from
conftest.py; skipped without Postgres. A word count stands in for the
tokenizer, with small budgets so the baseline produces several chunks."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from regrag.chunk import chunk_all
from regrag.config import settings
from regrag.db import Connection
from regrag.models import RECURSIVE, SECTION_AWARE, Document, Section, SectionKind
from regrag.parse import store_document

TEST_CELEX = "TEST00000"

SELECT_CHUNKS = """
SELECT c.content_hash, c.id
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND c.strategy = %s
"""
COUNT_LINKS = """
SELECT count(*), count(DISTINCT cs.chunk_id)
FROM chunk_sections cs
JOIN chunks c ON c.id = cs.chunk_id JOIN documents d ON d.id = c.document_id
WHERE d.celex = %s AND c.strategy = %s
"""


def words(text: str) -> int:
    return len(text.split())


def document(checksum: str = "v1") -> Document:
    return Document(
        celex=TEST_CELEX,
        title="Test Act",
        source_url="https://example.test/doc",
        checksum=checksum,
        retrieved_at=datetime(2026, 11, 21, tzinfo=UTC),
        size_bytes=100,
    )


def sections() -> list[Section]:
    return [
        Section("recital-1", SectionKind.RECITAL, None, None, None, "(1) " + "why " * 8),
        *(
            Section(
                f"chapter-I/article-{n}/paragraph-1",
                SectionKind.ARTICLE,
                str(n),
                "1",
                f"Heading {n}",
                "1. " + f"rule{n} " * 8,
            )
            for n in range(1, 6)
        ),
    ]


@pytest.fixture(autouse=True)
def small_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "celex_ids", [TEST_CELEX])
    monkeypatch.setattr(settings, "document_titles", {TEST_CELEX: "Test Act"})
    monkeypatch.setattr(settings, "chunk_target_tokens", 25)
    monkeypatch.setattr(settings, "chunk_overlap_tokens", 3)


@pytest.fixture
def stored_doc(conn: Connection) -> Connection:
    store_document(conn, document(), sections())
    return conn


def chunk_ids(conn: Connection, strategy: str) -> dict[str, int]:
    return dict(conn.execute(SELECT_CHUNKS, (TEST_CELEX, strategy)).fetchall())


def links(conn: Connection, strategy: str) -> tuple[int, int]:
    row = conn.execute(COUNT_LINKS, (TEST_CELEX, strategy)).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def test_first_run_stores_both_strategies(stored_doc: Connection) -> None:
    results = {r.strategy: r for r in chunk_all(conn=stored_doc, count=words)}

    aware = results[SECTION_AWARE]
    assert (aware.chunks, aware.inserted, aware.updated, aware.deleted) == (6, 6, 0, 0)
    assert links(stored_doc, SECTION_AWARE) == (6, 6)  # exactly one section per chunk

    base = results[RECURSIVE]
    assert base.inserted == base.chunks > 1
    total_links, linked_chunks = links(stored_doc, RECURSIVE)
    assert linked_chunks == base.chunks
    assert total_links > base.chunks  # some baseline chunks span several sections


def test_rerun_writes_nothing_and_keeps_ids(stored_doc: Connection) -> None:
    chunk_all(conn=stored_doc, count=words)
    before = {s: chunk_ids(stored_doc, s) for s in (SECTION_AWARE, RECURSIVE)}
    links_before = {s: links(stored_doc, s) for s in (SECTION_AWARE, RECURSIVE)}

    results = chunk_all(conn=stored_doc, count=words)

    assert all((r.inserted, r.updated, r.deleted) == (0, 0, 0) for r in results)
    assert {s: chunk_ids(stored_doc, s) for s in (SECTION_AWARE, RECURSIVE)} == before
    assert {s: links(stored_doc, s) for s in (SECTION_AWARE, RECURSIVE)} == links_before


def test_changed_section_replaces_only_its_chunk(stored_doc: Connection) -> None:
    chunk_all(SECTION_AWARE, conn=stored_doc, count=words)
    before = chunk_ids(stored_doc, SECTION_AWARE)
    edited = sections()
    edited[2] = replace(edited[2], text="1. " + "amended " * 8)
    store_document(stored_doc, document("v2"), edited)

    [result] = chunk_all(SECTION_AWARE, conn=stored_doc, count=words)

    assert (result.inserted, result.updated, result.deleted) == (1, 0, 1)
    after = chunk_ids(stored_doc, SECTION_AWARE)
    assert len(set(before.values()) & set(after.values())) == 5  # the rest kept their ids
    assert links(stored_doc, SECTION_AWARE) == (6, 6)


def test_single_strategy(stored_doc: Connection) -> None:
    results = chunk_all(RECURSIVE, conn=stored_doc, count=words)
    assert [r.strategy for r in results] == [RECURSIVE]
    assert chunk_ids(stored_doc, SECTION_AWARE) == {}


def test_unknown_strategy_raises(conn: Connection) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_all("semantic", conn=conn, count=words)


def test_missing_document_raises(conn: Connection) -> None:
    with pytest.raises(ValueError, match="run make parse"):
        chunk_all(conn=conn, count=words)
