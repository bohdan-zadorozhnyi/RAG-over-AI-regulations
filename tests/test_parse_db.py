"""Database tests for parse storage.

Need a running Postgres with the schema applied (`make db`); skipped otherwise,
as in CI's quality job. Every test runs inside a transaction that is always
rolled back (force_rollback), so nothing is left in the database; the code
under test's own transaction() blocks become savepoints inside it.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from regrag import parse
from regrag.config import settings
from regrag.db import Connection
from regrag.manifest import write_manifest
from regrag.models import Document, Section, SectionKind
from regrag.parse import ParseStatus, parse_all, store_document, upsert_document
from tests.test_parse import MINI_DOC

TEST_CELEX = "TEST00000"  # never a real CELEX id, so it cannot collide with real rows

COUNT_DOCUMENTS = "SELECT count(*) FROM documents WHERE celex = %s"
SELECT_DOCUMENT = "SELECT title, source_url, retrieved_at, checksum FROM documents WHERE id = %s"


def make_doc(checksum: str = "v1") -> Document:
    return Document(
        celex=TEST_CELEX,
        title="Test Act",
        source_url="https://example.test/doc",
        checksum=checksum,
        retrieved_at=datetime(2026, 11, 21, 10, 0, tzinfo=UTC),
        size_bytes=100,
    )


def count(conn: Connection) -> int:
    row = conn.execute(COUNT_DOCUMENTS, (TEST_CELEX,)).fetchone()
    assert row is not None
    return int(row[0])


def test_upsert_inserts_once(conn: Connection) -> None:
    first = upsert_document(conn, make_doc())
    second = upsert_document(conn, make_doc())
    assert first == second
    assert count(conn) == 1


def test_upsert_updates_fields_and_keeps_id(conn: Connection) -> None:
    doc_id = upsert_document(conn, make_doc("v1"))
    newer = replace(
        make_doc("v2"),
        title="Renamed Act",
        retrieved_at=datetime(2026, 12, 1, tzinfo=UTC),
    )
    assert upsert_document(conn, newer) == doc_id

    row = conn.execute(SELECT_DOCUMENT, (doc_id,)).fetchone()
    assert row is not None
    assert row == (
        "Renamed Act",
        "https://example.test/doc",
        datetime(2026, 12, 1, tzinfo=UTC),
        "v2",
    )


# --- store_document -------------------------------------------------------

SELECT_SECTIONS = """
SELECT s.path, s.id, s.kind, s.article_no, s.paragraph_no, s.heading, s.text
FROM sections s JOIN documents d ON d.id = s.document_id
WHERE d.celex = %s ORDER BY s.path
"""
SELECT_CHECKSUM = "SELECT checksum FROM documents WHERE celex = %s"


def make_sections() -> list[Section]:
    return [
        Section("recital-1", SectionKind.RECITAL, None, None, None, "(1) Why."),
        Section("chapter-I/article-1", SectionKind.ARTICLE, "1", None, "Scope", "Rules."),
        Section(
            "chapter-I/article-2/paragraph-1", SectionKind.ARTICLE, "2", "1", "Duties", "1. Do."
        ),
        Section("annex-I/point-1", SectionKind.ANNEX, "Annex I", "1.", "List", "1. Item"),
    ]


def stored(conn: Connection) -> dict[str, tuple[object, ...]]:
    return {row[0]: row[1:] for row in conn.execute(SELECT_SECTIONS, (TEST_CELEX,))}


def test_store_inserts_all_fields(conn: Connection) -> None:
    result = store_document(conn, make_doc(), make_sections())
    assert (result.written, result.deleted) == (4, 0)
    rows = stored(conn)
    assert rows["chapter-I/article-2/paragraph-1"][1:] == ("article", "2", "1", "Duties", "1. Do.")
    assert rows["recital-1"][1:] == ("recital", None, None, None, "(1) Why.")


def test_rerun_writes_nothing_and_keeps_ids(conn: Connection) -> None:
    store_document(conn, make_doc(), make_sections())
    ids_before = {path: row[0] for path, row in stored(conn).items()}

    result = store_document(conn, make_doc(), make_sections())

    assert (result.written, result.deleted) == (0, 0)
    assert {path: row[0] for path, row in stored(conn).items()} == ids_before


def test_changed_text_updates_only_that_row(conn: Connection) -> None:
    store_document(conn, make_doc(), make_sections())
    ids_before = {path: row[0] for path, row in stored(conn).items()}
    sections = make_sections()
    sections[1] = replace(sections[1], text="Amended rules.")

    result = store_document(conn, make_doc("v2"), sections)

    assert (result.written, result.deleted) == (1, 0)
    rows = stored(conn)
    assert rows["chapter-I/article-1"][-1] == "Amended rules."
    assert {path: row[0] for path, row in rows.items()} == ids_before


def test_removed_path_is_deleted(conn: Connection) -> None:
    store_document(conn, make_doc(), make_sections())
    result = store_document(conn, make_doc("v2"), make_sections()[:-1])
    assert (result.written, result.deleted) == (0, 1)
    assert "annex-I/point-1" not in stored(conn)


class _BogusKind:
    value = "bogus"  # violates the CHECK constraint on sections.kind


def test_failure_leaves_document_and_sections_unchanged(conn: Connection) -> None:
    store_document(conn, make_doc("v1"), make_sections())
    sections = make_sections()
    sections[1] = replace(sections[1], text="Amended rules.")  # a change that must not land
    bad = [*sections, replace(sections[0], path="recital-9", kind=_BogusKind())]  # type: ignore[arg-type]

    with pytest.raises(psycopg.errors.CheckViolation):
        store_document(conn, make_doc("v2"), bad)

    assert conn.execute(SELECT_CHECKSUM, (TEST_CELEX,)).fetchone() == ("v1",)
    rows = stored(conn)
    assert len(rows) == 4
    assert rows["chapter-I/article-1"][-1] == "Rules."


# --- parse_all ------------------------------------------------------------

MINI_SECTIONS = 15  # sections in tests.test_parse.MINI_DOC


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data/raw stand-in holding MINI_DOC as the only document."""
    monkeypatch.setattr(settings, "celex_ids", [TEST_CELEX])
    (tmp_path / f"{TEST_CELEX}.html").write_text(MINI_DOC, encoding="utf-8")
    write_manifest(tmp_path, {TEST_CELEX: make_doc("v1")})
    return tmp_path


def test_first_run_parses_and_stores(conn: Connection, raw_dir: Path) -> None:
    [result] = parse_all(raw_dir, conn=conn)
    assert result.status is ParseStatus.PARSED
    assert (result.sections, result.written, result.deleted) == (MINI_SECTIONS, MINI_SECTIONS, 0)
    assert result.by_kind == {"recital": 2, "article": 7, "annex": 6}
    assert len(stored(conn)) == MINI_SECTIONS


def test_second_run_skips_without_reading_the_file(
    conn: Connection, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parse_all(raw_dir, conn=conn)

    def must_not_parse(path: Path) -> list[Section]:
        raise AssertionError(f"parse_document called for {path}")

    monkeypatch.setattr(parse, "parse_document", must_not_parse)
    [result] = parse_all(raw_dir, conn=conn)
    assert result.status is ParseStatus.SKIPPED
    assert (result.sections, result.written, result.deleted) == (MINI_SECTIONS, 0, 0)


def test_force_reparses_and_changes_nothing(conn: Connection, raw_dir: Path) -> None:
    parse_all(raw_dir, conn=conn)
    [result] = parse_all(raw_dir, force=True, conn=conn)
    assert result.status is ParseStatus.UPDATED
    assert (result.written, result.deleted) == (0, 0)


def test_new_checksum_reparses(conn: Connection, raw_dir: Path) -> None:
    parse_all(raw_dir, conn=conn)
    write_manifest(raw_dir, {TEST_CELEX: make_doc("v2")})

    [result] = parse_all(raw_dir, conn=conn)

    assert result.status is ParseStatus.UPDATED
    assert conn.execute(SELECT_CHECKSUM, (TEST_CELEX,)).fetchone() == ("v2",)


def test_missing_manifest_raises(conn: Connection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        parse_all(tmp_path, conn=conn)


def test_celex_missing_from_manifest_raises(
    conn: Connection, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "celex_ids", [TEST_CELEX, "32024R1689"])
    with pytest.raises(ValueError, match="32024R1689 is not in the manifest"):
        parse_all(raw_dir, conn=conn)
