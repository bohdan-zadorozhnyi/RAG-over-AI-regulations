"""Unit tests for acquire: pure functions and the manifest. No network."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from regrag.acquire import (
    MANIFEST_NAME,
    FetchStatus,
    content_checksum,
    fetch_all,
    fetch_one,
    load_manifest,
    looks_like_document,
    write_manifest,
)
from regrag.config import settings
from regrag.models import Document

CELEX = "32024R1689"  # must be a real id: fetch_one looks up its title in settings


def make_doc(celex: str = "32024R1689", checksum: str = "abc123") -> Document:
    return Document(
        celex=celex,
        title="AI Act",
        source_url=f"https://example.test/{celex}",
        checksum=checksum,
        retrieved_at=datetime(2026, 11, 21, 10, 0, tzinfo=UTC),
        size_bytes=1234,
    )


def page(script_id: str, article_text: str = "Article 1") -> bytes:
    return (
        f'<html><head><script data-x="{script_id}"></script></head>'
        f'<body><div id="art_1"><p>{article_text}</p></div></body></html>'
    ).encode()


# --- manifest -------------------------------------------------------------

def test_manifest_roundtrip(tmp_path: Path) -> None:
    docs = {"32024R1689": make_doc("32024R1689"), "32016R0679": make_doc("32016R0679")}
    write_manifest(tmp_path, docs)
    assert load_manifest(tmp_path) == docs


def test_load_manifest_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == {}


def test_roundtrip_keeps_timezone(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"32024R1689": make_doc()})
    loaded = load_manifest(tmp_path)["32024R1689"]
    assert loaded.retrieved_at.tzinfo is not None


def test_write_leaves_no_tmp_file(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"32024R1689": make_doc()})
    assert (tmp_path / MANIFEST_NAME).exists()
    assert not (tmp_path / f"{MANIFEST_NAME}.tmp").exists()


def test_write_replaces_previous_manifest(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"32024R1689": make_doc(checksum="old")})
    write_manifest(tmp_path, {"32024R1689": make_doc(checksum="new")})
    assert load_manifest(tmp_path)["32024R1689"].checksum == "new"


# --- checksum and validation ----------------------------------------------

def test_checksum_ignores_script_noise() -> None:
    assert content_checksum(page("r1")) == content_checksum(page("r2"))


def test_checksum_sees_content_change() -> None:
    assert content_checksum(page("r1", "Article 1")) != content_checksum(page("r1", "Amended"))


def test_looks_like_document() -> None:
    assert looks_like_document(page("r1"))
    assert not looks_like_document(b"<html><body>Access denied</body></html>")


# --- fetch_one ------------------------------------------------------------


class FakeEurLex:
    """MockTransport handler that serves a configurable body and counts calls."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.calls = 0

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            self.status, content=self.body, headers={"content-type": "text/html"}
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def test_first_fetch_downloads(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1"))
    result = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    assert result.status is FetchStatus.DOWNLOADED
    assert (tmp_path / f"{CELEX}.html").read_bytes() == page("r1")
    assert result.document.title == "AI Act"
    assert result.document.size_bytes == len(page("r1"))
    assert result.document.retrieved_at.tzinfo is not None
    assert server.calls == 1


def test_second_fetch_makes_no_call(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1"))
    first = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    second = fetch_one(server.client(), CELEX, tmp_path, known=first.document, refresh=False)
    assert second.status is FetchStatus.KEPT
    assert second.document == first.document
    assert server.calls == 1


def test_refresh_unchanged_keeps_retrieved_at(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1"))
    first = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    server.body = page("r2")  # only the script noise differs
    second = fetch_one(server.client(), CELEX, tmp_path, known=first.document, refresh=True)
    assert second.status is FetchStatus.UNCHANGED
    assert second.document.retrieved_at == first.document.retrieved_at
    assert server.calls == 2


def test_refresh_changed_updates(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1", "Article 1"))
    first = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    server.body = page("r1", "Amended")
    second = fetch_one(server.client(), CELEX, tmp_path, known=first.document, refresh=True)
    assert second.status is FetchStatus.UPDATED
    assert second.document.checksum != first.document.checksum
    assert (tmp_path / f"{CELEX}.html").read_bytes() == page("r1", "Amended")


def test_missing_file_redownloads(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1"))
    first = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    (tmp_path / f"{CELEX}.html").unlink()
    second = fetch_one(server.client(), CELEX, tmp_path, known=first.document, refresh=False)
    assert second.status is FetchStatus.UNCHANGED
    assert (tmp_path / f"{CELEX}.html").exists()
    assert server.calls == 2


def test_corrupted_file_redownloads(tmp_path: Path) -> None:
    server = FakeEurLex(page("r1"))
    first = fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    (tmp_path / f"{CELEX}.html").write_bytes(b"<html>trunc")
    fetch_one(server.client(), CELEX, tmp_path, known=first.document, refresh=False)
    assert (tmp_path / f"{CELEX}.html").read_bytes() == page("r1")
    assert server.calls == 2


def test_rejects_page_without_articles(tmp_path: Path) -> None:
    server = FakeEurLex(b"<html><body>Access denied</body></html>")
    with pytest.raises(ValueError, match="no article markers"):
        fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    assert not (tmp_path / f"{CELEX}.html").exists()


def test_http_error_raises_and_writes_nothing(tmp_path: Path) -> None:
    server = FakeEurLex(b"oops", status=503)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_one(server.client(), CELEX, tmp_path, known=None, refresh=False)
    assert not (tmp_path / f"{CELEX}.html").exists()


# --- fetch_all ------------------------------------------------------------

ALL_CELEX = ["32024R1689", "32016R0679", "32022R2554"]


def test_fetch_all_writes_files_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "celex_ids", ALL_CELEX)
    server = FakeEurLex(page("r1"))
    raw_dir = tmp_path / "raw"  # does not exist yet: fetch_all must create it

    results = fetch_all(raw_dir, client=server.client())

    assert [r.status for r in results] == [FetchStatus.DOWNLOADED] * 3
    assert sorted(load_manifest(raw_dir)) == sorted(ALL_CELEX)
    assert all((raw_dir / f"{c}.html").exists() for c in ALL_CELEX)


def test_fetch_all_rerun_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "celex_ids", ALL_CELEX)
    server = FakeEurLex(page("r1"))
    fetch_all(tmp_path, client=server.client())
    before = (tmp_path / MANIFEST_NAME).read_text()

    results = fetch_all(tmp_path, client=server.client())

    assert [r.status for r in results] == [FetchStatus.KEPT] * 3
    assert server.calls == 3
    assert (tmp_path / MANIFEST_NAME).read_text() == before


def test_fetch_all_subset_keeps_other_manifest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeEurLex(page("r1"))
    monkeypatch.setattr(settings, "celex_ids", ALL_CELEX)
    fetch_all(tmp_path, client=server.client())

    monkeypatch.setattr(settings, "celex_ids", ["32024R1689"])
    fetch_all(tmp_path, refresh=True, client=server.client())

    assert sorted(load_manifest(tmp_path)) == sorted(ALL_CELEX)


def test_fetch_all_failure_leaves_manifest_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "celex_ids", ALL_CELEX)
    server = FakeEurLex(page("r1"))
    fetch_all(tmp_path, client=server.client())
    before = (tmp_path / MANIFEST_NAME).read_text()

    server.status = 503
    with pytest.raises(httpx.HTTPStatusError):
        fetch_all(tmp_path, refresh=True, client=server.client())

    assert (tmp_path / MANIFEST_NAME).read_text() == before