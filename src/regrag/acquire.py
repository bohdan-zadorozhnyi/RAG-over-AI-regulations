"""Step 1. Download source documents to data/raw/ with a manifest.

Idempotent: skip a document whose checksum is unchanged.
regulation gets amended, and an answer must be traceable to a version.
Manifest records celex, source url, retrieved_at and sha256 per document —
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from selectolax.lexbor import LexborHTMLParser

from regrag.config import settings
from regrag.models import Document

ARTICLE_MARKER = b'id="art_'
MANIFEST_NAME = "manifest.json"


class FetchStatus(StrEnum):
    KEPT = "kept"             # no network call
    UNCHANGED = "unchanged"   # fetched (refresh), same content
    DOWNLOADED = "downloaded" # first time
    UPDATED = "updated"       # content changed 

@dataclass(frozen=True)
class Fetched:
    document: Document
    status: FetchStatus


def fetch_all(
    raw_dir: Path,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> list[Fetched]:
    """Stage entry point: fetch every CELEX id in settings, update the manifest.

    Results are merged into the existing manifest rather than replacing it, so
    a run over a subset (REGRAG_CELEX_IDS) keeps the other documents' entries.
    `client` is injectable for tests; by default one client is shared by all
    downloads.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)

    owns_client = client is None
    http = client or httpx.Client(
        transport=httpx.HTTPTransport(retries=settings.http_retries),
        timeout=settings.http_timeout_s,
        headers={"User-Agent": settings.http_user_agent},
        follow_redirects=True,
    )
    try:
        results = [
            fetch_one(http, celex, raw_dir, manifest.get(celex), refresh)
            for celex in settings.celex_ids
        ]
    finally:
        if owns_client:
            http.close()

    for result in results:
        manifest[result.document.celex] = result.document
    write_manifest(raw_dir, manifest)
    return results


def fetch_one(
    client: httpx.Client,
    celex: str,
    raw_dir: Path,
    known: Document | None,
    refresh: bool,
) -> Fetched:
    """Fetch one document, or keep the copy on disk when it is still valid.

    `retrieved_at` marks when a content version was first seen, so it only
    changes when the content checksum changes.
    """
    path = raw_dir / f"{celex}.html"

    # Recompute from the file rather than trusting the manifest: catches a
    # deleted, truncated or hand-edited file.
    on_disk_ok = (
        known is not None
        and path.exists()
        and content_checksum(path.read_bytes()) == known.checksum
    )
    if known is not None and on_disk_ok and not refresh:
        return Fetched(known, FetchStatus.KEPT)

    url = settings.eurlex_html_url.format(celex=celex)
    resp = client.get(url)
    resp.raise_for_status()
    body = resp.content  # bytes, not .text: decoding could change the checksum
    if not looks_like_document(body):
        raise ValueError(
            f"{celex}: response has no article markers "
            f"({resp.headers.get('content-type')}, {len(body)} bytes)"
        )

    checksum = content_checksum(body)
    if known is not None and checksum == known.checksum:
        if not on_disk_ok:
            path.write_bytes(body)
        return Fetched(known, FetchStatus.UNCHANGED)

    path.write_bytes(body)
    document = Document(
        celex=celex,
        title=settings.document_titles[celex],
        source_url=url,
        checksum=checksum,
        retrieved_at=datetime.now(UTC),
        size_bytes=len(body),
    )
    status = FetchStatus.DOWNLOADED if known is None else FetchStatus.UPDATED
    return Fetched(document, status)


def write_manifest(raw_dir: Path, documents: dict[str, Document]) -> None:
    """Atomic: write manifest.json.tmp, then replace()."""
    payload: dict[str, dict[str, Any]] = {}
    
    for celex, doc in documents.items():
        entry = dataclasses.asdict(doc)
        entry['retrieved_at'] = doc.retrieved_at.isoformat()
        payload[celex] = entry
        
    tmp = raw_dir / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(raw_dir / MANIFEST_NAME)


def load_manifest(raw_dir: Path) -> dict[str, Document]:
    """Empty dict if no manifest yet."""
    documents : dict[str, Document] = {}
    
    try:
        raw_text = (raw_dir / MANIFEST_NAME).read_text()
    except FileNotFoundError:
        return {}
    
    if not raw_text:
        return {}
        
    payload = json.loads(raw_text)
    for celex, entry in payload.items():
        entry['retrieved_at'] = datetime.fromisoformat(entry['retrieved_at'])    
        doc = Document(**entry)
        documents[celex] = doc
        
    return documents


def content_checksum(html: bytes) -> str:
    """
    sha256 of the page with <script> elements removed.

    EUR-Lex injects a monitoring script with per-request ids, so hashing raw
    bytes gives a new checksum on every download.
    """
    tree = LexborHTMLParser(html)
    
    for node in tree.css("script"):
        node.decompose()

    serialized = tree.html
    if serialized is None:
        raise ValueError("could not serialize HTML")
    
    return hashlib.sha256(serialized.encode()).hexdigest()


def looks_like_document(html: bytes) -> bool:
    return ARTICLE_MARKER in html
