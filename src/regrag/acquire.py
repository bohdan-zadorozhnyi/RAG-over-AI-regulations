"""Step 1. Download source documents to data/raw/ with a manifest.

Idempotent: skip a document whose checksum is unchanged.
Manifest records celex, source url, retrieved_at and sha256 per document —
regulation gets amended, and an answer must be traceable to a version.
"""

from pathlib import Path

from regrag.models import Document


def fetch_all(raw_dir: Path) -> list[Document]:
    """Download every CELEX id in settings. Returns what was written or kept."""
    raise NotImplementedError


def fetch_one(celex: str, raw_dir: Path) -> Document:
    raise NotImplementedError


def write_manifest(raw_dir: Path, documents: list[Document]) -> None:
    raise NotImplementedError
