"""data/raw/manifest.json: the contract between acquire and parse.

Acquire writes it; parse reads it. It lives in its own module so neither
stage imports the other.
"""

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from regrag.models import Document

MANIFEST_NAME = "manifest.json"


def write_manifest(raw_dir: Path, documents: dict[str, Document]) -> None:
    """Atomic: write manifest.json.tmp, then replace()."""
    payload: dict[str, dict[str, Any]] = {}

    for celex, doc in documents.items():
        entry = dataclasses.asdict(doc)
        entry["retrieved_at"] = doc.retrieved_at.isoformat()
        payload[celex] = entry

    tmp = raw_dir / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(raw_dir / MANIFEST_NAME)


def load_manifest(raw_dir: Path) -> dict[str, Document]:
    """Empty dict if no manifest yet."""
    documents: dict[str, Document] = {}

    try:
        raw_text = (raw_dir / MANIFEST_NAME).read_text()
    except FileNotFoundError:
        return {}

    if not raw_text:
        return {}

    payload = json.loads(raw_text)
    for celex, entry in payload.items():
        entry["retrieved_at"] = datetime.fromisoformat(entry["retrieved_at"])
        doc = Document(**entry)
        documents[celex] = doc

    return documents
