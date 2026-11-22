"""Build a small index in CI without network access or a GPU.

Checked-in excerpts plus a deterministic fake embedder, so the eval measures
retrieval and fusion logic rather than model quality. About two seconds.

corpus.json (written by export_corpus.py) holds whole provisions of the three
regulations: every one the eval questions expect, plus the realistic
confusers. This runs the real pipeline stages on it, with two substitutions
that need no download: chunk sizes are counted in words instead of bge-m3
tokens, and vectors come from the hashing encoder. Then `regrag eval` runs
unchanged, encoding its questions with the same hashing encoder because
embedding_model names it.

    REGRAG_EMBEDDING_MODEL=regrag/hashing-bow python -m tests.fixtures.build_index

Refuses a database that holds the real corpus: the fixture documents use the
real CELEX ids, so writing them there would replace the parsed sections.
"""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from regrag.chunk import chunk_all
from regrag.config import settings
from regrag.db import Connection, connect
from regrag.embed import embed_pending
from regrag.encoder import HASHING_MODEL, hashing_encoder
from regrag.index import build_indexes
from regrag.models import Document, Section, SectionKind
from regrag.parse import store_document

CORPUS = Path(__file__).with_name("corpus.json")
CHECKSUM_PREFIX = "fixture:"

REAL_DOCUMENTS = "SELECT celex FROM documents WHERE checksum NOT LIKE %(prefix)s"


def word_count(text: str) -> int:
    return len(text.split())


def check_target(conn: Connection) -> None:
    if settings.embedding_model != HASHING_MODEL:
        raise SystemExit(
            f"embedding_model is {settings.embedding_model!r}; set "
            f"REGRAG_EMBEDDING_MODEL={HASHING_MODEL} so the fixture's vectors are "
            "stored, and questions encoded, under the hashing encoder's name"
        )
    real = [row[0] for row in conn.execute(REAL_DOCUMENTS, {"prefix": CHECKSUM_PREFIX + "%"})]
    if real:
        raise SystemExit(
            f"{settings.database_url} holds the real corpus ({', '.join(real)}); "
            "build the fixture index in an empty database (make eval-fixture uses its own)"
        )


def load_corpus() -> list[tuple[Document, list[Section]]]:
    raw = CORPUS.read_text()
    data = json.loads(raw)
    built = []
    for doc in data["documents"]:
        sections = [
            Section(
                s["path"],
                SectionKind(s["kind"]),
                s["article_no"],
                s["paragraph_no"],
                s["heading"],
                s["text"],
            )
            for s in doc["sections"]
        ]
        digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        document = Document(
            celex=doc["celex"],
            title=doc["title"],
            source_url=doc["source_url"],
            checksum=CHECKSUM_PREFIX + digest[:16],
            retrieved_at=datetime(2026, 11, 22, tzinfo=UTC),
            size_bytes=len(raw),
        )
        built.append((document, sections))
    return built


def main() -> None:
    started = time.perf_counter()
    with connect() as conn:
        check_target(conn)
        corpus = load_corpus()
        for document, sections in corpus:
            store_document(conn, document, sections)
        chunks = chunk_all(conn=conn, count=word_count)
        embedded = embed_pending(conn=conn, encoder=hashing_encoder)
    build_indexes()
    by_strategy: dict[str, int] = {}
    for r in chunks:
        by_strategy[r.strategy] = by_strategy.get(r.strategy, 0) + r.chunks
    counts = ", ".join(f"{n} {strategy}" for strategy, n in by_strategy.items())
    print(
        f"fixture index: {len(corpus)} documents, "
        f"{sum(len(sections) for _, sections in corpus)} sections, chunks: {counts}; "
        f"{sum(e.embedded for e in embedded)} embedded with {HASHING_MODEL}; "
        f"{time.perf_counter() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
