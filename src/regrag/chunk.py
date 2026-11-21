"""Step 3. Sections to chunks, two strategies.

`section_aware` — split at paragraph and article boundaries, never mid-article,
with a context header prepended so each chunk is interpretable alone.

`recursive` — LangChain RecursiveCharacterTextSplitter over each whole
document, no headers: the naive setup people actually build. Present only as
the eval baseline: the section-aware splitter has to beat it, and if it does
not, that is a finding worth reporting rather than hiding. Each baseline chunk
is linked (chunk_sections) to every section it overlaps, so the eval credits
it with every article it touches.

Token counts use the embedding model's own tokenizer, so a budget of N tokens
means what the model sees. Every function that measures text takes the
counter as an argument; tests pass a word count and need no download.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Protocol, cast

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.models import (
    SECTION_AWARE,
    Chunk,
    Section,
    SectionKind,
    select_strategies,
)

TokenCounter = Callable[[str], int]


class _Tokenizer(Protocol):
    """The one tokenizer method this module relies on."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@cache
def _tokenizer(model: str) -> _Tokenizer:
    """Load the tokenizer once per process. Downloads only tokenizer files, not
    model weights, on first use; loads from the Hugging Face cache, offline,
    afterwards."""
    from transformers import AutoTokenizer  # slow import: only when counting

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except OSError:  # not cached yet
        tokenizer = AutoTokenizer.from_pretrained(model)
    return cast(_Tokenizer, tokenizer)


def count_tokens(text: str) -> int:
    """Tokens the embedding model sees for `text`, excluding special tokens.

    Special tokens (<s>, </s>) are added once per embedded input, not per
    piece of text, so they are left out of chunk budgets.
    """
    ids: list[int] = _tokenizer(settings.embedding_model).encode(text, add_special_tokens=False)
    return len(ids)


# --- stage entry point ----------------------------------------------------

SELECT_DOCUMENT_ID = "SELECT id FROM documents WHERE celex = %(celex)s"

# Ordered by id, which follows document order: parse inserts sections in
# document order and upserts keep their ids.
SELECT_SECTIONS = """
SELECT id, path, kind, article_no, paragraph_no, heading, text
FROM sections WHERE document_id = %(document_id)s ORDER BY id
"""


@dataclass(frozen=True)
class ChunkResult:
    celex: str
    strategy: str
    chunks: int
    inserted: int
    updated: int  # kept chunks whose ordinal or start section moved
    deleted: int
    max_tokens: int


def chunk_all(
    strategy: str = "all",
    conn: Connection | None = None,
    count: TokenCounter = count_tokens,
) -> list[ChunkResult]:
    """Stage entry point: chunk every CELEX id in settings with one or both strategies.

    Always recomputes (chunking is fast) and writes only the difference, keyed
    on content hashes: unchanged chunks keep their ids and so their embeddings.
    `conn` and `count` are injectable for tests.
    """
    strategies = select_strategies(strategy)

    owns_conn = conn is None
    db = conn or connect()
    try:
        results = []
        for celex in settings.celex_ids:
            row = db.execute(SELECT_DOCUMENT_ID, {"celex": celex}).fetchone()
            if row is None:
                raise ValueError(f"{celex} has no documents row; run make parse")
            document_id = int(row[0])
            ids, sections = _load_sections(db, document_id)
            if not sections:
                raise ValueError(f"{celex} has no sections; run make parse")
            for name in strategies:
                chunks = _make_chunks(name, sections, celex, count)
                results.append(_store(db, celex, document_id, name, chunks, ids))
        return results
    finally:
        if owns_conn:
            db.close()


def _load_sections(conn: Connection, document_id: int) -> tuple[dict[str, int], list[Section]]:
    ids: dict[str, int] = {}
    sections: list[Section] = []
    for row in conn.execute(SELECT_SECTIONS, {"document_id": document_id}):
        section_id, path, kind, article_no, paragraph_no, heading, text = row
        ids[path] = section_id
        sections.append(Section(path, SectionKind(kind), article_no, paragraph_no, heading, text))
    return ids, sections


def _make_chunks(
    strategy: str, sections: list[Section], celex: str, count: TokenCounter
) -> list[Chunk]:
    if strategy == SECTION_AWARE:
        return section_aware(sections, celex, count, settings.chunk_target_tokens)
    return recursive_baseline(
        sections, count, settings.chunk_target_tokens, settings.chunk_overlap_tokens
    )


# --- storage --------------------------------------------------------------
#
# Per (document, strategy), in one transaction:
#   1. delete stored chunks whose content hash is no longer produced
#      (their embeddings and chunk_sections rows go with them, by cascade)
#   2. insert chunks with new hashes
#   3. update ordinal / start section of kept chunks where they moved
#   4. ensure every chunk_sections link exists
# A re-run with unchanged sections writes nothing.

SELECT_STORED_CHUNKS = """
SELECT id, content_hash, section_id, ordinal FROM chunks
WHERE document_id = %(document_id)s AND strategy = %(strategy)s
"""

DELETE_CHUNKS = "DELETE FROM chunks WHERE id = ANY(%(ids)s)"

INSERT_CHUNK = """
INSERT INTO chunks
    (document_id, section_id, strategy, ordinal, header, text, token_count, content_hash)
VALUES
    (%(document_id)s, %(section_id)s, %(strategy)s, %(ordinal)s, %(header)s, %(text)s,
     %(token_count)s, %(content_hash)s)
RETURNING id
"""

UPDATE_CHUNK_POSITION = """
UPDATE chunks SET section_id = %(section_id)s, ordinal = %(ordinal)s
WHERE id = %(id)s
"""

INSERT_CHUNK_SECTION = """
INSERT INTO chunk_sections (chunk_id, section_id) VALUES (%(chunk_id)s, %(section_id)s)
ON CONFLICT DO NOTHING
"""


def _store(
    conn: Connection,
    celex: str,
    document_id: int,
    strategy: str,
    chunks: list[Chunk],
    section_ids: dict[str, int],
) -> ChunkResult:
    scope = {"document_id": document_id, "strategy": strategy}
    with conn.transaction(), conn.cursor() as cur:
        stored = {
            content_hash: (chunk_id, section_id, ordinal)
            for chunk_id, content_hash, section_id, ordinal in cur.execute(
                SELECT_STORED_CHUNKS, scope
            )
        }
        wanted = {c.content_hash for c in chunks}

        stale = [chunk_id for h, (chunk_id, _, _) in stored.items() if h not in wanted]
        if stale:
            cur.execute(DELETE_CHUNKS, {"ids": stale})

        chunk_ids: dict[str, int] = {}
        moved: list[dict[str, int]] = []
        inserted = 0
        for chunk in chunks:
            section_id = section_ids[chunk.section_path]
            if chunk.content_hash in stored:
                chunk_id, old_section, old_ordinal = stored[chunk.content_hash]
                if (old_section, old_ordinal) != (section_id, chunk.ordinal):
                    moved.append(
                        {"id": chunk_id, "section_id": section_id, "ordinal": chunk.ordinal}
                    )
            else:
                row = cur.execute(
                    INSERT_CHUNK,
                    {
                        **scope,
                        "section_id": section_id,
                        "ordinal": chunk.ordinal,
                        "header": chunk.header,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "content_hash": chunk.content_hash,
                    },
                ).fetchone()
                if row is None:  # RETURNING always yields a row
                    raise RuntimeError(f"insert of chunk {chunk.content_hash[:12]} returned no id")
                chunk_id = int(row[0])
                inserted += 1
            chunk_ids[chunk.content_hash] = chunk_id

        if moved:
            cur.executemany(UPDATE_CHUNK_POSITION, moved)
        links = [
            {"chunk_id": chunk_ids[c.content_hash], "section_id": section_ids[path]}
            for c in chunks
            for path in c.covers
        ]
        if links:
            cur.executemany(INSERT_CHUNK_SECTION, links)

    return ChunkResult(
        celex=celex,
        strategy=strategy,
        chunks=len(chunks),
        inserted=inserted,
        updated=len(moved),
        deleted=len(stale),
        max_tokens=max((c.token_count for c in chunks), default=0),
    )


# --- headers --------------------------------------------------------------

_PATH_LABELS = {"chapter": "Chapter", "section": "Section", "title": "Title"}


def build_header(section: Section, celex: str) -> str:
    """Context header that makes a chunk readable on its own.

    'AI Act, Chapter III, Section 2, Article 9: Risk management system, paragraph 2'
    'AI Act, Chapter I, Article 3: Definitions, point (1)'
    'AI Act, Recital 12'
    'AI Act, Annex III: High-risk AI systems referred to in Article 6(2), point 5'
    """
    title = settings.document_titles[celex]
    if section.kind is SectionKind.RECITAL:
        return f"{title}, {_unit_label(section.path)}"

    parts = [title]
    unit = ""
    position = ""
    for segment in section.path.split("/"):
        name, _, value = segment.partition("-")
        if name in _PATH_LABELS:
            parts.append(f"{_PATH_LABELS[name]} {value}")
        elif name == "article":
            unit = f"Article {value}"
        elif name == "annex":
            unit = f"Annex {value}"
        elif name == "paragraph":
            position = f"paragraph {value}"
        elif name == "point":
            position = f"point {_point_text(section.paragraph_no, value)}"
        # "group-N" is not shown: the group heading is the chunk's first line

    unit_with_heading = f"{unit}: {section.heading}" if section.heading else unit
    parts.append(unit_with_heading)
    if position:
        parts.append(position)
    return ", ".join(parts)


def _unit_label(path: str) -> str:
    name, _, value = path.split("/")[-1].partition("-")
    return f"{name.capitalize()} {value}"


def _point_text(label: str | None, fallback: str) -> str:
    """'(1)' stays '(1)'; '5.' becomes '5' (the trailing dot reads oddly mid-sentence)."""
    return (label or fallback).rstrip(".")


# --- section-aware --------------------------------------------------------
#
# Almost every section fits one chunk. A section over budget is split into
# blocks (a top-level line plus its indented sub-lines), packed greedily, so
# breaks fall between top-level points where possible. The section's first
# line ("2. ... shall comprise:") is repeated at the top of each later chunk
# so a point like "(f)" still says what it belongs to; that stands in for the
# token overlap the baseline uses.

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def section_aware(
    sections: list[Section], celex: str, count: TokenCounter, max_tokens: int
) -> list[Chunk]:
    """One chunk per section, or several for a section over `max_tokens`.

    `max_tokens` covers the embedded text, header included.
    """
    chunks: list[Chunk] = []
    for section in sections:
        header = build_header(section, celex)
        texts = _split_section(section.text, header, count, max_tokens)
        for ordinal, text in enumerate(texts):
            chunks.append(
                Chunk(
                    section_path=section.path,
                    ordinal=ordinal,
                    header=header,
                    text=text,
                    token_count=count(f"{header}\n{text}"),
                    content_hash=_hash(section.path, header, text),
                    covers=(section.path,),
                )
            )
    return chunks


def _split_section(text: str, header: str, count: TokenCounter, max_tokens: int) -> list[str]:
    if count(f"{header}\n{text}") <= max_tokens:
        return [text]

    budget = max_tokens - count(f"{header}\n")
    lead, *rest = text.split("\n")
    # Repeat the lead-in only while it is short; a long first line would crowd
    # out the content it introduces.
    repeat_lead = count(lead) <= budget // 4

    pieces: list[str] = []
    current: list[str] = []

    def fits(lines: list[str]) -> bool:
        return count("\n".join(lines)) <= budget

    def start_new(unit: str) -> None:
        nonlocal current
        if current:
            pieces.append("\n".join(current))
        current = [lead, unit] if repeat_lead and fits([lead, unit]) else [unit]

    for block in _blocks([lead, *rest]):
        if fits([*current, *block]):
            current.extend(block)
            continue
        for unit in _units(block, count, budget):
            if fits([*current, unit]):
                current.append(unit)
            else:
                start_new(unit)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _blocks(lines: list[str]) -> list[list[str]]:
    """Group lines into blocks: a top-level line and the indented lines after it."""
    blocks: list[list[str]] = []
    for line in lines:
        if blocks and line.startswith(" "):
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def _units(block: list[str], count: TokenCounter, budget: int) -> list[str]:
    """Lines of a block, with any line over budget split into sentences, and any
    sentence still over budget split into words."""
    units: list[str] = []
    for line in block:
        if count(line) <= budget:
            units.append(line)
            continue
        for sentence in _SENTENCE_END.split(line):
            if count(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, count, budget))
    return units


def _split_words(text: str, count: TokenCounter, budget: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split(" "):
        if current and count(" ".join([*current, word])) > budget:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


# --- recursive baseline ---------------------------------------------------


def recursive_baseline(
    sections: list[Section], count: TokenCounter, chunk_size: int, overlap: int
) -> list[Chunk]:
    """LangChain's recursive splitter over one whole document, with defaults.

    `sections` are one document's sections in document order. No headers. Each
    chunk records every section whose text it overlaps (`covers`).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    separator = "\n\n"
    spans: list[tuple[int, int, str]] = []  # (start, end, path) in the joined text
    position = 0
    for section in sections:
        spans.append((position, position + len(section.text), section.path))
        position += len(section.text) + len(separator)
    joined = separator.join(s.text for s in sections)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=count
    )
    # Locate chunks ourselves rather than via add_start_index: LangChain
    # resumes its search at (previous start + length - chunk_overlap) counting
    # chunk_overlap in *characters*, which with a token length function lands
    # past the real start and returns -1. Chunks come out in order, each
    # starting after the previous one, so searching onward is exact.
    chunks: list[Chunk] = []
    search_from = 0
    for ordinal, text in enumerate(splitter.split_text(joined)):
        start = joined.find(text, search_from)
        if start < 0:
            raise ValueError(f"baseline chunk {ordinal} not found in the document text")
        search_from = start + 1
        end = start + len(text)
        covers = tuple(path for s, e, path in spans if s < end and e > start)
        chunks.append(
            Chunk(
                section_path=covers[0],
                ordinal=ordinal,
                header="",
                text=text,
                token_count=count(text),
                content_hash=_hash(*covers, text),
                covers=covers,
            )
        )
    return chunks


def _hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
