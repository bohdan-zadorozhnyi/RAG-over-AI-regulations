"""Row and result types passed between stages."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SectionKind(StrEnum):
    RECITAL = "recital"
    ARTICLE = "article"
    ANNEX = "annex"


# Chunking strategies, shared by chunk, embed and eval so no stage imports another.
SECTION_AWARE = "section_aware"
RECURSIVE = "recursive"
STRATEGIES = (SECTION_AWARE, RECURSIVE)


def select_strategies(strategy: str) -> tuple[str, ...]:
    """'all' -> every strategy; a strategy name -> just that one."""
    selected = STRATEGIES if strategy == "all" else (strategy,)
    unknown = set(selected) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategy {sorted(unknown)}; use one of {STRATEGIES} or all")
    return selected


@dataclass(frozen=True)
class Document:
    celex: str
    title: str
    source_url: str
    checksum: str
    retrieved_at: datetime
    size_bytes: int


@dataclass(frozen=True)
class Section:
    path: str           # e.g. "chapter-3/section-2/article-9"
    kind: SectionKind
    article_no: str | None
    paragraph_no: str | None
    heading: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    section_path: str   # the section the chunk starts in
    ordinal: int
    header: str         # "AI Act, Chapter III, Article 9: Risk management, para 2"
    text: str
    token_count: int    # of the embedded text: header + newline + text
    content_hash: str
    covers: tuple[str, ...]  # every section path the text overlaps, in order


@dataclass(frozen=True)
class Retrieved:
    chunk_id: int
    celex: str
    article_no: str | None
    header: str
    text: str
    vector_rank: int | None
    fts_rank: int | None
    fused_score: float
