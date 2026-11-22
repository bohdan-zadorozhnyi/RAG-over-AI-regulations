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
    path: str  # e.g. "chapter-3/section-2/article-9"
    kind: SectionKind
    article_no: str | None
    paragraph_no: str | None
    heading: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    section_path: str  # the section the chunk starts in
    ordinal: int
    header: str  # "AI Act, Chapter III, Article 9: Risk management, para 2"
    text: str
    token_count: int  # of the embedded text: header + newline + text
    content_hash: str
    covers: tuple[str, ...]  # every section path the text overlaps, in order


@dataclass(frozen=True)
class Retrieved:
    chunk_id: int
    celex: str
    article_no: str | None  # of the section the chunk starts in
    header: str
    text: str
    vector_rank: int | None  # None: not in the vector top-k
    fts_rank: int | None  # None: not in the full-text top-k
    reference_rank: int | None  # None: no provision it covers is named in the question
    pinned: bool  # the best chunk of a provision the question names; ranked first
    coverage: bool  # placed in the results' tail for a regulation the head left out
    rerank_score: float | None  # cross-encoder score; None when reranking is off
    fused_score: float
    similarity: float | None  # cosine similarity to the question; None: full-text only
    strategy: str
    kind: SectionKind  # of the section the chunk starts in
    section_path: str
    # Every provision the chunk overlaps, in document order: "32024R1689:9",
    # "32024R1689:Annex III", "32024R1689:Recital 12". One for section-aware
    # chunks; possibly several for baseline chunks.
    citations: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    hits: list[Retrieved]
    # Cosine similarity of the closest chunk in scope: the vector side's rank 1,
    # which fusion can push out of `hits`. The one absolute relevance signal
    # (fused scores depend only on ranks); eval measures how well it separates
    # in-scope from out-of-scope questions. None when nothing is in scope.
    nearest_similarity: float | None
