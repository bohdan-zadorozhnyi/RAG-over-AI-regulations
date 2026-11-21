"""Row and result types passed between stages."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SectionKind(StrEnum):
    RECITAL = "recital"
    ARTICLE = "article"
    ANNEX = "annex"


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
    section_path: str
    ordinal: int
    header: str         # "AI Act, Chapter III, Article 9: Risk management, para 2"
    text: str
    token_count: int
    content_hash: str


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
