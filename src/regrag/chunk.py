"""Step 3. Sections to chunks, two strategies.

`section_aware` — split at paragraph and article boundaries, never mid-article,
with a context header prepended so each chunk is interpretable alone.

`recursive` — LangChain RecursiveCharacterTextSplitter. Present only as the
eval baseline: the section-aware splitter has to beat it, and if it does not,
that is a finding worth reporting rather than hiding.
"""

from regrag.models import Chunk, Section


def chunk_sections(sections: list[Section], strategy: str) -> list[Chunk]:
    raise NotImplementedError


def section_aware(sections: list[Section]) -> list[Chunk]:
    raise NotImplementedError


def recursive_baseline(sections: list[Section]) -> list[Chunk]:
    raise NotImplementedError


def build_header(section: Section, celex: str) -> str:
    """'AI Act, Chapter III, Article 9: Risk management system, paragraph 2'"""
    raise NotImplementedError
