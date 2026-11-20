"""Step 2. XHTML to sections, preserving hierarchy.

The hard step and the one that decides retrieval quality. EUR-Lex XHTML
carries article structure as markup — walk it rather than regexing text.

Must distinguish recital / article / annex: recitals explain legislative
intent but are not binding, so excluding them is a real retrieval feature.

If this is still fighting you at the 4-hour mark, fall back to a simpler
article-boundary regex and move on. Downstream stages matter more.
"""

from pathlib import Path

from regrag.models import Section


def parse_document(html_path: Path) -> list[Section]:
    raise NotImplementedError


def store_sections(celex: str, sections: list[Section]) -> int:
    """Upsert on (document_id, path). Returns rows written."""
    raise NotImplementedError
