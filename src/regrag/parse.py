"""Step 2. XHTML to sections, preserving hierarchy.

The hard step and the one that decides retrieval quality. EUR-Lex XHTML
carries article structure as markup — walk it rather than regexing text.

Must distinguish recital / article / annex: recitals explain legislative
intent but are not binding, so excluding them is a real retrieval feature.

If this is still fighting you at the 4-hour mark, fall back to a simpler
article-boundary regex and move on. Downstream stages matter more.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.manifest import load_manifest
from regrag.models import Document, Section, SectionKind

# --- text extraction ------------------------------------------------------
#
# EUR-Lex renders every list point, (a), (b), 1., (12), as a two-cell table
# row: the label in the first cell and the text in the second. Plain .text()
# runs label and text together, so tables are flattened explicitly, one line
# per point, with nested points indented one level deeper.

_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "ul", "ol", "blockquote", "section", "h1", "h2", "h3", "h4", "h5", "h6"}
)
_SKIP_TAGS = frozenset({"script", "style", "col", "colgroup"})
_ROW_GROUP_TAGS = frozenset({"thead", "tbody", "tfoot"})
_FOOTNOTE_MARKER_CLASS = "oj-note-tag"  # inline "(1)" pointing at a footnote
_INDENT = "  "
_WHITESPACE = re.compile(r"\s+")


def element_text(node: LexborNode) -> str:
    """Clean text of an element.

    One line per block element. A list point becomes "(a) text"; its nested
    points and continuation paragraphs follow on lines indented by two spaces
    per level. Footnote markers are dropped and whitespace (including nbsp)
    is collapsed.
    """
    lines = _table_lines(node) if node.tag == "table" else _block_lines(node)
    return "\n".join(lines)


def _block_lines(node: LexborNode) -> list[str]:
    lines: list[str] = []
    inline: list[str] = []

    def flush() -> None:
        text = _normalise("".join(inline))
        if text:
            lines.append(text)
        inline.clear()

    for child in node.iter(include_text=True):
        if child.is_text_node:
            inline.append(child.text_content or "")
        elif not child.is_element_node or _is_skipped(child):
            continue
        elif child.tag == "table":
            flush()
            lines.extend(_table_lines(child))
        elif child.tag == "br":
            flush()
        elif child.tag in _BLOCK_TAGS:
            flush()
            lines.extend(_block_lines(child))
        else:
            inline.append(_inline_text(child))
    flush()
    return lines


def _inline_text(node: LexborNode) -> str:
    parts: list[str] = []
    for child in node.iter(include_text=True):
        if child.is_text_node:
            parts.append(child.text_content or "")
        elif child.is_element_node and not _is_skipped(child):
            parts.append(_inline_text(child))
    return "".join(parts)


def _table_lines(table: LexborNode) -> list[str]:
    lines: list[str] = []
    for row in _rows(table):
        # Rows are usually [label, body], sometimes [indent, label, body] with
        # an empty indent cell. Drop empty cells; the last one left is the body.
        cells = [cell_lines for cell in _cells(row) if (cell_lines := _block_lines(cell))]
        if not cells:
            continue
        *label_cells, body = cells
        if not label_cells:
            lines.extend(body)
            continue
        label = " ".join(line for cell in label_cells for line in cell)
        first, *rest = body
        lines.append(f"{label} {first}")
        lines.extend(_INDENT + line for line in rest)
    return lines


def _cells(row: LexborNode) -> list[LexborNode]:
    return [c for c in row.iter() if c.tag == "td"]


def _rows(table: LexborNode) -> Iterator[LexborNode]:
    """Rows of this table only, not of tables nested inside its cells."""
    for child in table.iter():
        if child.tag == "tr":
            yield child
        elif child.tag in _ROW_GROUP_TAGS:
            yield from (row for row in child.iter() if row.tag == "tr")


def _is_skipped(node: LexborNode) -> bool:
    if node.tag in _SKIP_TAGS:
        return True
    classes = (node.attributes.get("class") or "").split()
    return _FOOTNOTE_MARKER_CLASS in classes


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# --- sections -------------------------------------------------------------


#
# Structural rules, from a survey of all three documents:
#   - an article either has direct-child paragraph divs (id "009.002") and
#     nothing else besides its title, or has none
#   - an article without paragraph divs that is one lead-in <p> followed only
#     by list points (definitions, obligation lists) is split per point, with
#     the lead-in repeated so each point reads on its own
#   - only *direct* children count: DORA Article 60 quotes another regulation's
#     Article 79, including its paragraph divs, deep inside its own paragraphs
#   - an annex has two oj-doc-ti lines (label, title), optional group headings
#     (oj-ti-grseq-1), optional lead-in paragraphs, and one table per point

_PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")
_ARTICLE_TITLE_CLASSES = frozenset({"oj-ti-art", "eli-title"})
_ANNEX_TITLE_CLASS = "oj-doc-ti"
_GROUP_HEADING_CLASS = "oj-ti-grseq-1"
_STRUCTURE_NAMES = {"cpt": "chapter", "sct": "section", "tis": "title"}
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]")


def parse_document(html_path: Path) -> list[Section]:
    """All sections of one document, in document order. No database access."""
    tree = LexborHTMLParser(html_path.read_bytes())
    sections = parse_recitals(tree) + parse_articles(tree) + parse_annexes(tree)
    duplicates = sorted(p for p, n in Counter(s.path for s in sections).items() if n > 1)
    if duplicates:
        raise ValueError(f"{html_path.name}: duplicate section paths {duplicates[:10]}")
    return sections


def parse_recitals(tree: LexborHTMLParser) -> list[Section]:
    sections: list[Section] = []
    for recital in _top_level(tree, "rct_"):
        number = _id(recital).removeprefix("rct_")
        text = element_text(recital)
        if text:
            sections.append(
                Section(
                    path=f"recital-{number}",
                    kind=SectionKind.RECITAL,
                    article_no=None,
                    paragraph_no=None,
                    heading=None,
                    text=text,
                )
            )
    return sections


def parse_articles(tree: LexborHTMLParser) -> list[Section]:
    sections: list[Section] = []
    for article in _top_level(tree, "art_"):
        sections.extend(_article_sections(article))
    return sections


def _article_sections(article: LexborNode) -> list[Section]:
    article_no = _id(article).removeprefix("art_")
    base = _join_path(_structural_path(article), f"article-{article_no}")
    heading = _article_title(article)
    body = [c for c in _children(article) if not _classes(c) & _ARTICLE_TITLE_CLASSES]

    def section(path: str, paragraph_no: str | None, text: str) -> Section:
        return Section(
            path=path,
            kind=SectionKind.ARTICLE,
            article_no=article_no,
            paragraph_no=paragraph_no,
            heading=heading,
            text=text,
        )

    paragraphs = [c for c in body if _PARAGRAPH_ID.match(_id(c))]
    if paragraphs:
        result = []
        for para in paragraphs:
            number = str(int(_id(para)[4:]))  # "009.002" -> "2"
            result.append(section(f"{base}/paragraph-{number}", number, element_text(para)))
        return result

    if (
        len(body) >= 2
        and body[0].tag == "p"
        and all(c.tag == "table" for c in body[1:])
        and _all_points_labelled(body[1:])
    ):
        lead_in = element_text(body[0])
        result = []
        for table in body[1:]:
            label = _point_label(table)
            path = f"{base}/point-{_label_slug(label)}"
            result.append(section(path, label, _join_lines(lead_in, element_text(table))))
        return result

    text = _join_lines(*(element_text(c) for c in body))
    return [section(base, None, text)] if text else []


def parse_annexes(tree: LexborHTMLParser) -> list[Section]:
    sections: list[Section] = []
    for annex in _top_level(tree, "anx_"):
        sections.extend(_annex_sections(annex))
    return sections


def _annex_sections(annex: LexborNode) -> list[Section]:
    number = _id(annex).removeprefix("anx_")
    article_no = f"Annex {number}"
    children = _children(annex)
    titles = [c for c in children if _ANNEX_TITLE_CLASS in _classes(c)]
    heading = element_text(titles[1]) if len(titles) > 1 else None
    body = [c for c in children if _ANNEX_TITLE_CLASS not in _classes(c)]

    def section(path: str, paragraph_no: str | None, text: str) -> Section:
        return Section(
            path=path,
            kind=SectionKind.ANNEX,
            article_no=article_no,
            paragraph_no=paragraph_no,
            heading=heading,
            text=text,
        )

    tables = [c for c in body if c.tag == "table"]
    if not tables or not _all_points_labelled(tables):
        # No points, or unnumbered bullets ("—"): too small and unnamed to split.
        text = _join_lines(*(element_text(c) for c in body))
        return [section(f"annex-{number}", None, text)] if text else []

    result: list[Section] = []
    group: str | None = None
    group_no = 0
    lead_in: list[str] = []
    after_point = False  # a paragraph after a point starts a new lead-in
    for child in body:
        if _GROUP_HEADING_CLASS in _classes(child):
            group, group_no, lead_in, after_point = element_text(child), group_no + 1, [], False
        elif child.tag == "table":
            label = _point_label(child)
            prefix = f"annex-{number}/group-{group_no}" if group_no else f"annex-{number}"
            text = _join_lines(group, *lead_in, element_text(child))
            result.append(section(f"{prefix}/point-{_label_slug(label)}", label, text))
            after_point = True
        else:
            if after_point:
                lead_in, after_point = [], False
            lead_in.append(element_text(child))

    if lead_in and not after_point:
        # Text after the last point (a closing note): keep it with that point.
        last = result[-1]
        result[-1] = section(last.path, last.paragraph_no, _join_lines(last.text, *lead_in))
    return result


# --- structural helpers ---------------------------------------------------


def _top_level(tree: LexborHTMLParser, prefix: str) -> list[LexborNode]:
    """Elements whose id is '<prefix>N' (not 'art_9.tit_1'), not nested in another."""
    found = []
    for node in tree.css(f'div[id^="{prefix}"]'):
        if "." in _id(node) or _has_ancestor_with_prefix(node, prefix):
            continue
        found.append(node)
    return found


def _has_ancestor_with_prefix(node: LexborNode, prefix: str) -> bool:
    parent = node.parent
    while parent is not None:
        if _id(parent).startswith(prefix):
            return True
        parent = parent.parent
    return False


def _structural_path(article: LexborNode) -> str:
    """'cpt_III.sct_2' on the nearest chapter ancestor -> 'chapter-III/section-2'."""
    parent = article.parent
    while parent is not None:
        node_id = _id(parent)
        if node_id.startswith("cpt_"):
            parts = []
            for part in node_id.split("."):
                kind, _, value = part.partition("_")
                parts.append(f"{_STRUCTURE_NAMES.get(kind, kind)}-{value}")
            return "/".join(parts)
        parent = parent.parent
    return ""


def _article_title(article: LexborNode) -> str | None:
    for child in _children(article):
        if "eli-title" in _classes(child):
            return element_text(child) or None
    return None


def _point_label(table: LexborNode) -> str:
    """Label of the table's first row: its non-empty cells except the last."""
    first_row = next(_rows(table), None)
    if first_row is None:
        return ""
    texts = [t for t in (element_text(c) for c in _cells(first_row)) if t]
    return " ".join(texts[:-1])


def _all_points_labelled(tables: list[LexborNode]) -> bool:
    """True when every point has a usable label: '(1)', '1.', '(a)', not '—'."""
    return all(_label_slug(_point_label(t)) for t in tables)


def _label_slug(label: str) -> str:
    return _NON_ALNUM.sub("", label)


def _children(node: LexborNode) -> list[LexborNode]:
    return [c for c in node.iter() if c.is_element_node]


def _classes(node: LexborNode) -> set[str]:
    return set((node.attributes.get("class") or "").split())


def _id(node: LexborNode) -> str:
    return node.attributes.get("id") or ""


def _join_path(*parts: str) -> str:
    return "/".join(p for p in parts if p)


def _join_lines(*parts: str | None) -> str:
    return "\n".join(p for p in parts if p)


# --- stage entry point ----------------------------------------------------

SELECT_STORED_STATE = """
SELECT d.checksum, count(s.id)
FROM documents d LEFT JOIN sections s ON s.document_id = d.id
WHERE d.celex = %(celex)s
GROUP BY d.checksum
"""


class ParseStatus(StrEnum):
    PARSED = "parsed"  # first time this document is stored
    UPDATED = "updated"  # re-parsed: new checksum, or --force
    SKIPPED = "skipped"  # stored checksum matches the manifest; file not read


@dataclass(frozen=True)
class Parsed:
    celex: str
    status: ParseStatus
    sections: int  # sections parsed, or already stored when skipped
    written: int
    deleted: int
    by_kind: dict[str, int]


def parse_all(raw_dir: Path, force: bool = False, conn: Connection | None = None) -> list[Parsed]:
    """Stage entry point: parse every CELEX id in settings into sections rows.

    Skips a document whose stored checksum matches the manifest and which has
    sections. The checksum is written in the same transaction as the sections,
    so a match means that version was stored completely. Use `force` after
    changing the parser itself: the document checksum does not change then.
    `conn` is injectable for tests.
    """
    manifest = load_manifest(raw_dir)
    if not manifest:
        raise FileNotFoundError(f"no manifest in {raw_dir}; run make ingest")

    owns_conn = conn is None
    db = conn or connect()
    try:
        return [_parse_one(db, raw_dir, celex, manifest, force) for celex in settings.celex_ids]
    finally:
        if owns_conn:
            db.close()


def _parse_one(
    conn: Connection, raw_dir: Path, celex: str, manifest: dict[str, Document], force: bool
) -> Parsed:
    document = manifest.get(celex)
    if document is None:
        raise ValueError(f"{celex} is not in the manifest; run make ingest")

    stored = conn.execute(SELECT_STORED_STATE, {"celex": celex}).fetchone()
    if not force and stored is not None and stored[0] == document.checksum and stored[1] > 0:
        return Parsed(celex, ParseStatus.SKIPPED, int(stored[1]), 0, 0, {})

    sections = parse_document(raw_dir / f"{celex}.html")
    result = store_document(conn, document, sections)
    return Parsed(
        celex=celex,
        status=ParseStatus.PARSED if stored is None else ParseStatus.UPDATED,
        sections=len(sections),
        written=result.written,
        deleted=result.deleted,
        by_kind=dict(Counter(s.kind.value for s in sections)),
    )


# --- storage --------------------------------------------------------------
#
# documents.checksum records which version of the text the stored sections
# came from. It is written in the same transaction as the sections, so it is
# only ever set to a version whose sections were stored completely.

UPSERT_DOCUMENT = """
INSERT INTO documents (celex, title, source_url, retrieved_at, checksum)
VALUES (%(celex)s, %(title)s, %(source_url)s, %(retrieved_at)s, %(checksum)s)
ON CONFLICT (celex) DO UPDATE SET
    title        = EXCLUDED.title,
    source_url   = EXCLUDED.source_url,
    retrieved_at = EXCLUDED.retrieved_at,
    checksum     = EXCLUDED.checksum
RETURNING id
"""


def upsert_document(conn: Connection, document: Document) -> int:
    """Insert or update the documents row for one manifest entry. Returns its id.

    Keyed on celex, so re-running never duplicates and the id stays stable,
    which keeps the sections, chunks and embeddings that reference it.
    Runs in the caller's transaction.
    """
    row = conn.execute(
        UPSERT_DOCUMENT,
        {
            "celex": document.celex,
            "title": document.title,
            "source_url": document.source_url,
            "retrieved_at": document.retrieved_at,
            "checksum": document.checksum,
        },
    ).fetchone()
    if row is None:  # RETURNING always yields a row; this satisfies the type checker
        raise RuntimeError(f"upsert of document {document.celex} returned no id")
    document_id: int = row[0]
    return document_id


# Updates only rows whose content changed, so rowcount counts real changes and
# unchanged rows are not rewritten.
UPSERT_SECTION = """
INSERT INTO sections (document_id, path, kind, article_no, paragraph_no, heading, text)
VALUES (%(document_id)s, %(path)s, %(kind)s, %(article_no)s, %(paragraph_no)s,
        %(heading)s, %(text)s)
ON CONFLICT (document_id, path) DO UPDATE SET
    kind         = EXCLUDED.kind,
    article_no   = EXCLUDED.article_no,
    paragraph_no = EXCLUDED.paragraph_no,
    heading      = EXCLUDED.heading,
    text         = EXCLUDED.text
WHERE (sections.kind, sections.article_no, sections.paragraph_no, sections.heading,
       sections.text)
      IS DISTINCT FROM
      (EXCLUDED.kind, EXCLUDED.article_no, EXCLUDED.paragraph_no, EXCLUDED.heading,
       EXCLUDED.text)
"""

DELETE_STALE_SECTIONS = """
DELETE FROM sections
WHERE document_id = %(document_id)s AND NOT (path = ANY(%(paths)s))
"""


@dataclass(frozen=True)
class StoredSections:
    document_id: int
    written: int  # inserted or changed
    deleted: int  # paths no longer produced by the parser


def store_document(conn: Connection, document: Document, sections: list[Section]) -> StoredSections:
    """Store one document and its sections atomically.

    Either the document row (with its new checksum) and all its sections are
    written, or nothing is.
    """
    with conn.transaction():
        document_id = upsert_document(conn, document)
        written, deleted = store_sections(conn, document_id, sections)
    return StoredSections(document_id, written, deleted)


def store_sections(conn: Connection, document_id: int, sections: list[Section]) -> tuple[int, int]:
    """Upsert on (document_id, path), then delete paths that no longer exist.

    Upserting rather than delete-and-reinsert keeps section ids stable: chunks
    and embeddings reference sections with ON DELETE CASCADE, so a wholesale
    delete would throw away every embedding on each re-parse.
    Returns (rows inserted or changed, rows deleted). Runs in the caller's
    transaction.
    """
    with conn.cursor() as cur:
        cur.executemany(
            UPSERT_SECTION,
            [
                {
                    "document_id": document_id,
                    "path": s.path,
                    "kind": s.kind.value,  # the value ("article"), not the enum name
                    "article_no": s.article_no,
                    "paragraph_no": s.paragraph_no,
                    "heading": s.heading,
                    "text": s.text,
                }
                for s in sections
            ],
        )
        written = max(cur.rowcount, 0)
        cur.execute(
            DELETE_STALE_SECTIONS,
            {"document_id": document_id, "paths": [s.path for s in sections]},
        )
        deleted = max(cur.rowcount, 0)
    return written, deleted
