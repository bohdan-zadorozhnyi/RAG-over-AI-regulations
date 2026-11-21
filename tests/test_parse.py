"""Unit tests for parse. Snippets mirror real EUR-Lex markup; no database."""

from pathlib import Path

import pytest
from selectolax.lexbor import LexborHTMLParser, LexborNode

from regrag.models import Section, SectionKind
from regrag.parse import element_text, parse_document

RAW_DIR = Path("data/raw")


def node(html: str) -> LexborNode:
    found = LexborHTMLParser(f'<div id="x">{html}</div>').css_first('div[id="x"]')
    assert found is not None
    return found


def point(label: str, body: str) -> str:
    """One EUR-Lex list point: a two-cell table row."""
    return (
        '<table width="100%" border="0"><col width="4%"/><col width="96%"/><tbody><tr>'
        f'<td valign="top"><p class="oj-normal">{label}</p></td>'
        f'<td valign="top">{body}</td>'
        "</tr></tbody></table>"
    )


# --- element_text ---------------------------------------------------------


def test_paragraph_text() -> None:
    assert element_text(node('<p class="oj-normal">1. A risk management system.</p>')) == (
        "1. A risk management system."
    )


def test_each_block_is_a_line() -> None:
    html = "<p>First.</p><p>Second.</p>"
    assert element_text(node(html)) == "First.\nSecond."


def test_list_point_joins_label_and_text() -> None:
    html = "<p>It shall comprise:</p>" + point("(a)", "<p>the identification of risks;</p>")
    assert element_text(node(html)) == "It shall comprise:\n(a) the identification of risks;"


def test_consecutive_points_are_separate_lines() -> None:
    html = point("(a)", "<p>first;</p>") + point("(b)", "<p>second.</p>")
    assert element_text(node(html)) == "(a) first;\n(b) second."


def test_nested_points_are_indented() -> None:
    inner = point("(i)", "<p>sub one;</p>") + point("(ii)", "<p>sub two.</p>")
    html = point("1.", "<p>Biometrics:</p>" + point("(a)", "<p>remote identification:</p>" + inner))
    assert element_text(node(html)) == (
        "1. Biometrics:\n"
        "  (a) remote identification:\n"
        "    (i) sub one;\n"
        "    (ii) sub two."
    )


def test_continuation_paragraph_inside_point_is_indented() -> None:
    html = point(
        "(a)", "<p>remote biometric identification systems.</p><p>This shall not include X.</p>"
    )
    assert element_text(node(html)) == (
        "(a) remote biometric identification systems.\n  This shall not include X."
    )


def test_inline_markup_and_nbsp_are_flattened() -> None:
    html = (
        '<p>Regulation (EU)&nbsp;2016/679 of <span class="oj-italic">27   April</span> 2016</p>'
    )
    assert element_text(node(html)) == "Regulation (EU) 2016/679 of 27 April 2016"


def test_footnote_markers_are_dropped() -> None:
    html = (
        '<p>the principle of free movement'
        '<a class="oj-note-tag" href="#ntr1">(<span class="oj-super">1</span>)</a>.</p>'
    )
    assert element_text(node(html)) == "the principle of free movement."


def test_scripts_and_empty_blocks_are_ignored() -> None:
    html = "<script>var x = 1;</script><p>   </p><p>Text.</p>"
    assert element_text(node(html)) == "Text."


def test_recital_table() -> None:
    html = point("(1)", "<p>The purpose of this Regulation is to improve trade.</p>")
    assert element_text(node(html)) == "(1) The purpose of this Regulation is to improve trade."


def test_three_cell_row_with_empty_indent_cell() -> None:
    html = (
        "<table><tbody><tr><td> </td><td><p>1.</p></td>"
        "<td><span>Directive 2006/42/EC</span></td></tr></tbody></table>"
    )
    assert element_text(node(html)) == "1. Directive 2006/42/EC"


def test_element_text_called_on_a_table_itself() -> None:
    html = point("5.", "<p>Essential services:</p>" + point("(b)", "<p>credit scoring;</p>"))
    table = node(html).css_first("table")
    assert table is not None
    assert element_text(table) == "5. Essential services:\n  (b) credit scoring;"


# --- parse_document on a miniature document -------------------------------


def title(number: str, name: str) -> str:
    return (
        f'<p class="oj-ti-art">Article {number}</p>'
        f'<div class="eli-title" id="art_{number}.tit_1"><p class="oj-sti-art">{name}</p></div>'
    )


MINI_DOC = f"""<html><body><div id="enc_1">
<div class="eli-subdivision" id="rct_1">{point("(1)", "<p>Recital one.</p>")}</div>
<div class="eli-subdivision" id="rct_2">{point("(2)", "<p>Recital two.</p>")}</div>
<div id="cpt_I"><div class="eli-title" id="cpt_I.tit_1"><p>GENERAL</p></div>
  <div class="eli-subdivision" id="art_1">{title("1", "Subject matter")}
    <p class="oj-normal">This Regulation lays down rules.</p></div>
  <div class="eli-subdivision" id="art_2">{title("2", "Definitions")}
    <p class="oj-normal">For the purposes of this Regulation:</p>
    {point("(1)", "<p>'system' means a thing;</p>")}
    {point("(2)", "<p>'provider' means a person;</p>")}
    {point("(3)", "<p>'deployer' means a user.</p>")}</div>
</div>
<div id="cpt_III"><div id="cpt_III.sct_2">
  <div class="eli-subdivision" id="art_9">{title("9", "Risk management system")}
    <div id="009.001"><p class="oj-normal">1. A system shall be established.</p></div>
    <div id="009.002"><p class="oj-normal">2. It shall comprise:</p>
      {point("(a)", "<p>identification;</p>")}
      <div id="079.001"><p class="oj-normal">'1. A quoted article paragraph.</p></div>
    </div></div>
  <div class="eli-subdivision" id="art_10">{title("10", "Lists")}
    <p class="oj-normal">The following offences:</p>
    {point("—", "<p>terrorism;</p>")}{point("—", "<p>fraud.</p>")}</div>
</div></div>
<div class="eli-container" id="anx_I">
  <p class="oj-doc-ti">ANNEX I</p><p class="oj-doc-ti">List of legislation</p>
  <p class="oj-ti-grseq-1">Section A. First list</p>
  {point("1.", "<p>Directive A</p>")}{point("2.", "<p>Directive B</p>")}
  <p class="oj-ti-grseq-1">Section B. Second list</p>
  {point("3.", "<p>Regulation C</p>")}
</div>
<div class="eli-container" id="anx_III">
  <p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">High-risk AI systems</p>
  <p class="oj-normal">Systems in the following areas:</p>
  {point("1.", "<p>Biometrics</p>")}{point("5.", "<p>Essential services</p>")}
</div>
<div class="eli-container" id="anx_VI">
  <p class="oj-doc-ti">ANNEX VI</p><p class="oj-doc-ti">Internal control</p>
  <div class="oj-enumeration-spacing"><p>1. Verify the system.</p></div>
</div>
</div></body></html>"""


@pytest.fixture
def mini(tmp_path: Path) -> dict[str, Section]:
    doc = tmp_path / "mini.html"
    doc.write_text(MINI_DOC, encoding="utf-8")
    return {s.path: s for s in parse_document(doc)}


def test_mini_paths_in_document_order(mini: dict[str, Section]) -> None:
    assert list(mini) == [
        "recital-1",
        "recital-2",
        "chapter-I/article-1",
        "chapter-I/article-2/point-1",
        "chapter-I/article-2/point-2",
        "chapter-I/article-2/point-3",
        "chapter-III/section-2/article-9/paragraph-1",
        "chapter-III/section-2/article-9/paragraph-2",
        "chapter-III/section-2/article-10",
        "annex-I/group-1/point-1",
        "annex-I/group-1/point-2",
        "annex-I/group-2/point-3",
        "annex-III/point-1",
        "annex-III/point-5",
        "annex-VI",
    ]


def test_recital_fields(mini: dict[str, Section]) -> None:
    s = mini["recital-2"]
    assert (s.kind, s.article_no, s.paragraph_no, s.heading) == (
        SectionKind.RECITAL, None, None, None
    )
    assert s.text == "(2) Recital two."


def test_paragraph_fields(mini: dict[str, Section]) -> None:
    s = mini["chapter-III/section-2/article-9/paragraph-2"]
    assert (s.kind, s.article_no, s.paragraph_no, s.heading) == (
        SectionKind.ARTICLE, "9", "2", "Risk management system"
    )
    assert s.text.startswith("2. It shall comprise:\n(a) identification;")


def test_nested_quoted_paragraph_is_not_a_section(mini: dict[str, Section]) -> None:
    assert not any(s.paragraph_no == "79" for s in mini.values())
    # its text stays inside the paragraph that quotes it
    assert "quoted article" in mini["chapter-III/section-2/article-9/paragraph-2"].text


def test_single_paragraph_article(mini: dict[str, Section]) -> None:
    s = mini["chapter-I/article-1"]
    assert (s.paragraph_no, s.text) == (None, "This Regulation lays down rules.")


def test_definition_points_repeat_the_lead_in(mini: dict[str, Section]) -> None:
    s = mini["chapter-I/article-2/point-2"]
    assert (s.article_no, s.paragraph_no, s.heading) == ("2", "(2)", "Definitions")
    assert s.text == "For the purposes of this Regulation:\n(2) 'provider' means a person;"


def test_unnumbered_bullets_keep_article_whole(mini: dict[str, Section]) -> None:
    s = mini["chapter-III/section-2/article-10"]
    assert s.text == "The following offences:\n— terrorism;\n— fraud."


def test_annex_points_carry_group_heading(mini: dict[str, Section]) -> None:
    s = mini["annex-I/group-2/point-3"]
    assert (s.kind, s.article_no, s.paragraph_no, s.heading) == (
        SectionKind.ANNEX, "Annex I", "3.", "List of legislation"
    )
    assert s.text == "Section B. Second list\n3. Regulation C"


def test_annex_points_carry_lead_in(mini: dict[str, Section]) -> None:
    assert mini["annex-III/point-5"].text == (
        "Systems in the following areas:\n5. Essential services"
    )


def test_annex_without_points_is_one_section(mini: dict[str, Section]) -> None:
    s = mini["annex-VI"]
    assert (s.paragraph_no, s.heading) == (None, "Internal control")
    assert s.text == "1. Verify the system."


def test_duplicate_paths_raise(tmp_path: Path) -> None:
    doc = tmp_path / "dup.html"
    doc.write_text(
        '<div id="anx_I"><p class="oj-doc-ti">ANNEX I</p><p class="oj-doc-ti">T</p>'
        + point("1.", "<p>a</p>")
        + point("1.", "<p>b</p>")
        + "</div>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate section paths"):
        parse_document(doc)


# --- against the real documents (skipped until `make ingest` has run) -----

needs_ai_act = pytest.mark.skipif(
    not (RAW_DIR / "32024R1689.html").exists(), reason="run make ingest first"
)


@needs_ai_act
def test_real_article_9_paragraph_2_points_on_own_lines() -> None:
    tree = LexborHTMLParser((RAW_DIR / "32024R1689.html").read_bytes())
    para = tree.css_first('div[id="009.002"]')
    assert para is not None
    lines = element_text(para).splitlines()
    assert lines[0].startswith("2. The risk management system shall be understood")
    labels = [line.split(" ", 1)[0] for line in lines[1:]]
    assert labels == ["(a)", "(b)", "(c)", "(d)"]


@needs_ai_act
def test_real_article_3_definitions_one_per_line() -> None:
    tree = LexborHTMLParser((RAW_DIR / "32024R1689.html").read_bytes())
    article = tree.css_first('div[id="art_3"]')
    assert article is not None
    text = element_text(article)
    # EUR-Lex uses typographic quotes: \u2018 and \u2019
    assert "\n(1) \u2018AI system\u2019 means" in text
    assert "(1)\u2018AI system" not in text  # label and text never run together


needs_all = pytest.mark.skipif(
    not all((RAW_DIR / f"{c}.html").exists() for c in ("32024R1689", "32016R0679", "32022R2554")),
    reason="run make ingest first",
)

# celex -> (recitals, articles)
EXPECTED = {"32024R1689": (180, 113), "32016R0679": (173, 99), "32022R2554": (106, 64)}


@needs_all
@pytest.mark.parametrize("celex", list(EXPECTED))
def test_real_every_recital_and_article_present(celex: str) -> None:
    sections = parse_document(RAW_DIR / f"{celex}.html")
    recitals, articles = EXPECTED[celex]
    assert sum(s.kind is SectionKind.RECITAL for s in sections) == recitals
    found = {s.article_no for s in sections if s.kind is SectionKind.ARTICLE}
    assert found == {str(n) for n in range(1, articles + 1)}
    assert all(s.text for s in sections)


@needs_all
def test_real_article_9_paragraphs_in_order() -> None:
    sections = parse_document(RAW_DIR / "32024R1689.html")
    paras = [s.paragraph_no for s in sections if s.article_no == "9"]
    assert paras == [str(n) for n in range(1, 11)]


@needs_all
def test_real_ai_act_definition_point_1() -> None:
    by_path = {s.path: s for s in parse_document(RAW_DIR / "32024R1689.html")}
    text = by_path["chapter-I/article-3/point-1"].text
    assert text.startswith("For the purposes of this Regulation, the following definitions apply:")
    assert "\n(1) \u2018AI system\u2019 means" in text


@needs_all
def test_real_annex_iii_credit_scoring_point() -> None:
    by_path = {s.path: s for s in parse_document(RAW_DIR / "32024R1689.html")}
    text = by_path["annex-III/point-5"].text
    assert "\n  (b) AI systems intended to be used to evaluate the creditworthiness" in text


@needs_all
def test_real_dora_quoted_article_79_is_not_a_section() -> None:
    sections = parse_document(RAW_DIR / "32022R2554.html")
    assert not any("article-79" in s.path for s in sections)
