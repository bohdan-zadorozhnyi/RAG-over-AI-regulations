"""Answer rules as pure functions: prompt, citation tracing, refusal, cost."""

import json

import pytest

from regrag.answer import (
    MODEL_DECLINED,
    NOTHING_RETRIEVED,
    UNCITED,
    AnswerError,
    Generation,
    Source,
    build_prompt,
    build_sources,
    cited_source_numbers,
    compose,
    cost_usd,
    extract_citations,
    join_sections,
    parse_reply,
    provision_label,
    sources_from_chunks,
)
from regrag.models import Retrieved, SectionKind


def chunk(chunk_id: int, *citations: str, header: str = "", text: str = "") -> Retrieved:
    celex, _, number = citations[0].partition(":")
    return Retrieved(
        chunk_id=chunk_id,
        celex=celex,
        article_no=number,
        header=header or f"Header {chunk_id}",
        text=text or f"Text {chunk_id}.",
        vector_rank=chunk_id,
        fts_rank=None,
        reference_rank=None,
        pinned=False,
        coverage=False,
        rerank_score=None,
        fused_score=0.01,
        similarity=0.6,
        strategy="section_aware",
        kind=SectionKind.ARTICLE,
        section_path=f"article-{number}",
        citations=citations,
    )


HITS = [
    chunk(1, "32016R0679:32"),
    chunk(2, "32024R1689:10"),
    chunk(3, "32024R1689:Annex IV"),
]
SOURCES = sources_from_chunks(HITS)  # one per provision, in HITS order


def reply(answer: str, status: str = "answer", missing: str = "", **kw: object) -> Generation:
    fields: dict[str, object] = {
        "text": json.dumps({"answer": answer, "missing": missing, "status": status}),
        "model": "claude-haiku-5-5",
        "input_tokens": 4000,
        "output_tokens": 600,
        "stop_reason": "end_turn",
    }
    fields.update(kw)
    return Generation(**fields)  # type: ignore[arg-type]


def test_provision_labels_use_the_short_title() -> None:
    assert provision_label("32016R0679:32") == "GDPR Article 32"
    assert provision_label("32024R1689:Annex IV") == "AI Act Annex IV"
    assert provision_label("32024R1689:Recital 12") == "AI Act Recital 12"


def test_prompt_numbers_sources_before_the_question() -> None:
    sources = [
        Source("32016R0679:32", "Security of processing", "1. The controller shall", False),
        Source("32024R1689:5", "Prohibited AI practices", "1. (a) ...\n[...]\n(h)", True),
    ]
    prompt = build_prompt("What is required?", sources)
    assert prompt.index("<sources>") < prompt.index("<question>What is required?</question>")
    assert (
        '<source id="1" provision="GDPR Article 32">\n'
        "GDPR Article 32: Security of processing\n1. The controller shall\n</source>"
    ) in prompt
    assert '<source id="2" provision="AI Act Article 5" excerpt="true">' in prompt


def test_markers_resolve_in_order_and_out_of_range_ones_are_invalid() -> None:
    valid, invalid = cited_source_numbers("A [2]. B [1, 2]; C [9][3]. D [0].", n_sources=3)
    assert valid == [2, 1, 3]
    assert invalid == [9, 0]


def test_sources_are_provisions_merged_across_chunks() -> None:
    # A baseline chunk spanning two provisions: 32 joins source 1, 33 is new.
    sources = sources_from_chunks([*HITS, chunk(4, "32016R0679:32", "32016R0679:33")])
    assert [s.provision for s in sources] == [
        "32016R0679:32",
        "32024R1689:10",
        "32024R1689:Annex IV",
        "32016R0679:33",
    ]
    assert sources[0].text == "Text 1.\nText 4."
    assert extract_citations("x [4] y [1] z [7]", sources) == ["32016R0679:33", "32016R0679:32"]


def test_a_cited_answer_keeps_its_status_and_citations() -> None:
    a = compose("q", HITS, SOURCES, reply("Article 32 GDPR requires [1]. Annex IV lists [3]."), 900)
    assert a.status == "answer" and not a.refused and not a.downgraded
    assert a.citations == ["32016R0679:32", "32024R1689:Annex IV"]
    assert a.cited_sources == [1, 3]
    assert a.cost_usd == pytest.approx((4000 * 0.10 + 600 * 0.50) / 1e6)


def test_an_uncited_answer_is_withheld_as_a_refusal() -> None:
    a = compose("q", HITS, SOURCES, reply("The law requires security measures."), 900)
    assert a.refused and a.downgraded
    assert a.missing == UNCITED and a.citations == []


def test_only_invalid_markers_count_as_uncited() -> None:
    a = compose("q", HITS, SOURCES, reply("Requires security [5].", status="scoped"), 900)
    assert a.refused and a.downgraded and a.invalid_sources == [5]


def test_a_refusal_cites_nothing_even_if_it_mentions_a_source() -> None:
    a = compose(
        "q", HITS, SOURCES, reply("", status="refuse", missing="Source [1] is about security."), 9
    )
    assert a.refused and not a.downgraded
    assert a.citations == [] and a.cited_sources == []


def test_a_scoped_answer_keeps_what_is_missing() -> None:
    a = compose(
        "q", HITS, SOURCES, reply("AI Act 10 [2].", status="scoped", missing="GDPR side"), 9
    )
    assert a.status == "scoped" and a.missing == "GDPR side"


def test_nothing_retrieved_refuses_without_a_model_call() -> None:
    a = compose("q", [], [], None, 12)
    assert a.refused and a.model is None and a.cost_usd == 0.0
    assert a.missing == NOTHING_RETRIEVED


def test_a_safety_decline_is_a_refusal() -> None:
    assert parse_reply(reply("", stop_reason="refusal", text="")) == ("refuse", "", MODEL_DECLINED)


@pytest.mark.parametrize(
    ("text", "stop_reason", "message"),
    [
        ('{"answer": "cut', "max_tokens", "truncated"),
        ("not json", "end_turn", "not JSON"),
        ('{"answer": "", "missing": "", "status": "maybe"}', "end_turn", "schema"),
    ],
)
def test_unusable_replies_raise(text: str, stop_reason: str, message: str) -> None:
    with pytest.raises(AnswerError, match=message):
        parse_reply(reply("", text=text, stop_reason=stop_reason))


def test_cost_is_none_for_an_unpriced_model() -> None:
    assert cost_usd("some-other-model", 1000, 100) is None
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)


def test_join_drops_a_lead_in_repeated_from_the_previous_section() -> None:
    lead = "The Board shall advise the Commission. To that end, it may in particular: "
    assert join_sections([lead + "(a) collect expertise;", lead + "(b) contribute;"]) == (
        lead + "(a) collect expertise;\n(b) contribute;"
    )
    paragraphs = ["1. The controller shall act.", "2. The controller shall record."]
    assert (
        join_sections(paragraphs) == "1. The controller shall act.\n2. The controller shall record."
    )


ROWS = [(10 + i, "Heading", f"{i + 1}. Paragraph {i + 1}.") for i in range(6)]


def test_a_provision_that_fits_is_read_whole() -> None:
    [source] = build_sources([chunk(1, "X:7")], {"X:7": ROWS}, retrieved={13})
    assert not source.excerpt and source.heading == "Heading"
    assert source.text == "\n".join(f"{i}. Paragraph {i}." for i in range(1, 7))


def test_a_long_provision_is_cut_to_retrieved_paragraphs_and_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("regrag.answer.settings.answer_article_max_chars", 50)
    [source] = build_sources([chunk(1, "X:7")], {"X:7": ROWS}, retrieved={10, 14})
    assert source.excerpt
    assert source.text == (
        "1. Paragraph 1.\n2. Paragraph 2.\n[...]\n4. Paragraph 4.\n5. Paragraph 5.\n6. Paragraph 6."
    )


def test_the_context_cap_turns_later_provisions_into_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("regrag.answer.settings.answer_context_max_chars", 120)
    sections = {"X:1": ROWS, "X:2": [(30 + i, "Other", r[2]) for i, r in enumerate(ROWS)]}
    first, second = build_sources([chunk(1, "X:1"), chunk(2, "X:2")], sections, retrieved={10, 31})
    assert not first.excerpt
    assert second.excerpt and second.text.startswith("1. Paragraph 1.\n2. Paragraph 2.\n3.")


def test_a_provision_without_sections_is_read_as_its_chunks() -> None:
    [source] = build_sources([chunk(1, "X:Recital 12", text="(12) Context.")], {}, set())
    assert source.excerpt and source.text == "(12) Context." and source.heading is None
