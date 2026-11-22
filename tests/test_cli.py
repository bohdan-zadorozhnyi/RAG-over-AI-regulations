"""CLI output formatting, with the pipeline functions replaced by fakes."""

import pytest
from typer.testing import CliRunner

from regrag import evaluate, retrieve
from regrag.cli import _snippet, app
from regrag.models import Retrieved, SearchResult, SectionKind
from regrag.retrieve import SearchParams

runner = CliRunner()


def hit(**overrides: object) -> Retrieved:
    fields: dict[str, object] = {
        "chunk_id": 1,
        "celex": "32024R1689",
        "article_no": "9",
        "header": "AI Act, Chapter III, Section 2, Article 9: Risk management system, paragraph 2",
        "text": "2. The system shall comprise:\n(a) identification;\n(b) estimation.",
        "vector_rank": 2,
        "fts_rank": None,
        "reference_rank": None,
        "pinned": False,
        "coverage": False,
        "fused_score": 0.0161,
        "similarity": 0.7123,
        "strategy": "section_aware",
        "kind": SectionKind.ARTICLE,
        "section_path": "chapter-III/section-2/article-9/paragraph-2",
        "citations": ("32024R1689:9",),
    }
    fields.update(overrides)
    return Retrieved(**fields)  # type: ignore[arg-type]


def test_ask_prints_ranked_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, SearchParams]] = []

    def fake_search(question: str, params: SearchParams) -> SearchResult:
        calls.append((question, params))
        second = hit(
            chunk_id=2,
            header="",
            similarity=None,
            reference_rank=1,
            pinned=True,
            citations=("32024R1689:7", "32024R1689:8"),
        )
        return SearchResult([hit(), second], nearest_similarity=0.7512)

    monkeypatch.setattr(retrieve, "search", fake_search)
    result = runner.invoke(app, ["ask", "What does Article 9 require?", "--exclude-recitals"])

    assert result.exit_code == 0, result.output
    [(question, params)] = calls
    assert question == "What does Article 9 require?" and params.exclude_recitals
    assert "nearest similarity 0.751" in result.output
    assert "1. 32024R1689:9   score 0.0161   vector #2  text #-   sim 0.712" in result.output
    assert "Article 9: Risk management system, paragraph 2" in result.output
    assert "2. The system shall comprise: / (a) identification; / (b) estimation." in result.output
    assert "2. 32024R1689:7, 32024R1689:8" in result.output  # baseline: several provisions
    assert "sim -   ref #1  pinned" in result.output  # a pinned reference hit


def test_ask_include_recitals_overrides_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[SearchParams] = []

    def fake_search(question: str, params: SearchParams) -> SearchResult:
        seen.append(params)
        return SearchResult([], None)

    monkeypatch.setattr(retrieve, "search", fake_search)
    monkeypatch.setattr(retrieve.settings, "exclude_recitals", True)
    runner.invoke(app, ["ask", "q"])
    runner.invoke(app, ["ask", "q", "--include-recitals"])

    assert [p.exclude_recitals for p in seen] == [True, False]


def test_ask_reports_errors_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_search(question: str, params: SearchParams) -> None:
        raise ValueError("no section_aware embeddings for model 'x'")

    monkeypatch.setattr(retrieve, "search", failing_search)
    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "ask failed: no section_aware embeddings" in result.output


def test_eval_reports_an_incomplete_question_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def incomplete(*args: object, **kwargs: object) -> None:
        raise ValueError("eval needs 10 questions per category; found direct_lookup 3")

    monkeypatch.setattr(evaluate, "run", incomplete)
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 1
    assert "eval failed: eval needs 10 questions per category" in result.output


def test_snippet_cuts_at_a_word_boundary() -> None:
    text = "word " * 100
    snippet = _snippet(text, limit=30)
    assert snippet.endswith(" …") and len(snippet) <= 32
    assert "wor …" not in snippet
