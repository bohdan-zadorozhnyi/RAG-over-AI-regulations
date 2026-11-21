"""CLI output formatting, with the pipeline functions replaced by fakes."""

import pytest
from typer.testing import CliRunner

from regrag import retrieve
from regrag.cli import _snippet, app
from regrag.models import Retrieved, SectionKind

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
        "fused_score": 0.0161,
        "strategy": "section_aware",
        "kind": SectionKind.ARTICLE,
        "section_path": "chapter-III/section-2/article-9/paragraph-2",
        "citations": ("32024R1689:9",),
    }
    fields.update(overrides)
    return Retrieved(**fields)  # type: ignore[arg-type]


def test_ask_prints_ranked_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None, bool]] = []

    def fake_search(question: str, strategy: str | None, exclude_recitals: bool) -> list[Retrieved]:
        calls.append((question, strategy, exclude_recitals))
        return [hit(), hit(chunk_id=2, header="", citations=("32024R1689:7", "32024R1689:8"))]

    monkeypatch.setattr(retrieve, "search", fake_search)
    result = runner.invoke(app, ["ask", "What does Article 9 require?", "--exclude-recitals"])

    assert result.exit_code == 0, result.output
    assert calls == [("What does Article 9 require?", None, True)]
    assert "1. 32024R1689:9   score 0.0161   vector #2  text #-" in result.output
    assert "Article 9: Risk management system, paragraph 2" in result.output
    assert "2. The system shall comprise: / (a) identification; / (b) estimation." in result.output
    assert "2. 32024R1689:7, 32024R1689:8" in result.output  # baseline: several provisions


def test_ask_reports_errors_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_search(question: str, strategy: str | None, exclude_recitals: bool) -> None:
        raise ValueError("no section_aware embeddings for model 'x'")

    monkeypatch.setattr(retrieve, "search", failing_search)
    result = runner.invoke(app, ["ask", "anything"])

    assert result.exit_code == 1
    assert "ask failed: no section_aware embeddings" in result.output


def test_snippet_cuts_at_a_word_boundary() -> None:
    text = "word " * 100
    snippet = _snippet(text, limit=30)
    assert snippet.endswith(" …") and len(snippet) <= 32
    assert "wor …" not in snippet
