"""answer() end to end on the topic corpus: retrieval, a fake model, query_log."""

import json

import pytest

from regrag.answer import Generation, Generator, answer
from regrag.db import Connection
from tests.topic_corpus import TEST_CELEX, index_corpus, topic_encoder, use_test_settings

LOGGED = """
SELECT question, status, citations, model, input_tokens, output_tokens, answer,
       jsonb_array_length(retrieved)
FROM query_log WHERE id = (SELECT max(id) FROM query_log)
"""


@pytest.fixture(autouse=True)
def test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    use_test_settings(monkeypatch)


def fake_model(answer_text: str, status: str = "answer") -> tuple[list[str], Generator]:
    prompts: list[str] = []

    def generate(system: str, user: str) -> Generation:
        prompts.append(user)
        return Generation(
            json.dumps({"answer": answer_text, "missing": "", "status": status}),
            "claude-haiku-5-5",
            1200,
            150,
            "end_turn",
        )

    return prompts, generate


def test_answer_cites_the_retrieved_chunk_and_is_logged(conn: Connection) -> None:
    index_corpus(conn)
    prompts, generate = fake_model("Gamma rules apply [1].")
    reply = answer(
        "What gamma rules apply?",
        conn=conn,
        encoder=topic_encoder,
        generate=generate,
    )

    [prompt] = prompts
    assert "<question>What gamma rules apply?</question>" in prompt
    assert reply.hits[0].citations == (f"{TEST_CELEX}:3",)  # the gamma article ranks first
    assert reply.status == "answer" and reply.citations == [f"{TEST_CELEX}:3"]
    # The model read the whole provision, from sections, with its heading.
    assert reply.sources[0].provision == f"{TEST_CELEX}:3"
    assert reply.sources[0].heading == "Heading 3" and not reply.sources[0].excerpt
    assert "Test Act Article 3: Heading 3\n1. Gamma rules apply." in prompt

    row = conn.execute(LOGGED).fetchone()
    assert row is not None
    question, status, citations, model, tokens_in, tokens_out, text, n_retrieved = row
    assert (question, status, citations) == (
        "What gamma rules apply?",
        "answer",
        [f"{TEST_CELEX}:3"],
    )
    assert (model, tokens_in, tokens_out) == ("claude-haiku-5-5", 1200, 150)
    assert text == "Gamma rules apply [1]." and n_retrieved == len(reply.hits)


def test_eval_answers_are_not_logged(conn: Connection) -> None:
    index_corpus(conn)
    before = conn.execute("SELECT count(*) FROM query_log").fetchone()
    _, generate = fake_model("", status="refuse")
    reply = answer("Delta?", conn=conn, encoder=topic_encoder, generate=generate, log=False)

    assert reply.refused and reply.citations == []
    assert conn.execute("SELECT count(*) FROM query_log").fetchone() == before
