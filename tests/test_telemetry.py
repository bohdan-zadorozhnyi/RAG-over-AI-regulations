"""Tracing: a no-op when off; with a client, the spans answer() records.

A fake stands in for the Langfuse client, so nothing reaches the network.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from regrag import telemetry
from regrag.answer import Generation, Source, _generate
from tests.test_answer import reply


class FakeSpan:
    def __init__(self, record: list[tuple[str, str, dict[str, object]]], name: str) -> None:
        self.record, self.name, self.trace_id = record, name, "trace-1"

    def update(self, **fields: object) -> "FakeSpan":
        self.record.append(("update", self.name, fields))
        return self

    def score_trace(self, *, name: str, value: float | str, **kwargs: object) -> None:
        self.record.append(("score_trace", name, {"value": value}))


class FakeLangfuse:
    def __init__(self) -> None:
        self.record: list[tuple[str, str, dict[str, object]]] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **fields: object) -> Iterator[FakeSpan]:
        name = str(fields["name"])
        self.record.append(("start", name, fields))
        yield FakeSpan(self.record, name)

    def create_score(self, **fields: object) -> None:
        self.record.append(("score", str(fields["name"]), fields))

    def flush(self) -> None:
        self.flushed = True


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuse:
    client = FakeLangfuse()
    monkeypatch.setattr(telemetry, "client", lambda: client)
    return client


def test_everything_is_a_no_op_when_tracing_is_off() -> None:
    assert telemetry.client() is None  # langfuse_enabled defaults to False
    with telemetry.trace("answer", input="q") as root, telemetry.observe("retrieve") as span:
        span.update(output=[1])
        root.score_trace(name="status", value="answer")
    assert root.trace_id is None
    telemetry.score(None, "behaviour_ok", 1.0)
    telemetry.flush()


def test_enabled_without_keys_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry.client.cache_clear()
    monkeypatch.setattr(telemetry.settings, "langfuse_enabled", True)
    monkeypatch.setattr(telemetry.settings, "langfuse_public_key", None)
    try:
        with pytest.raises(ValueError, match="REGRAG_LANGFUSE_PUBLIC_KEY"):
            telemetry.client()
    finally:
        telemetry.client.cache_clear()


def test_a_generation_records_prompt_usage_and_cost(fake: FakeLangfuse) -> None:
    sources = [Source("32016R0679:32", "Security of processing", "1. The controller", False)]
    generation = _generate("q", sources, lambda system, user: reply("Security [1]."))

    start = next(r for r in fake.record if r[0] == "start")
    assert start[1] == "generate" and start[2]["as_type"] == "generation"
    messages = start[2]["input"]
    assert isinstance(messages, list) and messages[1]["content"].endswith("<question>q</question>")
    update = next(r for r in fake.record if r[0] == "update")[2]
    assert update["usage_details"] == {"input": 4000, "output": 600}
    assert update["cost_details"] == {"total": pytest.approx((4000 * 0.10 + 600 * 0.50) / 1e6)}
    assert generation.text == reply("Security [1].").text


def test_a_cached_generation_costs_nothing(fake: FakeLangfuse) -> None:
    cached = Generation('{"answer": "", "missing": "", "status": "refuse"}', "m", 10, 5, None, True)
    _generate("q", [Source("X:1", None, "t", True)], lambda system, user: cached)
    update = next(r for r in fake.record if r[0] == "update")[2]
    assert update["cost_details"] == {"total": 0.0}
    assert update["metadata"] == {"stop_reason": None, "cached": True}


def test_scores_go_to_the_given_trace(fake: FakeLangfuse) -> None:
    telemetry.score("trace-9", "citation_recall", 0.5, comment="cd-01")
    telemetry.score(None, "ignored", 1.0)  # no trace (tracing was off for it)
    assert fake.record == [
        (
            "score",
            "citation_recall",
            {"trace_id": "trace-9", "name": "citation_recall", "value": 0.5, "comment": "cd-01"},
        )
    ]
