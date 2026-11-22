"""Shared fixtures."""

from collections.abc import Iterator

import psycopg
import pytest

from regrag.config import settings
from regrag.db import Connection


@pytest.fixture
def conn() -> Iterator[Connection]:
    """A connection inside a transaction that is always rolled back.

    Needs Postgres with the schema applied (`make db`); the test is skipped
    otherwise, as in CI's quality job. Transaction blocks in the code under
    test become savepoints inside this one, so nothing is left behind.
    """
    try:
        connection = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run make db")
    with connection, connection.transaction(force_rollback=True):
        yield connection


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never reach Langfuse or the Claude API, whatever .env enables.

    Settings load .env, which holds real keys on a developer machine; without
    this, a test calling answer() sends traces, and one that forgot to inject
    a generator would make a paid call.
    """
    from regrag import telemetry
    from regrag.answer import load_generator

    monkeypatch.setattr(settings, "langfuse_enabled", False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    # Held here: a test may monkeypatch telemetry.client with a fake.
    cached = (telemetry.client, load_generator)
    for function in cached:
        function.cache_clear()
    yield
    for function in cached:
        function.cache_clear()
