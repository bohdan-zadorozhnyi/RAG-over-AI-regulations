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
