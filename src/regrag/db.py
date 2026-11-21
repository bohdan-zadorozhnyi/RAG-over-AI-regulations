"""Database connections. The URL comes from config; nothing else holds one."""

import psycopg
from psycopg.rows import TupleRow

from regrag.config import settings

Connection = psycopg.Connection[TupleRow]


def connect() -> Connection:
    """A new autocommit connection. Group writes with `with conn.transaction():`.

    Autocommit makes every transaction() block a real transaction that commits
    at its end. Without it, psycopg opens an implicit transaction on the first
    statement and a later transaction() block becomes a savepoint inside it,
    committed only if the connection is closed cleanly.
    """
    return psycopg.connect(settings.database_url, autocommit=True)
