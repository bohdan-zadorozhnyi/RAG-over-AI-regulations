"""Langfuse tracing, and the query_log fallback when it is disabled.

Cut Langfuse first if the weekend runs short — query_log alone still gives
you latency and retrieval numbers for the README.
"""

from contextlib import AbstractContextManager


def trace(name: str, **metadata: object) -> AbstractContextManager[None]:
    raise NotImplementedError
