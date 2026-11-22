"""Langfuse tracing (Langfuse Cloud), beside query_log.

query_log is the durable record in our own database. Langfuse is the trace
UI: one trace per question, with a retriever span (what was found, what the
model read) and a generation (prompt, reply, tokens, cost), a status score,
and, for answer-eval runs, the eval's scores grouped into one session.

Off unless langfuse_enabled is set with both keys; then every call here is
a no-op, so tests, CI and keyless machines never import or reach Langfuse.
The SDK exports in a background thread: a slow or failing Langfuse never
fails an answer. Short-lived commands call flush() before exiting.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from regrag.config import settings

if TYPE_CHECKING:
    from langfuse import Langfuse

ObservationType = Literal["span", "retriever", "generation"]


class Observation(Protocol):
    """What the code needs from a Langfuse span; the no-op implements it too."""

    trace_id: str | None

    def update(self, **kwargs: object) -> object: ...

    def score_trace(self, *, name: str, value: float | str, **kwargs: object) -> None: ...


class _Noop:
    trace_id: str | None = None

    def update(self, **kwargs: object) -> "_Noop":
        return self

    def score_trace(self, *, name: str, value: float | str, **kwargs: object) -> None:
        return None


NOOP = _Noop()


@cache
def client() -> "Langfuse | None":
    """The Langfuse client, or None when tracing is off. Created once."""
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or settings.langfuse_secret_key is None:
        raise ValueError(
            "langfuse_enabled needs REGRAG_LANGFUSE_PUBLIC_KEY and "
            "REGRAG_LANGFUSE_SECRET_KEY in .env"
        )
    from langfuse import Langfuse  # deferred: never imported when off

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


@contextmanager
def trace(
    name: str,
    input: object,
    tags: Sequence[str] = (),
    session_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Iterator[Observation]:
    """The root span of one trace; trace-level tags, session and metadata
    apply to every observation opened inside it."""
    langfuse = client()
    if langfuse is None:
        yield NOOP
        return
    from langfuse import propagate_attributes

    with (
        langfuse.start_as_current_observation(name=name, as_type="span", input=input) as root,
        propagate_attributes(
            trace_name=name, tags=list(tags), session_id=session_id, metadata=metadata
        ),
    ):
        yield cast(Observation, root)


@contextmanager
def observe(
    name: str, as_type: ObservationType = "span", **fields: object
) -> Iterator[Observation]:
    """A child observation of the current trace."""
    langfuse = client()
    if langfuse is None:
        yield NOOP
        return
    # as_type is chosen at run time, which the SDK's per-type overloads can't type.
    start: Any = langfuse.start_as_current_observation
    with start(name=name, as_type=as_type, **fields) as obs:
        yield cast(Observation, obs)


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach a numeric score to a finished trace (eval results)."""
    langfuse = client()
    if langfuse is None or trace_id is None:
        return
    langfuse.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def trace_url(trace_id: str | None) -> str | None:
    langfuse = client()
    if langfuse is None or trace_id is None:
        return None
    url: str | None = langfuse.get_trace_url(trace_id=trace_id)
    return url


def flush() -> None:
    """Send buffered spans; call before a short-lived command exits."""
    langfuse = client()
    if langfuse is not None:
        langfuse.flush()
