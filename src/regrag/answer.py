"""Step 7. Answer with citations, or refuse.

A refusal is a correct output. If the retrieved provisions do not support an
answer, say so — most RAG demos get this wrong and it is the first thing a
reviewer tests.
"""

from dataclasses import dataclass

from regrag.models import Retrieved


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]      # ["32024R1689:9", ...]
    refused: bool
    latency_ms: int
    cost_usd: float | None


def answer(question: str) -> Answer:
    raise NotImplementedError


def build_prompt(question: str, chunks: list[Retrieved]) -> str:
    raise NotImplementedError


def extract_citations(text: str, chunks: list[Retrieved]) -> list[str]:
    """Only citations traceable to a retrieved chunk count. Drop the rest."""
    raise NotImplementedError
