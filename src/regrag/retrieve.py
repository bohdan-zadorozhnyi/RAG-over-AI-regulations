"""Step 5. Hybrid retrieval.

Two independent rankings fused by reciprocal rank fusion:

  score(d) = sum over rankings r of 1 / (rrf_k + rank_r(d))

Both paths exist because legal queries are bimodal. "What does Article 9
require" is lexical — an embedding blurs the article number. "What are my
obligations around training data" is semantic — full-text misses paraphrase.
Pure vector search fails badly on the first kind.

Do the fusion in SQL, one round trip, not in Python over two result sets.
"""

from regrag.models import Retrieved, SectionKind


def search(
    question: str,
    kinds: list[SectionKind] | None = None,
    strategy: str | None = None,
) -> list[Retrieved]:
    raise NotImplementedError


def vector_search(question: str, k: int) -> list[tuple[int, int]]:
    """Returns (chunk_id, rank)."""
    raise NotImplementedError


def fts_search(question: str, k: int) -> list[tuple[int, int]]:
    """Returns (chunk_id, rank). websearch_to_tsquery over chunks.tsv."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, int]]], k: int, top_n: int
) -> list[tuple[int, float]]:
    raise NotImplementedError
