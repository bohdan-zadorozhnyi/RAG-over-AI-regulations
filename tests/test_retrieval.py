"""Unit tests for the parts of retrieval that are pure functions.

RRF is the one piece of real algorithmic logic here, so test it properly:
a document ranked first in both rankings must outrank one ranked first in
only one; absent documents must not contribute.
"""

import pytest

from regrag.retrieve import reciprocal_rank_fusion


def test_rrf_prefers_documents_ranked_well_in_both() -> None:
    vector = [(1, 1), (2, 2)]
    fts = [(2, 1), (1, 5)]
    fused = dict(reciprocal_rank_fusion([vector, fts], k=60, top_n=2))
    assert fused[2] > fused[1]


def test_rrf_first_in_both_beats_first_in_one() -> None:
    vector = [(1, 1), (2, 2)]
    fts = [(1, 1), (3, 2)]
    order = [chunk for chunk, _ in reciprocal_rank_fusion([vector, fts], k=60, top_n=3)]
    assert order[0] == 1


def test_rrf_scores_are_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[(7, 1)], [(7, 3)]], k=60, top_n=1))
    assert fused[7] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_absent_documents_contribute_nothing() -> None:
    fused = dict(reciprocal_rank_fusion([[(1, 1)], [(2, 1)]], k=60, top_n=2))
    assert fused == {1: pytest.approx(1 / 61), 2: pytest.approx(1 / 61)}


def test_rrf_respects_top_n() -> None:
    ranking = [(n, n) for n in range(1, 11)]
    assert len(reciprocal_rank_fusion([ranking], k=60, top_n=3)) == 3


def test_rrf_ties_break_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([[(9, 1)], [(4, 1)]], k=60, top_n=2)
    assert [chunk for chunk, _ in fused] == [4, 9]


def test_rrf_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []], k=60, top_n=8) == []


def test_rrf_smaller_k_rewards_top_ranks_more() -> None:
    # rank 1 vs rank 10 in a single ranking: the gap shrinks as k grows
    def gap(k: int) -> float:
        fused = dict(reciprocal_rank_fusion([[(1, 1), (2, 10)]], k=k, top_n=2))
        return fused[1] / fused[2]

    assert gap(0) > gap(60)


@pytest.mark.parametrize(("k", "rank"), [(-1, 1), (60, 0)])
def test_rrf_rejects_invalid_k_or_rank(k: int, rank: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[(1, rank)]], k=k, top_n=1)
