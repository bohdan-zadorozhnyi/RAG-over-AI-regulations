"""Unit tests for the parts of retrieval that are pure functions.

RRF is the one piece of real algorithmic logic here, so test it properly:
a document ranked first in both rankings must outrank one ranked first in
only one; absent documents must not contribute.
"""

import pytest

from regrag.models import Retrieved, SectionKind
from regrag.retrieve import reciprocal_rank_fusion, with_coverage


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


# --- regulation coverage (pure) ---------------------------------------------

def chunk(chunk_id: int, celex: str) -> Retrieved:
    return Retrieved(
        chunk_id=chunk_id,
        celex=celex,
        article_no="1",
        header="",
        text="",
        vector_rank=chunk_id,
        fts_rank=None,
        reference_rank=None,
        pinned=False,
        coverage=False,
        fused_score=1.0 / chunk_id,
        similarity=0.5,
        strategy="section_aware",
        kind=SectionKind.ARTICLE,
        section_path="p",
        citations=(f"{celex}:1",),
    )


def test_coverage_fills_the_tail_with_missing_regulations() -> None:
    fused = [chunk(i, "A") for i in range(1, 9)]  # one regulation fills all 8
    best = [chunk(20, "B"), chunk(30, "C"), chunk(1, "A")]
    out = with_coverage(fused, best, top_n=8, regulations=3)
    assert [h.chunk_id for h in out] == [1, 2, 3, 4, 5, 6, 20, 30]
    assert [h.coverage for h in out] == [False] * 6 + [True, True]


def test_coverage_leaves_the_head_and_a_covered_tail_alone() -> None:
    fused = [chunk(1, "A"), chunk(2, "B"), chunk(3, "C"), chunk(4, "A"), chunk(5, "A")]
    best = [chunk(2, "B"), chunk(3, "C")]
    out = with_coverage(fused, best, top_n=5, regulations=3)
    assert out == fused  # every regulation already in the head


def test_coverage_reuses_the_fused_hit_when_the_chunk_was_already_ranked() -> None:
    fused = [chunk(1, "A"), chunk(2, "A"), chunk(3, "B")]
    best = [chunk(3, "B")]
    out = with_coverage(fused, best, top_n=3, regulations=2)
    assert [h.chunk_id for h in out] == [1, 2, 3]
    assert out[2].coverage and out[2].vector_rank == 3  # fused ranks kept


def test_coverage_with_a_short_fused_list() -> None:
    out = with_coverage([chunk(1, "A")], [chunk(9, "B")], top_n=8, regulations=2)
    assert [h.chunk_id for h in out] == [1, 9]
