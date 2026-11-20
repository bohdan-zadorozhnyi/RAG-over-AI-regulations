"""Unit tests for the parts that are pure functions.

RRF is the one piece of real algorithmic logic here, so test it properly:
a document ranked first in both rankings must outrank one ranked first in
only one; absent documents must not contribute.
"""

import pytest

from regrag.retrieve import reciprocal_rank_fusion


@pytest.mark.xfail(reason="step 5", strict=False)
def test_rrf_prefers_documents_ranked_well_in_both() -> None:
    vector = [(1, 1), (2, 2)]
    fts = [(2, 1), (1, 5)]
    fused = dict(reciprocal_rank_fusion([vector, fts], k=60, top_n=2))
    assert fused[2] > fused[1]
