"""The CuTeDSL short-prefill listing against the Triton top-k it replaces.

One Hopper GPU. A listing need not match the Triton one key for key - the fill finds its own stopping key in any
listing that holds a row's best top-p - so each row's best top-p are compared, and a listing must repeat exactly.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.cutedsl import top_k as cutedsl_top_k
from lsa.core.kernels.triton.top_k import TopK

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
TOP_P = 1024


def scores_of(queries, buckets, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    index_queries = F.normalize(
        torch.randn(queries, 64, device=DEVICE, generator=generator), dim=-1
    ).bfloat16()
    centroids = F.normalize(torch.randn(buckets, 64, device=DEVICE, generator=generator), dim=-1).bfloat16()
    scores = torch.mm(index_queries, centroids.t(), out_dtype=torch.float32)
    lengths = torch.randint(
        buckets // 2, buckets + 1, (queries,), device=DEVICE, dtype=torch.int32, generator=generator
    )
    return scores, lengths


def best(keys, listed, want):
    return [keys[r, : listed[r]].sort(descending=True).values[:want].tolist() for r in range(keys.shape[0])]


def assert_holds_the_triton_best(scores, lengths):
    want_keys, want_listed = TopK(TOP_P).listed(scores, lengths)
    keys, listed = cutedsl_top_k.list_rows(scores, lengths, TOP_P)
    assert bool((listed <= keys.shape[1]).all())
    assert best(keys, listed, TOP_P) == best(want_keys, want_listed, TOP_P)
    ordered, ordered_listed = cutedsl_top_k.list_rows(scores, lengths, TOP_P, ordered=True)
    assert best(ordered, ordered_listed, TOP_P) == best(want_keys, want_listed, TOP_P)
    for r in range(keys.shape[0]):
        columns = 0x7FFFFFFF - (ordered[r, : ordered_listed[r]] & 0xFFFFFFFF)
        assert bool((columns[1:] > columns[:-1]).all())


@pytest.mark.parametrize(
    ("queries", "buckets"),
    [(300, 3000), (256, 4096), (512, 9000), (128, 32767)],
    ids=["narrower than the top-p's listing", "whole rows", "sampled", "widest short prefill"],
)
def test_each_row_holds_the_triton_best_top_p(queries, buckets):
    assert_holds_the_triton_best(*scores_of(queries, buckets))


def test_a_listing_repeats_exactly():
    scores, lengths = scores_of(256, 16384, seed=1)
    keys, listed = cutedsl_top_k.list_rows(scores, lengths, TOP_P)
    again, again_listed = cutedsl_top_k.list_rows(scores, lengths, TOP_P)
    assert torch.equal(listed, again_listed)
    assert all(torch.equal(keys[r, : listed[r]], again[r, : listed[r]]) for r in range(256))


def test_a_row_the_sample_misjudges_is_listed_exactly():
    # A few hundred sampled columns far above the rest list fewer than the top-p, every sampled column more than room
    scores, _ = scores_of(64, 32767, seed=2)
    stride, _, _ = cutedsl_top_k.plan(TOP_P, 32767)
    scores[:32, : 300 * stride : stride] = 10.0
    scores[32:, ::stride] = 10.0
    lengths = torch.full((64,), 32767, device=DEVICE, dtype=torch.int32)
    _, listed = cutedsl_top_k.list_rows(scores, lengths, TOP_P)
    assert listed.tolist() == [TOP_P] * 64
    assert_holds_the_triton_best(scores, lengths)


def test_rows_of_strided_scores_are_refused():
    scores, lengths = scores_of(8, 4096)
    with pytest.raises(ValueError, match="unit column stride"):
        cutedsl_top_k.list_rows(scores.t(), lengths, TOP_P)
