"""The spread keep and pick against the single-CTA kernels they replace, slot for slot.

One GPU. A cluster of CTAs shares a row, and the columns it keeps and the children and keys it writes are those one CTA
keeps and writes - the greatest by score, a tie to the lower column, in column order.
"""

import pytest
import torch

from lsa.core.kernels.cutedsl import beam as cutedsl_beam
from lsa.core.kernels.cutedsl import spread

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
NODE_BUCKET_SIZE = 4
POOL = 50000


def scored_row(rows, width, seed, tied, keep, length=None, scores=None):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    drawn = torch.randn(rows, width, device=DEVICE, generator=generator)
    if scores is None:
        scores = drawn
        if tied == "grid":
            # Words repeated a few at a time, which the first two passes leave for the ranking
            scores = torch.round(scores * 2048) / 2048
        elif tied == "coarse":
            # Words repeated by the hundred, which they leave for the last two passes
            scores = torch.round(scores * 64) / 64
        elif tied:
            # A repeated text - half the row at one score, and the rest on a coarse grid
            scores = torch.round(scores * 4) / 4
            scores[:, ::2] = scores[:, :1]
        scores[torch.rand(scores.shape, device=DEVICE, generator=generator) < 0.02] = float("-inf")
    candidates = torch.randint(0, POOL, (rows, width), device=DEVICE, generator=generator).int()
    candidates[torch.rand(candidates.shape, device=DEVICE, generator=generator) < 0.2] = -1
    children = torch.randint(0, 10**6, (POOL, NODE_BUCKET_SIZE), device=DEVICE, generator=generator).int()
    if length is not None:
        return scores, candidates, children, torch.full((rows,), length, device=DEVICE, dtype=torch.int32)
    # Nothing, one column, one under the keep, the keep, one over it, the width, and anything between
    lengths = torch.randint(0, width + 1, (rows,), device=DEVICE, generator=generator).int()
    marks = torch.tensor([0, 1, keep - 1, keep, keep + 1, width], device=DEVICE, dtype=torch.int32)
    lengths[: min(rows, len(marks))] = torch.minimum(marks, torch.tensor(width, device=DEVICE))[:rows]
    return scores, candidates, children, lengths


def words_scores(rows, width, seed, patch):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = torch.randn(rows, width, device=DEVICE, generator=generator)
    return patch(scores)


def bits_to_scores(bits):
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32).to(DEVICE)


def poisoned(scores):
    # A NaN with and without the sign bit among finite scores and infinities
    scores[:, 3::101] = bits_to_scores(0x7FC00000)
    scores[:, 5::103] = bits_to_scores(-0x400000)
    scores[:, 7::107] = float("inf")
    scores[:, 11::109] = float("-inf")
    return scores


def boundary_scores(rows, width, count, keep, identical, seed):
    # `count` columns share the top two bytes of the keep-th greatest score, and the cut falls inside them
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    above = keep - count // 2
    rows_bits = []
    for _ in range(rows):
        high = torch.cat(
            [
                0x3F01 + torch.randint(0, 0x7E, (above,), device=DEVICE, generator=generator),
                torch.full((count,), 0x3F00, device=DEVICE),
                0x3E00
                + torch.randint(0, 0x100, (width - above - count,), device=DEVICE, generator=generator),
            ]
        )
        low = torch.randint(0, 2**16, (width,), device=DEVICE, generator=generator)
        if identical:
            low[above : above + count] = 0x1234
        bits = (high << 16 | low).int()
        rows_bits.append(bits[torch.randperm(width, device=DEVICE, generator=generator)])
    return torch.stack(rows_bits).view(torch.float32)


def keep_expected_and_got(scores, candidates, children, keep, lengths, cluster):
    rows = len(scores)
    index_queries = torch.zeros(rows, 64, device=DEVICE, dtype=torch.bfloat16)
    centroids = torch.zeros(POOL, 64, device=DEVICE, dtype=torch.bfloat16)
    expected = cutedsl_beam.keep_scored(index_queries, scores, candidates, centroids, children, keep, lengths)
    got = spread.keep_scored(scores, candidates, children, keep, lengths, spread.SPREAD_THREADS, cluster)
    return expected, got


def pick_expected_and_got(scores, candidates, lengths, top_p, cluster):
    expected = cutedsl_beam.pick_scored(scores, candidates, lengths, top_p)
    got = spread.pick_scored(scores, candidates, lengths, top_p, spread.SPREAD_THREADS, cluster)
    return expected, got


CLUSTERS = pytest.mark.parametrize("cluster", [4, 8])


@CLUSTERS
@pytest.mark.parametrize(
    ("rows", "width", "keep", "tied", "length"),
    [
        (1, 12288, 2048, False, 12288),
        (1, 12288, 2048, False, 8000),
        (1, 12288, 2048, True, 12288),
        (1, 12288, 2048, False, 2049),
        (16, 12288, 2048, False, None),
        (16, 12288, 2048, True, None),
        (7, 32768, 2048, True, None),
        (7, 12288, 2048, "grid", None),
        (7, 12288, 2048, "coarse", None),
        (7, 3000, 512, False, None),
        (6, 700, 2048, True, None),
        (6, 9, 2048, False, None),
        (1, 600, 599, True, 600),
    ],
    ids=[
        "B=1, full",
        "B=1, partial",
        "B=1, full, ties",
        "B=1, one over the keep",
        "B=16",
        "B=16, ties",
        "start level, ties",
        "few ties",
        "many ties",
        "narrow",
        "narrower than the keep",
        "tiny",
        "keep-1",
    ],
)
def test_the_spread_keep_keeps_the_children_one_cta_keeps(cluster, rows, width, keep, tied, length):
    scores, candidates, children, lengths = scored_row(rows, width, width + rows, tied, keep, length)
    expected, got = keep_expected_and_got(scores, candidates, children, keep, lengths, cluster)
    assert torch.equal(got, expected)


@CLUSTERS
@pytest.mark.parametrize(
    ("rows", "width", "top_p", "tied", "length"),
    [
        (1, 12288, 1024, False, 12288),
        (1, 12288, 1024, False, 8000),
        (1, 12288, 1024, True, 12288),
        (16, 12288, 1024, False, None),
        (16, 12288, 1024, True, None),
        (6, 12288, 1024, "grid", None),
        (6, 12288, 1024, "coarse", None),
        (6, 700, 1024, False, None),
        (6, 3000, 1024, True, None),
        (6, 5, 1024, False, None),
    ],
    ids=[
        "B=1, full",
        "B=1, partial",
        "B=1, full, ties",
        "B=16",
        "B=16, ties",
        "few ties",
        "many ties",
        "narrower than the top-p",
        "whole row ties",
        "tiny",
    ],
)
def test_the_spread_pick_names_the_buckets_one_cta_names(cluster, rows, width, top_p, tied, length):
    scores, candidates, _, lengths = scored_row(rows, width, width + rows, tied, top_p, length)
    expected, got = pick_expected_and_got(scores, candidates, lengths, top_p, cluster)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


@pytest.mark.parametrize("rows", [1, spread.FEW_ROWS, spread.FEW_ROWS + 1, 16])
def test_the_layout_the_descent_ships_keeps_and_picks_what_one_cta_does(rows):
    width, keep = 12288, 2048
    layout = spread.layout(rows, width, keep)
    scores, candidates, children, lengths = scored_row(rows, width, rows, True, keep, width)
    index_queries = torch.zeros(rows, 64, device=DEVICE, dtype=torch.bfloat16)
    centroids = torch.zeros(POOL, 64, device=DEVICE, dtype=torch.bfloat16)
    expected = cutedsl_beam.keep_scored(index_queries, scores, candidates, centroids, children, keep, lengths)
    assert torch.equal(spread.keep_scored(scores, candidates, children, keep, lengths, *layout), expected)
    expected_keys, expected_listed = cutedsl_beam.pick_scored(scores, candidates, lengths, keep // 2)
    keys, listed = spread.pick_scored(scores, candidates, lengths, keep // 2, *layout)
    assert torch.equal(keys, expected_keys)
    assert torch.equal(listed, expected_listed)


SPECIAL_SCORES = pytest.mark.parametrize(
    "special",
    [
        lambda scores: poisoned(scores),
        lambda scores: scores.fill_(float("-inf")),
        lambda scores: scores.fill_(0.5),
    ],
    ids=["NaN, -NaN and infinities", "all -inf", "all equal"],
)


@CLUSTERS
@SPECIAL_SCORES
@pytest.mark.parametrize("rows", [1, 16])
def test_the_spread_keep_keeps_what_one_cta_keeps_of_scores_that_are_not_numbers_or_all_alike(
    cluster, special, rows
):
    width, keep = 12288, 2048
    scores = words_scores(rows, width, rows, special)
    scores, candidates, children, lengths = scored_row(rows, width, rows, False, keep, width, scores)
    expected, got = keep_expected_and_got(scores, candidates, children, keep, lengths, cluster)
    assert torch.equal(got, expected)


@CLUSTERS
@SPECIAL_SCORES
@pytest.mark.parametrize("rows", [1, 16])
def test_the_spread_pick_names_what_one_cta_names_of_scores_that_are_not_numbers_or_all_alike(
    cluster, special, rows
):
    width, top_p = 12288, 1024
    scores = words_scores(rows, width, rows, special)
    scores, candidates, _, lengths = scored_row(rows, width, rows, False, top_p, width, scores)
    expected, got = pick_expected_and_got(scores, candidates, lengths, top_p, cluster)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


@CLUSTERS
@pytest.mark.parametrize("identical", [False, True], ids=["distinct", "identical"])
@pytest.mark.parametrize("count", [255, 256, 257])
def test_the_spread_keep_and_pick_rank_the_words_the_first_two_passes_leave_as_one_cta_does(
    cluster, count, identical
):
    width, keep, rows = 12288, 2048, 2
    scores = boundary_scores(rows, width, count, keep, identical, count)
    scores, candidates, children, lengths = scored_row(rows, width, count, False, keep, width, scores)
    expected, got = keep_expected_and_got(scores, candidates, children, keep, lengths, cluster)
    assert torch.equal(got, expected)
    top_p = keep // 2
    scores = boundary_scores(rows, width, count, top_p, identical, count)
    scores, candidates, _, lengths = scored_row(rows, width, count, False, top_p, width, scores)
    expected, got = pick_expected_and_got(scores, candidates, lengths, top_p, cluster)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_an_empty_batch_keeps_nothing():
    scores = torch.empty(0, 600, device=DEVICE)
    candidates = torch.empty(0, 600, device=DEVICE, dtype=torch.int32)
    children = torch.zeros(10, NODE_BUCKET_SIZE, device=DEVICE, dtype=torch.int32)
    lengths = torch.empty(0, device=DEVICE, dtype=torch.int32)
    assert spread.keep_scored(scores, candidates, children, 100, lengths).shape == (0, 100 * NODE_BUCKET_SIZE)


def test_a_row_whose_slice_does_not_fit_is_refused():
    width = spread.CLUSTER * cutedsl_beam.SHARED_WORDS
    scores = torch.empty(1, width, device=DEVICE)
    candidates = torch.empty(1, width, device=DEVICE, dtype=torch.int32)
    children = torch.zeros(10, NODE_BUCKET_SIZE, device=DEVICE, dtype=torch.int32)
    with pytest.raises(ValueError, match="shared"):
        spread.keep_scored(
            scores, candidates, children, 2048, torch.zeros(1, device=DEVICE, dtype=torch.int32)
        )


def test_a_narrow_row_stays_in_one_cta_and_a_wide_batch_takes_smaller_clusters():
    assert spread.layout(1, spread.SPREAD_FROM - 1, 2048) is None
    assert spread.layout(1, 12288, 2048) == (spread.SPREAD_THREADS, spread.CLUSTER)
    assert spread.layout(spread.FEW_ROWS + 1, 12288, 2048) == (spread.SPREAD_THREADS, spread.CLUSTER // 2)
    assert spread.layout(1, spread.CLUSTER * cutedsl_beam.SHARED_WORDS, 2048) is None
