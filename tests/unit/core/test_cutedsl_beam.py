"""The CuTeDSL tree levels against the Triton chains they replace - child scores, the exact keep, the kept children and
the picked buckets' keys.

One GPU. Both score each child as the same fp32 sum in the same order, so they keep the same columns and name the
same children and cache rows, bit for bit and slot for slot.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.cutedsl import beam as cutedsl_beam
from lsa.core.kernels.triton import beam
from lsa.core.kernels.triton.beam import NODE_CENTROID_SCALE
from lsa.core.kernels.triton.keys import EMPTY
from lsa.core.kernels.triton.shape import score_shaped_rows
from lsa.core.kernels.triton.top_k import TopK
from lsa.core.shape import pack_shape_basis

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
NODE_BUCKET_SIZE = 4
PAGE_ROWS = 64
# The widest row a 2048-wide keep leaves room for
WIDEST = cutedsl_beam.SHARED_WORDS - cutedsl_beam.room(cutedsl_beam.SHARED_WORDS, 2048)[1]


def level(rows, open_count, carried_width, node_buckets, dtype, seed, tied=False, holes=0.05):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    scale = NODE_CENTROID_SCALE if dtype == torch.float8_e4m3fn else 1.0
    index_queries = F.normalize(torch.randn(rows, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    centroids = F.normalize(torch.randn(node_buckets, 64, device=DEVICE, generator=generator), dim=1)
    if tied:
        # A repeated text - half the node buckets share one centroid
        centroids[: node_buckets // 2] = centroids[0]
    centroids = (centroids * scale).to(dtype)
    open_centroids = F.normalize(
        torch.randn(open_count, 64, device=DEVICE, generator=generator), dim=1
    ).bfloat16()
    open_scores = torch.mm(index_queries, open_centroids.t(), out_dtype=torch.float32)
    open_children = torch.randint(
        0, 10**6, (open_count, NODE_BUCKET_SIZE), device=DEVICE, generator=generator
    )
    closed_children = torch.randint(
        0, 10**6, (node_buckets, NODE_BUCKET_SIZE), device=DEVICE, generator=generator
    )
    carried = torch.randint(0, node_buckets, (rows, carried_width), device=DEVICE, generator=generator)
    carried[torch.rand(carried.shape, device=DEVICE, generator=generator) < holes] = -1
    return (
        index_queries,
        open_scores,
        open_children.int(),
        carried.int(),
        centroids,
        closed_children.int(),
        scale,
    )


def triton_level(index_queries, open_scores, open_children, carried, centroids, closed_children, scale, keep):
    rows, open_count = open_scores.shape
    width = open_count + carried.shape[1]
    scores = torch.empty(rows, width, device=DEVICE)
    scores[:, :open_count] = open_scores
    beam.score_children(index_queries, centroids, carried, out=scores[:, open_count:], centroid_scale=scale)
    kept = TopK(keep)(scores, torch.full((rows,), width, device=DEVICE, dtype=torch.int32))
    return beam.kept_children(
        scores, kept, open_count, open_children, carried, closed_children, NODE_BUCKET_SIZE
    )


@pytest.mark.parametrize(
    ("rows", "open_count", "carried_width", "keep", "dtype", "tied", "holes"),
    [
        (64, 300, 2048, 512, torch.float8_e4m3fn, False, 0.05),
        (64, 300, 2048, 512, torch.float8_e4m3fn, True, 0.05),
        (64, 0, 1024, 2048, torch.float8_e4m3fn, False, 0.6),
        (64, 700, 8192, 2048, torch.bfloat16, False, 0.05),
        (300, 900, 8192, 2048, torch.float8_e4m3fn, False, 0.05),
        (17, 5, 40, 64, torch.float8_e4m3fn, True, 0.3),
        (16, 0, WIDEST, 2048, torch.float8_e4m3fn, False, 0.05),
    ],
    ids=[
        "fp8",
        "tied",
        "open none and holes",
        "bf16",
        "served shape",
        "narrower than the beam",
        "widest row",
    ],
)
def test_a_level_keeps_the_children_the_triton_chain_keeps(
    rows, open_count, carried_width, keep, dtype, tied, holes
):
    inputs = level(
        rows, open_count, carried_width, 200000, dtype, seed=rows + open_count, tied=tied, holes=holes
    )
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
    expected = triton_level(*inputs, keep)
    kept = cutedsl_beam.beam_step(
        index_queries, open_scores, open_children, carried, centroids, closed_children, keep, scale
    )
    assert torch.equal(kept, expected)


def shapes_of(node_buckets, rank, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    signs = torch.randn(node_buckets, rank, 64, device=DEVICE, generator=generator).sign()
    variances = torch.rand(node_buckets, rank, device=DEVICE, generator=generator) * 0.02
    floor = torch.rand(node_buckets, device=DEVICE, generator=generator) * 0.002
    return pack_shape_basis(signs), variances, floor, 2.0


@pytest.mark.parametrize(
    ("rows", "open_count", "carried_width", "keep", "dtype"),
    [
        (64, 300, 2048, 512, torch.float8_e4m3fn),
        (64, 700, 8192, 2048, torch.bfloat16),
        (300, 900, 8192, 2048, torch.float8_e4m3fn),
        (17, 5, 40, 64, torch.float8_e4m3fn),
    ],
    ids=["fp8", "bf16", "served shape", "narrower than the beam"],
)
def test_a_shaped_level_keeps_the_children_the_triton_chain_keeps(
    rows, open_count, carried_width, keep, dtype
):
    inputs = level(rows, open_count, carried_width, 200000, dtype, seed=rows + keep)
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
    shapes = shapes_of(200000, cutedsl_beam.SHAPE_RANK, seed=rows)
    width = open_count + carried_width
    scores = torch.empty(rows, width, device=DEVICE)
    scores[:, :open_count] = open_scores
    score_shaped_rows(
        index_queries,
        centroids,
        *shapes[:3],
        carried,
        shapes[3],
        out=scores[:, open_count:],
        centroid_scale=scale,
    )
    kept = TopK(keep)(scores, torch.full((rows,), width, device=DEVICE, dtype=torch.int32))
    expected = beam.kept_children(
        scores, kept, open_count, open_children, carried, closed_children, NODE_BUCKET_SIZE
    )
    got = cutedsl_beam.beam_step(
        index_queries, open_scores, open_children, carried, centroids, closed_children, keep, scale, shapes
    )
    assert torch.equal(got, expected)


def test_shapes_of_another_rank_are_refused():
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = level(
        4, 8, 32, 100, torch.float8_e4m3fn, seed=1
    )
    shapes = shapes_of(100, 1, seed=1)
    assert not cutedsl_beam.supports(index_queries, centroids, carried, open_scores.shape[1], 16, 1)
    with pytest.raises(ValueError, match="shapes of 2 directions"):
        cutedsl_beam.beam_step(
            index_queries, open_scores, open_children, carried, centroids, closed_children, 16, scale, shapes
        )


def test_rows_the_kernel_does_not_take_are_refused():
    inputs = level(4, 8, 32, 100, torch.float8_e4m3fn, seed=1)
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
    assert not cutedsl_beam.supports(index_queries.float(), centroids, carried, open_scores.shape[1], 16)
    with pytest.raises(ValueError, match="bf16 index queries of 64 dims"):
        cutedsl_beam.beam_step(
            index_queries.float(), open_scores, open_children, carried, centroids, closed_children, 16, scale
        )


def test_a_row_wider_than_shared_memory_is_refused():
    inputs = level(4, 1, cutedsl_beam.SHARED_WORDS, 100, torch.float8_e4m3fn, seed=1)
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
    assert not cutedsl_beam.supports(index_queries, centroids, carried, open_scores.shape[1], 16)
    with pytest.raises(ValueError, match="within"):
        cutedsl_beam.beam_step(
            index_queries, open_scores, open_children, carried, centroids, closed_children, 16, scale
        )


@pytest.mark.parametrize(
    ("open_count", "carried_width", "keep", "dtype"),
    [(0, 1024, 64, torch.float8_e4m3fn), (7, 9000, 64, torch.float8_e4m3fn), (0, 8192, 2048, torch.bfloat16)],
    ids=["whole rows", "sampled and whole rows", "served shape"],
)
def test_a_query_scores_and_keeps_only_its_first_columns_when_given_lengths(
    open_count, carried_width, keep, dtype
):
    rows = 24
    inputs = level(rows, open_count, carried_width, 50000, dtype, seed=carried_width)
    index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    lengths = torch.randint(0, carried_width + 1, (rows,), device=DEVICE, generator=generator).int()
    # Nothing, one column, the keep, and every column
    lengths[:4] = torch.tensor([0, 1, keep, carried_width], device=DEVICE, dtype=torch.int32)
    columns = torch.arange(carried_width, device=DEVICE)[None]
    # A column past its query's length is no column, which the Triton chain reads as a -1 row
    expected = triton_level(
        index_queries,
        open_scores,
        open_children,
        torch.where(columns < lengths[:, None], carried, -1),
        centroids,
        closed_children,
        scale,
        keep,
    )
    kept = cutedsl_beam.beam_step(
        index_queries,
        open_scores,
        open_children,
        carried,
        centroids,
        closed_children,
        keep,
        scale,
        lengths=lengths,
    )
    assert torch.equal(kept, expected)


def test_one_compile_serves_every_row_width():
    cutedsl_beam.COMPILED.clear()
    for carried_width in (40, 3000, 9000):
        inputs = level(8, 7, carried_width, 1000, torch.float8_e4m3fn, seed=carried_width)
        index_queries, open_scores, open_children, carried, centroids, closed_children, scale = inputs
        kept = cutedsl_beam.beam_step(
            index_queries, open_scores, open_children, carried, centroids, closed_children, 64, scale
        )
        assert torch.equal(kept, triton_level(*inputs, 64))
    assert len(cutedsl_beam.COMPILED) == 1


@pytest.mark.parametrize(
    ("rows", "width", "keep", "dtype", "tied"),
    [
        (6, 32768, 2048, torch.float8_e4m3fn, False),
        (6, 32768, 2048, torch.float8_e4m3fn, True),
        (5, 20000, 2048, torch.bfloat16, False),
        (6, 6000, 512, torch.float8_e4m3fn, True),
    ],
    ids=["start level", "start level, ties", "bf16", "ties under the sample"],
)
def test_a_scored_row_keeps_the_children_the_triton_chain_keeps(rows, width, keep, dtype, tied):
    inputs = level(rows, 0, width, 50000, dtype, seed=width, tied=tied, holes=0.3)
    index_queries, _, _, candidates, centroids, closed_children, scale = inputs
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    # Nothing, one column, one under the keep, the keep, one over it, and anything between
    lengths = torch.randint(0, width + 1, (rows,), device=DEVICE, generator=generator).int()
    lengths[: min(rows, 5)] = torch.tensor(
        [0, 1, keep - 1, keep, keep + 1], device=DEVICE, dtype=torch.int32
    )[: min(rows, 5)]
    scores = beam.score_children(index_queries, centroids, candidates, centroid_scale=scale, lengths=lengths)
    kept = TopK(keep)(scores, lengths)
    expected = beam.kept_children(
        scores, kept, 0, closed_children, candidates, closed_children, NODE_BUCKET_SIZE
    )
    got = cutedsl_beam.keep_scored(
        index_queries, scores, candidates, centroids, closed_children, keep, lengths
    )
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    ("rows", "width", "top_p", "tied"),
    [(6, 12288, 1024, False), (6, 12288, 1024, True), (6, 700, 1024, False), (6, 3000, 1024, True)],
    ids=["served shape", "ties", "narrower than the top-p", "whole row ties"],
)
def test_a_scored_pick_names_the_buckets_the_triton_pick_names(rows, width, top_p, tied):
    generator = torch.Generator(device=DEVICE).manual_seed(width)
    index_queries = F.normalize(torch.randn(rows, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    centroids = F.normalize(torch.randn(70000, 64, device=DEVICE, generator=generator), dim=1)
    if tied:
        centroids[::2] = centroids[0]
    centroids = centroids.bfloat16()
    candidates = torch.randint(0, 70000, (rows, width), device=DEVICE, generator=generator).int()
    candidates[torch.rand(candidates.shape, device=DEVICE, generator=generator) < 0.2] = -1
    lengths = torch.randint(0, width + 1, (rows,), device=DEVICE, generator=generator).int()
    marks = torch.tensor([0, 1, top_p - 1, top_p, top_p + 1, width], device=DEVICE, dtype=torch.int32)
    lengths[:] = torch.minimum(torch.cat([marks, lengths])[:rows], torch.tensor(width, device=DEVICE))
    scores = beam.score_children(index_queries, centroids, candidates, lengths=lengths)
    keys, listed = TopK(top_p, exact=True).listed(scores, lengths)
    expected = beam.cache_row_keys(keys, listed, candidates)
    keys, listed = cutedsl_beam.pick_scored(scores, candidates, lengths, top_p)
    assert torch.equal(keys, expected)
    assert listed.tolist() == [min(width, top_p)] * rows


def top_scores(rows, width, seed, tied=False, lost=0.02):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = torch.randn(rows, width, device=DEVICE, generator=generator)
    if tied:
        # A repeated text - a third of every row at one score, which lists past any room
        scores[:, ::3] = scores[:, :1]
    scores[torch.rand(scores.shape, device=DEVICE, generator=generator) < lost] = float("-inf")
    children = torch.randint(0, 10**6, (width, NODE_BUCKET_SIZE), device=DEVICE, generator=generator).int()
    index_queries = torch.zeros(rows, 64, device=DEVICE, dtype=torch.bfloat16)
    return index_queries, scores, children


@pytest.mark.parametrize(
    ("rows", "width", "keep", "tied"),
    [
        (64, 19456, 2048, False),
        (64, 19456, 2048, True),
        (16, 32768, 2048, False),
        (16, 32768, 2048, True),
        (64, 6000, 2048, False),
        (9, 300, 512, False),
    ],
    ids=["sampled", "sampled ties", "listed first", "listed first, ties", "whole", "narrower than the beam"],
)
def test_a_start_level_keeps_the_children_the_triton_keep_keeps(rows, width, keep, tied):
    index_queries, scores, children = top_scores(rows, width, seed=width, tied=tied)
    kept = TopK(keep)(scores, torch.full((rows,), width, device=DEVICE, dtype=torch.int32))
    expected = beam.kept_children(scores, kept, width, children, None, None, NODE_BUCKET_SIZE)
    assert torch.equal(cutedsl_beam.keep_start_level(index_queries, scores, children, keep), expected)


def test_a_row_its_sample_overrates_is_searched_whole():
    # Only the sampled columns score high, so the sample's threshold lists fewer than the keep above it
    rows, width, keep = 4, 16384, 1024
    index_queries, scores, children = top_scores(rows, width, seed=3, lost=0)
    scores.fill_(-1)
    sampled = torch.arange(cutedsl_beam.SAMPLES, device=DEVICE) * width // cutedsl_beam.SAMPLES
    scores[:, sampled] = torch.linspace(1, 0.5, len(sampled), device=DEVICE)
    scores[:, 1::2] += torch.rand(rows, width // 2, device=DEVICE) * 1e-3
    kept = TopK(keep)(scores, torch.full((rows,), width, device=DEVICE, dtype=torch.int32))
    expected = beam.kept_children(scores, kept, width, children, None, None, NODE_BUCKET_SIZE)
    assert torch.equal(cutedsl_beam.keep_start_level(index_queries, scores, children, keep), expected)


def last_level(rows, open_count, carried_width, bucket_rows, seed, tied=False, holes=0.05):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    index_queries = F.normalize(torch.randn(rows, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    bucket_centroids = F.normalize(torch.randn(bucket_rows, 64, device=DEVICE, generator=generator), dim=1)
    if tied:
        bucket_centroids[: bucket_rows // 2] = bucket_centroids[0]
    bucket_centroids = bucket_centroids.bfloat16()
    own = torch.mm(index_queries, bucket_centroids[:open_count].t(), out_dtype=torch.float32)
    # A query sees fewer of the uncovered buckets than the last
    seen = torch.arange(open_count, device=DEVICE)[None] < torch.randint(
        open_count // 2, open_count + 1, (rows, 1), device=DEVICE, generator=generator
    )
    own.masked_fill_(~seen, float("-inf"))
    carried = torch.randint(0, bucket_rows, (rows, carried_width), device=DEVICE, generator=generator).int()
    carried[torch.rand(carried.shape, device=DEVICE, generator=generator) < holes] = -1
    pages = torch.randperm(bucket_rows // PAGE_ROWS, device=DEVICE, generator=generator).int()
    pages[3] = -1
    return index_queries, own, carried, bucket_centroids, pages


def triton_pick(index_queries, own, carried, bucket_centroids, pages, covered, top_p):
    rows, open_count = own.shape
    width = open_count + carried.shape[1]
    scores = torch.empty(rows, width, device=DEVICE)
    scores[:, :open_count] = own
    beam.score_children(index_queries, bucket_centroids, carried, out=scores[:, open_count:])
    pick = TopK(top_p, exact=True)
    keys, listed = pick.listed(scores, torch.full((rows,), width, device=DEVICE, dtype=torch.int32))
    return beam.bucket_keys(keys, listed, carried, pages, open_count, covered, PAGE_ROWS)


@pytest.mark.parametrize(
    ("rows", "open_count", "carried_width", "top_p", "tied"),
    [
        (64, 16, 8192, 1024, False),
        (64, 300, 8192, 1024, True),
        (64, 900, 2048, 1024, False),
        (64, 2000, 0, 1024, False),
        (64, 0, 4096, 1024, False),
        (7, 40, 100, 1024, False),
    ],
    ids=["served shape", "sampled ties", "whole", "no beam", "no uncovered", "narrower than the top-p"],
)
def test_the_pick_names_the_buckets_the_triton_pick_names(rows, open_count, carried_width, top_p, tied):
    index_queries, own, carried, bucket_centroids, pages = last_level(
        rows, open_count, carried_width, 65536, seed=open_count + carried_width, tied=tied
    )
    covered = 640
    expected = triton_pick(index_queries, own, carried, bucket_centroids, pages, covered, top_p)
    picked = cutedsl_beam.pick_top_p(
        index_queries,
        own,
        carried if carried_width else None,
        bucket_centroids,
        pages,
        covered,
        PAGE_ROWS,
        top_p,
    )
    assert torch.equal(picked, expected)
    assert bool((picked == EMPTY).any()) == bool((expected == EMPTY).any())


PAGE_SIZE = 8
LEVEL = 2


def whole_level(dtype, seed=9):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    node_buckets = 40
    pages = torch.stack([torch.randperm(5, device=DEVICE, generator=generator) for _ in range(5)]).int()
    pages[3, 3:] = -1
    made = torch.tensor([30, 9, 24, 20, 0], device=DEVICE, dtype=torch.int32)
    closed = torch.tensor([10, 4, 8, 7, 0], device=DEVICE, dtype=torch.int32)
    start = torch.tensor([1, 2, 3, 3, 0], device=DEVICE, dtype=torch.int32)
    rows = torch.tensor([2, 0, 1, 3, 4], device=DEVICE)
    carried = torch.randint(-1, 30, (5, 200), device=DEVICE, generator=generator).int()
    carried[3] = -1
    scale = NODE_CENTROID_SCALE if dtype == torch.float8_e4m3fn else 1.0
    centroids = (
        F.normalize(torch.randn(node_buckets, 64, device=DEVICE, generator=generator), dim=1) * scale
    ).to(dtype)
    children = torch.randint(
        0, 10**6, (node_buckets, NODE_BUCKET_SIZE), device=DEVICE, generator=generator
    ).int()
    index_queries = F.normalize(torch.randn(5, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    arguments = (pages, made, closed, start, rows, carried)
    return arguments, index_queries, centroids, children, scale


def whole_level_children(arguments, index_queries, centroids, children, width, scale):
    return beam.level_children(*arguments, index_queries, centroids, children, LEVEL, PAGE_SIZE, width, scale)


def scored_keep_children(arguments, index_queries, centroids, children, width, scale):
    candidates, scores, lengths = beam.level_scores(
        *arguments, index_queries, centroids, LEVEL, PAGE_SIZE, width, scale
    )
    return scores, cutedsl_beam.keep_scored(
        index_queries, scores, candidates, centroids, children, width, lengths
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 64, 300])
def test_a_level_that_keeps_every_column_writes_the_children_the_scored_keep_writes(dtype, width):
    arguments, index_queries, centroids, children, scale = whole_level(dtype)
    _, expected = scored_keep_children(arguments, index_queries, centroids, children, width, scale)
    got = whole_level_children(arguments, index_queries, centroids, children, width, scale)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("width", [64, 300])
def test_a_level_names_no_children_under_a_score_the_scored_keep_names_none_under(width):
    arguments, index_queries, centroids, children, scale = whole_level(torch.bfloat16)
    centroids = centroids.abs() + 0.01
    # A query whose first dim is a NaN or an infinity scores every column of a centroid at it there
    poison = torch.tensor([0x7FC0, 0x7FC0, 0xFF80 - 0x10000, 0x7F80], device=DEVICE, dtype=torch.int16)
    index_queries = index_queries.abs()
    index_queries[:4, 0] = poison.view(torch.bfloat16)
    scores, expected = scored_keep_children(arguments, index_queries, centroids, children, width, scale)
    got = whole_level_children(arguments, index_queries, centroids, children, width, scale)
    assert torch.equal(got, expected)
    classes = {
        "+NaN": scores.isnan(),
        "-inf": scores == float("-inf"),
        "+inf": scores == float("inf"),
    }
    # Every class is scored, or the agreement would be blind
    assert all(bool(column.any()) for column in classes.values())
    # +NaN and +inf name their children, -inf none
    named = (expected.view(len(expected), width, NODE_BUCKET_SIZE) >= 0).any(dim=2)
    assert named[classes["+NaN"] | classes["+inf"]].all()
    assert not named[classes["-inf"]].any()


def test_the_scored_keep_names_no_children_under_a_nan_with_the_sign_bit():
    # The GPU's arithmetic gives only the NaN without the sign bit, so a scored row is made to hold the other
    bits = torch.tensor([0x7FC00000, -0x400000, -0x800000, 0x7F800000, 0x3F800000], dtype=torch.int32)
    scores = bits.view(torch.float32).to(DEVICE)[None]
    candidates = torch.arange(5, device=DEVICE, dtype=torch.int32)[None]
    children = torch.arange(5 * NODE_BUCKET_SIZE, device=DEVICE, dtype=torch.int32).view(5, NODE_BUCKET_SIZE)
    index_queries = torch.zeros(1, 64, device=DEVICE, dtype=torch.bfloat16)
    centroids = torch.zeros(5, 64, device=DEVICE, dtype=torch.bfloat16)
    lengths = torch.tensor([5], device=DEVICE, dtype=torch.int32)
    kept = cutedsl_beam.keep_scored(index_queries, scores, candidates, centroids, children, 5, lengths)
    named = (kept.view(5, NODE_BUCKET_SIZE) >= 0).all(dim=1)
    assert named.tolist() == [True, False, False, True, True]
