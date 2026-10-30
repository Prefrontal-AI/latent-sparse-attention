"""The beam's index plumbing and child scoring against plain torch loops."""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.beam import (
    NODE_CENTROID_SCALE,
    bucket_keys,
    cache_row_keys,
    children_of,
    kept_children,
    level_candidates,
    level_scores,
    pick_candidates,
    pick_scores,
    score_children,
)
from lsa.core.kernels.triton.keys import EMPTY, pack_keys, unpack_ids, unpack_scores

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
NODE_BUCKET_SIZE = 4


def beam_inputs(queries=11, open_count=37, carried=9, beam_width=50, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    width = open_count + carried * NODE_BUCKET_SIZE
    scores = torch.randn(queries, width, device=DEVICE, generator=generator)
    # A column the top-k returned from under -inf keeps nothing
    scores[torch.rand(queries, width, device=DEVICE, generator=generator) < 0.2] = -torch.inf
    kept = torch.randint(-1, width + 3, (queries, beam_width), device=DEVICE, generator=generator).int()
    open_children = torch.randint(
        0, 1000, (open_count, NODE_BUCKET_SIZE), device=DEVICE, generator=generator
    ).int()
    # Carried rows lie anywhere in the closed table, a node bucket's four nowhere in particular
    carried_children = torch.randint(
        -1, 20, (queries, carried * NODE_BUCKET_SIZE), device=DEVICE, generator=generator
    ).int()
    closed_children = torch.randint(0, 1000, (20, NODE_BUCKET_SIZE), device=DEVICE, generator=generator).int()
    return scores, kept, open_count, open_children, carried_children, closed_children


def kept_children_loop(scores, kept, open_count, open_children, carried_children, closed_children):
    out = torch.full((*kept.shape, NODE_BUCKET_SIZE), -1, dtype=torch.int32)
    for query in range(kept.shape[0]):
        for slot in range(kept.shape[1]):
            column = int(kept[query, slot])
            if column < 0 or column >= scores.shape[1] or scores[query, column] == -torch.inf:
                continue
            if column < open_count:
                out[query, slot] = open_children[column]
                continue
            node = int(carried_children[query, column - open_count])
            if node >= 0:
                out[query, slot] = closed_children[node]
    return out.flatten(1)


def test_kept_columns_name_the_children_of_the_open_node_buckets_and_of_the_carried_ones():
    scores, kept, open_count, open_children, carried_children, closed_children = beam_inputs()
    got = kept_children(
        scores, kept, open_count, open_children, carried_children, closed_children, NODE_BUCKET_SIZE
    )
    want = kept_children_loop(scores, kept, open_count, open_children, carried_children, closed_children)
    assert torch.equal(got.cpu(), want)


def test_at_the_top_level_only_the_open_node_buckets_are_kept():
    scores, kept, open_count, open_children, _, _ = beam_inputs(carried=0)
    got = kept_children(scores[:, :open_count], kept, open_count, open_children, None, None, NODE_BUCKET_SIZE)
    want = kept_children_loop(scores[:, :open_count], kept, open_count, open_children, None, None)
    assert torch.equal(got.cpu(), want)


def bucket_rows_loop(selected, scores, carried_children, pages, open_count, covered_buckets, page_size):
    out = torch.full(selected.shape, -1, dtype=torch.int32)
    for query in range(selected.shape[0]):
        for slot in range(selected.shape[1]):
            column = int(selected[query, slot])
            if column < 0 or column >= scores.shape[1] or scores[query, column] == -torch.inf:
                continue
            if column < open_count:
                logical = covered_buckets + column
                page = int(pages[logical // page_size])
                if page >= 0:
                    out[query, slot] = page * page_size + logical % page_size
                continue
            out[query, slot] = carried_children[query, column - open_count]
    return out


@pytest.mark.parametrize("carried", [0, 9])
def test_a_pick_is_a_cache_row_through_the_pages_or_the_row_its_node_bucket_names(carried):
    scores, selected, open_count, _, carried_children, _ = beam_inputs(carried=carried, beam_width=40, seed=1)
    pages = torch.randint(-1, 50, (30,), device=DEVICE).int()
    covered = 21
    kept = None if carried == 0 else carried_children
    # Picks in the order a top-k lists them, the columns that are none the empty key
    columns = selected.clamp(0, scores.shape[1] - 1).long()
    keys = pack_keys(
        scores.gather(1, columns), torch.where((selected >= 0) & (selected < scores.shape[1]), selected, -1)
    )
    counts = torch.full((keys.shape[0],), keys.shape[1], dtype=torch.int32, device=DEVICE)
    got = bucket_keys(keys, counts, kept, pages, open_count, covered, 8)
    want = bucket_rows_loop(selected, scores, kept, pages, open_count, covered, 8)
    assert torch.equal(unpack_ids(got).cpu(), want)
    live = want >= 0
    assert torch.equal(unpack_scores(got).cpu()[live], scores.gather(1, columns).cpu()[live])


def test_children_are_scored_from_the_rows_they_name():
    torch.manual_seed(0)
    index_queries = F.normalize(torch.randn(37, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    centroids = F.normalize(torch.randn(4096, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    # 1,200 rows a query - not a multiple of the kernel's block - every ninth one padding
    children = torch.randint(0, 4096, (37, 1200), device=DEVICE, dtype=torch.int32)
    children[:, ::9] = -1
    scores = score_children(index_queries, centroids, children)
    expected = torch.einsum(
        "sd,scd->sc", index_queries.float(), centroids[children.long().clamp_min(0)].float()
    )
    expected = expected.masked_fill(children < 0, float("-inf"))
    torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)


def test_children_past_a_lengths_end_are_left_unscored():
    torch.manual_seed(0)
    index_queries = F.normalize(torch.randn(6, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    centroids = F.normalize(torch.randn(512, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    children = torch.randint(0, 512, (6, 300), device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([0, 1, 63, 64, 65, 300], device=DEVICE, dtype=torch.int32)
    whole = score_children(index_queries, centroids, children)
    out = torch.full((6, 300), 7.0, device=DEVICE)
    score_children(index_queries, centroids, children, out=out, lengths=lengths)
    inside = torch.arange(300, device=DEVICE)[None] < lengths[:, None]
    assert torch.equal(out[inside], whole[inside])
    assert (out[~inside] == 7.0).all()


def test_fp8_children_score_as_the_rows_they_stand_for():
    torch.manual_seed(0)
    index_queries = F.normalize(torch.randn(37, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    rounded = (F.normalize(torch.randn(4096, 64, device=DEVICE), dim=-1) * NODE_CENTROID_SCALE).to(
        torch.float8_e4m3fn
    )
    children = torch.randint(0, 4096, (37, 1200), device=DEVICE, dtype=torch.int32)
    children[:, ::9] = -1
    scores = score_children(index_queries, rounded, children, centroid_scale=NODE_CENTROID_SCALE)
    # What the bf16 rows they round to score, a sixteenth of the way
    want = score_children(index_queries, rounded.to(torch.bfloat16) / NODE_CENTROID_SCALE, children)
    torch.testing.assert_close(scores, want, rtol=0, atol=1e-6)


def test_children_are_scored_into_a_view_of_a_wider_buffer():
    index_queries = F.normalize(torch.randn(5, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    centroids = F.normalize(torch.randn(256, 64, device=DEVICE), dim=-1).to(torch.bfloat16)
    children = torch.randint(0, 256, (5, 40), device=DEVICE, dtype=torch.int32)
    wide = torch.zeros(5, 100, device=DEVICE)
    out = score_children(index_queries, centroids, children, out=wide[:, 7:47])
    assert torch.equal(out, wide[:, 7:47])
    assert torch.equal(out, score_children(index_queries, centroids, children))
    assert (wide[:, :7] == 0).all() and (wide[:, 47:] == 0).all()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_picked_nodes_name_their_children_rows_and_a_missing_one_none(dtype):
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    table = torch.randint(0, 5000, (300, NODE_BUCKET_SIZE), device=DEVICE, generator=generator).int()
    node_buckets = torch.randint(-1, 300, (13, 70), device=DEVICE, generator=generator).to(dtype)
    want = torch.where(node_buckets[..., None] >= 0, table[node_buckets.clamp_min(0).long()], -1).flatten(1)
    assert torch.equal(children_of(node_buckets, table), want)


def level_candidates_loop(pages, made, closed, start, rows, carried, level, page_size):
    """One request's candidates as the eager beam lists them, as creation indices and then pooled rows"""
    out, lengths = [], []
    for query, row in enumerate(rows.tolist()):
        opened = list(range(int(closed[row]), int(made[row])))
        if int(start[row]) == level:
            creation = opened + list(range(int(closed[row])))
        elif int(start[row]) > level:
            creation = opened + carried[query].tolist()
        else:
            creation = []
        # Trailing nodes that hold nothing are not columns
        while creation and creation[-1] < 0:
            creation.pop()
        pooled = [
            -1 if index < 0 else int(pages[row, index // page_size]) * page_size + index % page_size
            for index in creation
        ]
        out.append(pooled)
        lengths.append(len(pooled))
    return out, lengths


def test_a_level_names_each_requests_node_buckets_as_pooled_rows_in_the_eager_order():
    page_size, width = 8, 60
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    pages = torch.stack([torch.randperm(5, device=DEVICE, generator=generator) for _ in range(5)]).int()
    pages[3, 3:] = -1
    made = torch.tensor([30, 9, 24, 20, 0], device=DEVICE, dtype=torch.int32)
    closed = torch.tensor([10, 4, 8, 7, 0], device=DEVICE, dtype=torch.int32)
    # Above its start, at its start, below it, below it with nothing carried, no levels
    start = torch.tensor([1, 2, 3, 3, 0], device=DEVICE, dtype=torch.int32)
    rows = torch.tensor([2, 0, 1, 3, 4], device=DEVICE)
    carried = torch.randint(-1, 30, (5, 16), device=DEVICE, generator=generator).int()
    carried[3] = -1
    candidates, lengths = level_candidates(pages, made, closed, start, rows, carried, 2, page_size, width)
    wanted, wanted_lengths = level_candidates_loop(
        pages.cpu(), made.cpu(), closed.cpu(), start.cpu(), rows.cpu(), carried.cpu(), 2, page_size
    )
    assert lengths.tolist() == wanted_lengths
    for query, pooled in enumerate(wanted):
        assert candidates[query, : len(pooled)].tolist() == pooled
        assert (candidates[query, len(pooled) :] == -1).all()


def test_a_pick_names_the_uncovered_buckets_as_cache_rows_and_then_the_carried_ones():
    page_rows, block_positions, block_buckets, width = 16, 64, 8, 120
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    pages = torch.stack([torch.randperm(8, device=DEVICE, generator=generator) for _ in range(4)]).int()
    covered = torch.tensor([0, 24, 40, 8], device=DEVICE, dtype=torch.int32)
    # 16 buckets seen, 56, 64 and none: before the first block, past the covered buckets, and the position of a padded row
    positions = torch.tensor([140, 450, 520, -1], device=DEVICE, dtype=torch.int32)
    rows = torch.tensor([0, 1, 2, 3], device=DEVICE)
    carried = torch.randint(-1, 3000, (4, 30), device=DEVICE, generator=generator).int()
    carried[2, 20:] = -1
    candidates, lengths = pick_candidates(
        positions, rows, covered, pages, carried, page_rows, block_positions, block_buckets, width + 30
    )
    for query in range(4):
        visible = max(int(positions[query]) // block_positions * block_buckets, 0)
        own = [
            int(pages[query, logical // page_rows]) * page_rows + logical % page_rows
            for logical in range(int(covered[query]), visible)
        ]
        children = carried[query].tolist()
        while children and children[-1] < 0:
            children.pop()
        wanted = own + children
        assert lengths[query].item() == len(wanted)
        assert candidates[query, : len(wanted)].tolist() == wanted
        assert (candidates[query, len(wanted) :] == -1).all()


def test_listed_columns_become_the_cache_rows_they_stand_for_with_their_scores_kept():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    scores = torch.randn(5, 40, device=DEVICE, generator=generator)
    scores[:, ::7] = -torch.inf
    candidates = torch.randint(-1, 500, (5, 40), device=DEVICE, generator=generator).int()
    columns = torch.randint(0, 40, (5, 12), device=DEVICE, generator=generator).int()
    keys = pack_keys(scores.gather(1, columns.long()), columns)
    keys[:, 9:] = EMPTY
    listed = torch.tensor([12, 3, 0, 9, 11], device=DEVICE, dtype=torch.int32)
    got = cache_row_keys(keys, listed, candidates)
    for query in range(5):
        for slot in range(12):
            column = int(columns[query, slot])
            row = int(candidates[query, column])
            score = scores[query, column].item()
            held = slot < int(listed[query]) and slot < 9 and score > -torch.inf and row >= 0
            if not held:
                assert got[query, slot].item() == EMPTY
                continue
            assert unpack_scores(got[query, slot]).item() == score
            assert 0x7FFFFFFF - (int(got[query, slot]) & 0xFFFFFFFF) == row


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_a_level_scored_in_one_launch_is_the_candidates_then_the_scores(dtype):
    page_size, width, node_buckets = 8, 300, 40
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    pages = torch.stack([torch.randperm(5, device=DEVICE, generator=generator) for _ in range(5)]).int()
    pages[3, 3:] = -1
    made = torch.tensor([30, 9, 24, 20, 0], device=DEVICE, dtype=torch.int32)
    closed = torch.tensor([10, 4, 8, 7, 0], device=DEVICE, dtype=torch.int32)
    start = torch.tensor([1, 2, 3, 3, 0], device=DEVICE, dtype=torch.int32)
    rows = torch.tensor([2, 0, 1, 3, 4], device=DEVICE)
    carried = torch.randint(-1, 30, (5, 200), device=DEVICE, generator=generator).int()
    carried[3] = -1
    scale = 16.0 if dtype == torch.float8_e4m3fn else 1.0
    centroids = (
        F.normalize(torch.randn(node_buckets, 64, device=DEVICE, generator=generator), dim=1) * scale
    ).to(dtype)
    index_queries = F.normalize(torch.randn(5, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    wanted, wanted_lengths = level_candidates(pages, made, closed, start, rows, carried, 2, page_size, width)
    wanted_scores = score_children(index_queries, centroids, wanted, centroid_scale=scale)
    candidates, scores, lengths = level_scores(
        pages, made, closed, start, rows, carried, index_queries, centroids, 2, page_size, width, scale
    )
    assert torch.equal(candidates, wanted)
    assert torch.equal(lengths, wanted_lengths)
    assert torch.equal(scores, wanted_scores)


def test_a_pick_scored_in_one_launch_is_the_candidates_then_the_scores():
    page_rows, block_positions, block_buckets, width = 16, 64, 8, 150
    generator = torch.Generator(device=DEVICE).manual_seed(10)
    pages = torch.stack([torch.randperm(8, device=DEVICE, generator=generator) for _ in range(4)]).int()
    covered = torch.tensor([0, 24, 40, 8], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([140, 450, 520, -1], device=DEVICE, dtype=torch.int32)
    rows = torch.tensor([0, 1, 2, 3], device=DEVICE)
    carried = torch.randint(-1, 128, (4, 30), device=DEVICE, generator=generator).int()
    carried[2, 20:] = -1
    centroids = F.normalize(torch.randn(128, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    index_queries = F.normalize(torch.randn(4, 64, device=DEVICE, generator=generator), dim=1).bfloat16()
    wanted, wanted_lengths = pick_candidates(
        positions, rows, covered, pages, carried, page_rows, block_positions, block_buckets, width
    )
    wanted_scores = score_children(index_queries, centroids, wanted)
    candidates, scores, lengths = pick_scores(
        positions,
        rows,
        covered,
        pages,
        carried,
        index_queries,
        centroids,
        page_rows,
        block_positions,
        block_buckets,
        width,
    )
    assert torch.equal(candidates, wanted)
    assert torch.equal(lengths, wanted_lengths)
    assert torch.equal(scores, wanted_scores)
