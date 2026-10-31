import pytest
import torch

from lsa.training.native.kernels import reference
from tests.helpers.training.kernels.inputs import PAD, make_buckets, unit_rows

BLOCK = 32


def naive_selection(
    index_queries, centroids, members, positions, top_p, top_k, window_size, sink_size, bucket_size
):
    """Pick a query's buckets one at a time, as the module docstring says."""
    batch, count = index_queries.shape[:2]
    per_block = BLOCK // bucket_size
    out = torch.full((batch, count, top_k), PAD, dtype=torch.long)
    for b in range(batch):
        for i, position in enumerate(positions.tolist()):
            visible = [m for m in range(members.shape[1]) if m // per_block < position // BLOCK]
            query = index_queries[b, i]
            if query.dim() == 2:
                scores = {m: (query.float() @ centroids[b, m]).max().item() for m in visible}
            else:
                scores = {m: (query.float() @ centroids[b, m]).item() for m in visible}
            best = sorted(visible, key=lambda m: -scores[m])[:top_p]
            taken = []
            for m in best:
                for key in members[b, m].tolist():
                    if (
                        key >= sink_size
                        and key <= position - window_size
                        and key != PAD
                        and len(taken) < top_k
                    ):
                        taken.append(key)
            out[b, i, : len(taken)] = torch.tensor(taken, dtype=torch.long)
    return out


@pytest.mark.parametrize("heads", [None, 3])
def test_the_selection_is_each_visible_buckets_keys_best_bucket_first(heads):
    generator = torch.Generator().manual_seed(0)
    keys = unit_rows((2, 160, 8), generator)
    centroids, members = make_buckets(keys, BLOCK, seed=1, pool_size=2, bucket_size=4, bucket_cap=8)
    shape = (2, 160, 8) if heads is None else (2, 160, heads, 8)
    index_queries = unit_rows(shape, generator)
    positions = torch.arange(160)
    got = reference.select_indices(index_queries, positions, (centroids, members), BLOCK, 4, 6, 20, 8, 2)
    want = naive_selection(index_queries, centroids, members, positions, 6, 20, 8, 2, 4)
    assert torch.equal(got, want)
    assert (got >= 0).any()


def test_no_buckets_select_nothing():
    index_queries = torch.randn(1, 5, 8)
    selected = reference.select_indices(index_queries, torch.arange(5), None, BLOCK, 4, 6, 20, 8, 2)
    assert selected.shape == (1, 5, 20)
    assert (selected == PAD).all()
