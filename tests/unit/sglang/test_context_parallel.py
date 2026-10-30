"""Context parallel without an engine - the span and slot arithmetic, the refusals, and the index keys a decode group splits.

The layouts themselves are sglang's, on the fork's image, and their end to end run is the served smoke. What is
worth holding on a CPU is what the indexer does with them: where a chunk's positions and a request's rows start,
which index keys a rank holds and that a sum across the group gives every rank all of them, and which designs are
refused. Fixed and tree run under both layouts - their parity with one rank is `test_layer_index.py` and
`tests/integ/sglang/test_context_parallel_ranks.py`.
"""

import importlib.util

import pytest
import torch

from lsa.sglang import context_parallel
from lsa.sglang.index.layer_index import KeyBuffer, Shard
from lsa.sglang.settings.design import DesignSettings

FLAT = {
    "window_size": 512,
    "sink_size": 4,
    "top_k": 4096,
    "top_p": 1024,
    "index_dim": 64,
    "pool_size": 4,
    "bucket_size": 8,
    "bucket_cap": 16,
    "top_level": 0,
}
TREE = {
    **FLAT,
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 4096,
    "beam_width": 2048,
    "start_nodes": 32768,
}


def test_a_chunk_s_positions_follow_each_request_s_cached_prefix():
    positions = context_parallel.chunk_positions([5, 0], [3, 2], "cpu")
    assert positions.dtype == torch.int32
    assert positions.tolist() == [5, 6, 7, 0, 1]


def test_a_request_starts_where_the_requests_before_it_end_in_the_ragged_rows():
    assert context_parallel.ragged_offsets([7, 4, 10]) == [0, 7, 11]
    assert context_parallel.ragged_offsets([]) == []


def test_without_sglang_neither_layout_is_active():
    if importlib.util.find_spec("sglang") is not None:
        pytest.skip("sglang is installed, so the layouts answer for themselves")
    assert not context_parallel.is_active(object())
    assert context_parallel.decode_cp() == (1, 0)
    assert context_parallel.prefill_split() is None


@pytest.mark.parametrize("split", ["interleave", "zigzag"])
def test_a_prefill_split_that_scatters_a_request_s_queries_is_refused(split):
    with pytest.raises(ValueError, match="--cp-strategy contiguous"):
        context_parallel.refuse_split(split)


def test_the_contiguous_split_and_no_prefill_context_parallel_run():
    context_parallel.refuse_split("contiguous")
    context_parallel.refuse_split(None)


def test_a_rank_outside_its_group_is_refused():
    with pytest.raises(ValueError, match="not in a group of 4"):
        Shard(4, 4, lambda t: t)


def test_a_group_with_no_sum_is_refused():
    with pytest.raises(ValueError, match="sum across its group"):
        Shard(2, 0, None)


class Group:
    """Decode context-parallel ranks in one process, the terms of every rank summed where a collective would"""

    def __init__(self, size, rows, index_dim=4):
        self.slots = None
        self.buffers = [
            KeyBuffer(rows, index_dim, "cpu", Shard(size, rank, self.total)) for rank in range(size)
        ]

    def total(self, local):
        # Every rank's term for the slots being read - the sum is the same whichever rank asks
        terms = []
        for buffer in self.buffers:
            rows, owned = buffer.rows(self.slots)
            terms.append(buffer.keys[rows] * owned[:, None])
        return sum(terms[1:], terms[0])

    def read(self, rank, slots):
        self.slots = slots
        return self.buffers[rank].read(slots)


def test_a_rank_holds_only_the_index_keys_of_the_slots_it_owns():
    size, slots = 4, 40
    group = Group(size, slots // size)
    keys = torch.randn(slots, 4).bfloat16() + 5
    every_slot = torch.arange(slots)
    for buffer in group.buffers:
        buffer.write(every_slot, keys)
    for rank, buffer in enumerate(group.buffers):
        # Row s // size of the rank that owns slot s, as the KV pool places it
        assert torch.equal(buffer.keys[: slots // size], keys[every_slot % size == rank])
        # The row past the slots took the writes of everyone else's
        assert buffer.keys.shape[0] == slots // size + 1


def test_a_sum_across_the_group_gives_every_rank_every_index_key():
    size, slots = 4, 40
    group = Group(size, slots // size)
    keys = torch.randn(slots, 4).bfloat16() + 5
    for buffer in group.buffers:
        buffer.write(torch.arange(slots), keys)
    wanted = torch.tensor([3, 0, 39, 17, 17, 8])
    for rank in range(size):
        assert torch.equal(group.read(rank, wanted), keys[wanted])


def test_one_rank_holds_every_slot_and_reads_it_back_without_a_sum():
    buffer = KeyBuffer(10, 4, "cpu", Shard())
    keys = torch.randn(10, 4).bfloat16()
    buffer.write(torch.arange(10), keys)
    assert buffer.keys.shape[0] == 10
    assert torch.equal(buffer.read(torch.tensor([9, 2])), keys[[9, 2]])


@pytest.mark.parametrize("knob", ["bucket_shape", "node_bucket_shape"])
def test_a_shaped_design_is_refused_under_decode_context_parallel_and_served_without_it(knob):
    design = DesignSettings.from_blocks({**TREE, knob: {"rank": 2, "deviations": 2}}, {})
    with pytest.raises(ValueError, match=f"{knob} is not served under decode context parallel 8"):
        context_parallel.refuse_shapes(design, 8)
    context_parallel.refuse_shapes(design, 1)
    context_parallel.refuse_shapes(DesignSettings.from_blocks(FLAT, {}), 8)


LATENT = {**FLAT, "input": "key_and_query"}
X = {**FLAT, "input": "attention_input"}


@pytest.mark.parametrize(("decode_size", "split"), [(8, None), (1, "contiguous"), (8, "contiguous")])
def test_the_key_and_query_input_under_context_parallel_reads_every_head_on_every_rank(decode_size, split):
    every_head = DesignSettings.from_blocks(LATENT, {})
    assert every_head.indexer.query_heads == "all"
    context_parallel.refuse_key_and_query(every_head, decode_size, split, 1)
    with pytest.raises(ValueError, match="split over 8 GPUs"):
        context_parallel.refuse_key_and_query(every_head, decode_size, split, 8)
    # A rank's own heads are every head when attention is one GPU wide - no rank's share is left to select for
    with pytest.raises(ValueError, match="query_heads: all"):
        context_parallel.refuse_key_and_query(
            DesignSettings.from_blocks({**LATENT, "query_heads": "rank"}, {}), decode_size, split, 1
        )


def test_the_key_and_query_input_without_context_parallel_and_the_attention_input_under_it_are_served():
    context_parallel.refuse_key_and_query(
        DesignSettings.from_blocks({**LATENT, "query_heads": "rank"}, {}), 1, None, 8
    )
    context_parallel.refuse_key_and_query(DesignSettings.from_blocks(LATENT, {}), 1, None, 8)
    context_parallel.refuse_key_and_query(DesignSettings.from_blocks(X, {}), 8, "contiguous", 8)
