"""Context parallel across real ranks - the fork's gather and the decode group's sum run as collectives.

The unit tests stand a group up in one process, with a gather or a sum replaced by reading every rank's terms. Here
each rank is a process on a GPU of its own, sglang's parallel state is built the way a server builds it - prefill
context parallel, decode context parallel and tensor parallel over the same ranks, as the served layout has them -
and the rows move through NCCL. Every rank must end up with what one rank holding everything has.
Under fixed and tree that includes the tree, grown at three page ends from index keys the group gathers and
sums, which every rank checks against the group's after every forward. Under the key and query input a rank holds
every head and the chunk's key latents whole, and its share of the pool's key latents, which it projects and sums.

Needs two GPUs or more and the fork's context-parallel image, the only one with the layouts.
"""

import socket
from collections.abc import Callable, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from lsa.sglang import context_parallel
from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index.hierarchy import Hierarchy
from lsa.sglang.index.layer_index import KeyAndQuery, KeyBuffer, LayerIndex, Shard, Span
from lsa.sglang.settings.design import DesignSettings
from tests.unit.sglang.test_layer_index import (
    CHUNK,
    KEY_DIM,
    TREE_CHUNKS,
    TREE_GROWTH_BUCKETS,
    TREE_LENGTH,
    TREES,
    LatentLayout,
    Layout,
    latent_maps,
    maps,
    sets,
)

pytestmark = pytest.mark.gpu

# The served layout's strategy, and the one its gather comes from
STRATEGIES = ("contiguous", "zigzag")
# A KV row as the pool stores it, with and without GLM's index keys, and an index key
ROW_FORMATS = ((torch.uint8, 1156), (torch.uint8, 1024), (torch.bfloat16, 64))
INDEX_DIM = 64


def group_size() -> int:
    """Return how many ranks the group runs at, skipping where there is no group or no fork to run it"""
    pytest.importorskip(
        "sglang.srt.layers.cp.base", reason="the context-parallel layouts are on the fork's image"
    )
    if torch.cuda.device_count() < 2:
        pytest.skip("a group needs two GPUs or more")
    return min(torch.cuda.device_count(), 4)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def run_ranks(check: Callable[[int, int], None]) -> None:
    """Run a check on every rank of a group, each a process on its own GPU"""
    size = group_size()
    mp.spawn(join_group, args=(size, free_port(), check), nprocs=size, join=True)


def join_group(rank: int, size: int, port: int, check: Callable[[int, int], None]) -> None:
    """Build sglang's parallel state for one rank as a server does, then run the check on it"""
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from sglang.test.test_utils import publish_build_topology

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    publish_build_topology(
        world_rank=rank,
        tp_size=size,
        enable_prefill_cp=True,
        cp_strategy="contiguous",
        attn_cp_size=size,
        dcp_size=size,
    )
    initialize_model_parallel()
    try:
        check(rank, size)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def chunk_batch(prefixes: Sequence[int], lengths: Sequence[int]) -> Any:
    """Return a prefill batch with the metadata the model runner gives it under prefill context parallel"""
    from sglang.srt.layers.cp.utils import prepare_cp_forward
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.zeros(sum(lengths), dtype=torch.int64, device="cuda"),
        seq_lens_cpu=torch.tensor([prefix + n for prefix, n in zip(prefixes, lengths, strict=True)]),
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefixes),
        attn_cp_metadata=None,
        out_cache_loc=None,
    )
    assert context_parallel.is_active(batch)
    prepare_cp_forward(batch)
    return batch


def strategy_batch(name: str, size: int, prefixes: Sequence[int], lengths: Sequence[int]) -> tuple[Any, Any]:
    """Return a strategy of the named layout and a batch carrying its metadata, padded as the model runner pads it"""
    from sglang.srt.layers.cp.contiguous import ContiguousCPStrategy
    from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
    from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

    strategy = {"contiguous": ContiguousCPStrategy, "zigzag": ZigzagCPStrategy}[name](size)
    seq_lens = [prefix + n for prefix, n in zip(prefixes, lengths, strict=True)]
    metadata = strategy.build_metadata(sum(lengths), seq_lens, list(lengths))
    pad_logical_token_to_physical(metadata)
    return strategy, SimpleNamespace(attn_cp_metadata=metadata, extend_seq_lens_cpu=list(lengths))


def chunk_rows(token_count: int, dtype: torch.dtype, width: int, seed: int) -> torch.Tensor:
    """The same rows on every rank, a different value in every byte"""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if dtype == torch.uint8:
        return torch.randint(0, 256, (token_count, width), generator=generator, device="cuda", dtype=dtype)
    return torch.randn(token_count, width, generator=generator, device="cuda").to(dtype)


def chunks(size: int) -> list[tuple[list[int], list[int]]]:
    """Prefixes and new tokens of the chunks the gathers are tried on, each request long enough for zigzag"""
    return [
        ([0, 300, 17], [37, 1003, 2 * size + 1]),
        ([0], [4096]),
        ([0], [2 * size]),
        ([5000, 0], [2 * size, 2 * size + 3]),
    ]


def check_fork_gathers(rank: int, size: int) -> None:
    for name in STRATEGIES:
        for prefixes, lengths in chunks(size):
            strategy, batch = strategy_batch(name, size, prefixes, lengths)
            mine = batch.attn_cp_metadata.per_rank_logical_token[rank]
            for seed, (dtype, width) in enumerate(ROW_FORMATS):
                rows = chunk_rows(sum(lengths), dtype, width, seed)
                # The model runner hands a layer its rows padded to the group's longest share
                local = strategy.shard_hidden_states(rows, batch)
                assert local.shape[0] == batch.attn_cp_metadata.per_rank_actual_token[rank]
                for share in (local, local[:mine]):
                    gathered = strategy.gather_kv_cache(share.contiguous(), batch)
                    assert gathered.dtype == dtype
                    assert torch.equal(gathered, rows), (name, lengths, dtype, width, share.shape)


def test_the_fork_gathers_a_chunk_s_kv_rows_and_index_keys_whole_on_every_rank():
    run_ranks(check_fork_gathers)


def check_gather_chunk(rank: int, size: int) -> None:
    from sglang.srt.layers.cp.base import get_cp_strategy

    # A chunk of one token a rank is padded to as many rows as the chunk has tokens
    for prefixes, lengths in [*chunks(size), ([0], [size]), ([0, 0], [size, size + 1])]:
        batch = chunk_batch(prefixes, lengths)
        metadata = batch.attn_cp_metadata
        assert sum(context_parallel.local_query_counts(batch)) == metadata.per_rank_logical_token[rank]
        for seed, (dtype, width) in enumerate(ROW_FORMATS):
            rows = chunk_rows(sum(lengths), dtype, width, seed)
            local = get_cp_strategy().shard_hidden_states(rows, batch)
            assert torch.equal(context_parallel.gather_chunk(local, batch), rows), (lengths, dtype, width)


def test_a_rank_s_rows_gather_into_the_whole_chunk_through_the_served_layout():
    run_ranks(check_gather_chunk)


def check_decode_group_sum(rank: int, size: int) -> None:
    assert context_parallel.decode_cp() == (size, rank)
    slot_count = 1001
    keys = chunk_rows(slot_count, torch.bfloat16, INDEX_DIM, 0)
    every_slot = torch.arange(slot_count, device="cuda")
    alone = KeyBuffer(slot_count, INDEX_DIM, "cuda", Shard())
    shard = Shard(size, rank, context_parallel.decode_group_sum)
    shared = KeyBuffer(-(-slot_count // size), INDEX_DIM, "cuda", shard)
    # Every rank writes the whole batch, as a decode step does, and keeps the slots it owns
    for buffer in (alone, shared):
        buffer.write(every_slot, keys)
    owned = every_slot[every_slot % size == rank]
    assert torch.equal(shared.keys[owned // size], keys[owned])
    generator = torch.Generator(device="cuda").manual_seed(1)
    for slots in (every_slot, torch.randint(0, slot_count, (4097,), generator=generator, device="cuda")):
        read = shared.read(slots)
        assert torch.equal(read, keys[slots])
        assert torch.equal(read, alone.read(slots))


def test_a_decode_group_reads_every_index_key_by_a_sum_as_one_rank_holding_all_of_them_does():
    run_ranks(check_decode_group_sum)


def rank_layer(layout: Layout, size: int, rank: int, check_index: bool = False) -> LayerIndex:
    rows = -(-layout.slot_count // size)
    return LayerIndex(
        layout.settings,
        maps(layout.seed),
        rows,
        rows,
        lambda: layout.table,
        layout.total,
        CHUNK,
        "cuda",
        Shard(size, rank, context_parallel.decode_group_sum),
        check_index,
    )


def chunk_span(batch: Any) -> Span:
    """The span the indexer reads off a context-parallel batch"""
    return Span(
        context_parallel.chunk_positions(batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu, "cuda"),
        list(batch.extend_seq_lens_cpu),
        context_parallel.local_query_counts(batch),
        context_parallel.local_query_starts(batch),
        lambda rows: context_parallel.gather_chunk(rows, batch),
    )


def own_rows(lengths: Sequence[int], rank: int, size: int) -> torch.Tensor:
    """The chunk rows a rank holds under the contiguous layout - block `rank` of each request"""
    rows, start = [], 0
    for n in lengths:
        rows.append(torch.arange(start + n * rank // size, start + n * (rank + 1) // size))
        start += n
    return torch.cat(rows).cuda()


def check_ranks_select_as_one_rank(rank: int, size: int) -> None:
    from sglang.srt.layers.cp.base import get_cp_strategy

    layout = Layout([3100, 2300])
    alone, mine = layout.layer, rank_layer(layout, size, rank)
    rids, table_rows = ["a", "b"], layout.rows
    # The second chunk starts inside a cluster block, so its keys come partly from the pool and partly from the chunk
    for starts, stops in (([0, 0], [1500, 1500]), ([1500, 1500], [3000, 2200])):
        for layer in (alone, mine):
            layer.maintain(rids, table_rows, starts)
        lengths = [stop - start for start, stop in zip(starts, stops, strict=True)]
        spans = list(enumerate(zip(starts, stops, strict=True)))
        x = torch.cat([layout.x[i][start:stop] for i, (start, stop) in spans])
        slots = torch.cat([layout.slots(i, start, stop) for i, (start, stop) in spans])
        positions = context_parallel.chunk_positions(starts, lengths, "cuda")
        wanted = alone.extend(x, positions, slots, table_rows, lengths, starts)
        batch = chunk_batch(starts, lengths)
        strategy = get_cp_strategy()
        lists = mine.extend(
            strategy.shard_hidden_states(x, batch),
            strategy.shard_position_ids(positions, batch),
            slots,
            table_rows,
            lengths,
            starts,
            chunk_span(batch),
        )
        expected = wanted[own_rows(lengths, rank, size)]
        assert lists.shape == expected.shape
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(expected))
    # The decode step the group takes reads what one rank holding every key reads
    ends = [3000, 2200]
    for layer in (alone, mine):
        layer.maintain(rids, table_rows, ends)
    x = torch.cat([layout.x[i][end : end + 1] for i, end in enumerate(ends)])
    tokens = torch.tensor(ends, device="cuda", dtype=torch.int32)
    slots = torch.cat([layout.slots(i, end, end + 1) for i, end in enumerate(ends)])
    request_rows = torch.tensor(table_rows, device="cuda")
    wanted = alone.decode(x, tokens, slots, request_rows)
    lists = mine.decode(x, tokens, slots, request_rows)
    assert sets(layout.positions_of(lists)) == sets(layout.positions_of(wanted))


def test_prefill_and_decode_context_parallel_ranks_select_as_one_rank_holding_everything_does():
    run_ranks(check_ranks_select_as_one_rank)


def latent_rank_layer(layout: LatentLayout, size: int, rank: int) -> tuple[LayerIndex, torch.Tensor]:
    """A rank's key and query index over its share of the KV pool, its heads summed through the attention TP group"""
    from sglang.srt.layers.dp_attention import attn_tp_all_reduce

    rows = -(-layout.slot_count // size)
    pool = torch.zeros(rows, 1, KEY_DIM + 8, device="cuda", dtype=torch.bfloat16)
    layer = LayerIndex(
        layout.settings,
        latent_maps(layout.seed),
        rows,
        rows,
        lambda: layout.table,
        layout.total,
        CHUNK,
        "cuda",
        Shard(size, rank, context_parallel.decode_group_sum),
        key_buffer=lambda: pool,
        sum_heads=attn_tp_all_reduce,
    )
    return layer, pool


def attend(pool: torch.Tensor, slots: torch.Tensor, keys: torch.Tensor, rank: int, size: int) -> None:
    """Write a forward's key latents as attention does under decode context parallel - the slots this rank owns"""
    owned = slots % size == rank
    pool[slots[owned] // size, 0, :KEY_DIM] = keys[owned]


def check_key_and_query_ranks(rank: int, size: int) -> None:
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.runtime_context import get_parallel

    # Attention is one GPU wide, so a rank holds every head and the sum across its attention TP group adds nothing
    assert get_parallel().attn_tp_size == 1
    layout = LatentLayout([3100, 2300])
    alone = layout.layer
    mine, pool = latent_rank_layer(layout, size, rank)
    rids, table_rows = ["a", "b"], layout.rows
    # The second chunk starts inside a cluster block, so its keys come partly from the pool and partly from the chunk
    for starts, stops in (([0, 0], [1500, 1500]), ([1500, 1500], [3000, 2200])):
        for layer in (alone, mine):
            layer.maintain(rids, table_rows, starts)
        lengths = [stop - start for start, stop in zip(starts, stops, strict=True)]
        spans = list(enumerate(zip(starts, stops, strict=True)))
        key = torch.cat([layout.keys[i][start:stop] for i, (start, stop) in spans])
        query = torch.cat([layout.queries[i][start:stop] for i, (start, stop) in spans])
        slots = torch.cat([layout.slots(i, start, stop) for i, (start, stop) in spans])
        positions = context_parallel.chunk_positions(starts, lengths, "cuda")
        wanted = alone.extend(KeyAndQuery(key, query), positions, slots, table_rows, lengths, starts)
        batch = chunk_batch(starts, lengths)
        strategy = get_cp_strategy()
        # The fork's DSA attention gathers the chunk's key latents, as it gathers a chunk's KV rows
        whole_key = context_parallel.gather_chunk(strategy.shard_hidden_states(key, batch), batch)
        assert torch.equal(whole_key, key)
        lists = mine.extend(
            KeyAndQuery(whole_key, strategy.shard_hidden_states(query, batch)),
            strategy.shard_position_ids(positions, batch),
            slots,
            table_rows,
            lengths,
            starts,
            chunk_span(batch),
        )
        expected = wanted[own_rows(lengths, rank, size)]
        assert lists.shape == expected.shape
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(expected))
        attend(pool, slots, key, rank, size)
    # The decode step reads every earlier key through the group's sum, as one rank holding the whole pool reads it
    ends = [3000, 2200]
    for layer in (alone, mine):
        layer.maintain(rids, table_rows, ends)
    inputs = KeyAndQuery(
        torch.cat([layout.keys[i][end : end + 1] for i, end in enumerate(ends)]),
        torch.cat([layout.queries[i][end : end + 1] for i, end in enumerate(ends)]),
    )
    tokens = torch.tensor(ends, device="cuda", dtype=torch.int32)
    slots = torch.cat([layout.slots(i, end, end + 1) for i, end in enumerate(ends)])
    request_rows = torch.tensor(table_rows, device="cuda")
    wanted = alone.decode(inputs, tokens, slots, request_rows)
    lists = mine.decode(inputs, tokens, slots, request_rows)
    assert sets(layout.positions_of(lists)) == sets(layout.positions_of(wanted))
    # And the pool's every index key reads back through the sum as the one rank's
    every_slot = torch.cat([layout.slots(i, 0, end) for i, end in enumerate(ends)])
    assert torch.equal(mine.buffer.read(every_slot), alone.buffer.read(every_slot))


def test_the_key_and_query_input_under_prefill_and_decode_context_parallel_selects_as_one_rank_does():
    run_ranks(check_key_and_query_ranks)


def check_short_prompt_keys(rank: int, size: int) -> None:
    from sglang.srt.layers.cp.base import get_cp_strategy

    layout = Layout([size])
    alone, mine = layout.layer, rank_layer(layout, size, rank)
    slots = layout.slots(0, 0, size)
    batch = chunk_batch([0], [size])
    alone.write_chunk(layout.x[0], slots)
    mine.write_chunk(get_cp_strategy().shard_hidden_states(layout.x[0], batch), slots, chunk_span(batch))
    assert torch.equal(mine.buffer.read(slots), alone.buffer.read(slots))


def test_a_prompt_of_one_token_a_rank_writes_the_index_keys_one_rank_would():
    run_ranks(check_short_prompt_keys)


def tree_layout(design: str, decode: str) -> Layout:
    """One request of three pages under a tree design, which grows at every page end"""
    hierarchy_module.GROWTH_BUCKETS = TREE_GROWTH_BUCKETS
    return Layout([TREE_LENGTH], settings=DesignSettings.from_blocks(TREES[design], {"decode": decode}))


def tree_of(layer: LayerIndex, row: int) -> Hierarchy:
    hierarchies = layer.selector.cache.hierarchies
    assert hierarchies is not None
    return hierarchies[row]


def prefill_tree_chunk(
    layout: Layout, alone: LayerIndex, mine: LayerIndex, rank: int, size: int, chunk: Any
) -> None:
    """Prefill one chunk on one rank holding everything and on this rank, which must read its share as that one does"""
    from sglang.srt.layers.cp.base import get_cp_strategy

    start, stop = chunk
    table_rows, lengths = layout.rows[:1], [stop - start]
    x, slots = layout.x[0][start:stop], layout.slots(0, start, stop)
    positions = context_parallel.chunk_positions([start], lengths, "cuda")
    wanted = alone.extend(x, positions, slots, table_rows, lengths, [start])
    batch = chunk_batch([start], lengths)
    strategy = get_cp_strategy()
    # Decode context parallel's prefill reads rows of [prefix, chunk] keys - from 0 here, so positions
    lists = mine.extend(
        strategy.shard_hidden_states(x, batch),
        strategy.shard_position_ids(positions, batch),
        slots,
        table_rows,
        lengths,
        [start],
        chunk_span(batch),
        [0],
    )
    assert sets(lists) == sets(layout.positions_of(wanted[own_rows(lengths, rank, size)]))


def check_tree_ranks(design: str, decode: str, rank: int, size: int) -> None:
    layout = tree_layout(design, decode)
    # Every forward checks that every rank holds the same buckets and tree, as a server asked to check does
    alone, mine = layout.layer, rank_layer(layout, size, rank, check_index=True)
    rids, table_rows = ["a"], layout.rows[:1]
    # Two growths inside prefill chunks
    for start, stop in TREE_CHUNKS:
        for layer in (alone, mine):
            layer.maintain(rids, table_rows, [start])
        prefill_tree_chunk(layout, alone, mine, rank, size, (start, stop))
    # And one in a decode step's maintenance, from index keys the group sums
    request_rows = torch.tensor(table_rows, device="cuda")
    for position in range(TREE_CHUNKS[-1][1], TREE_LENGTH):
        for layer in (alone, mine):
            layer.maintain(rids, table_rows, [position])
        token = torch.tensor([position], device="cuda", dtype=torch.int32)
        x, slots = layout.x[0][position : position + 1], layout.slots(0, position, position + 1)
        wanted = alone.decode(x, token, slots, request_rows)
        assert sets(mine.decode(x, token, slots, request_rows)) == sets(wanted)
    held = tree_of(mine, table_rows[0])
    assert held == tree_of(alone, table_rows[0])
    assert held.covered_buckets == 3 * 1024
    assert held.levels == (1 if design == "fixed" else 3)


@pytest.mark.parametrize("decode", ["flat", "descend"])
@pytest.mark.parametrize("design", ["fixed", "tree"])
def test_every_rank_grows_the_tree_one_rank_grows_and_selects_as_it_does(design, decode):
    run_ranks(partial(check_tree_ranks, design, decode))


def check_drifted_tree(rank: int, size: int) -> None:
    layout = tree_layout("tree", "descend")
    alone, mine = layout.layer, rank_layer(layout, size, rank, check_index=True)
    rids, table_rows = ["a"], layout.rows[:1]
    # Past the first page end, so the tree has grown
    for chunk in TREE_CHUNKS[:2]:
        for layer in (alone, mine):
            layer.maintain(rids, table_rows, [chunk[0]])
        prefill_tree_chunk(layout, alone, mine, rank, size, chunk)
    if rank == size - 1:
        # Two level-1 nodes trade children - same shapes, so nothing but the check notices
        children = tree_of(mine, table_rows[0]).closed_node_buckets[0]["children"]
        children[[0, 1]] = children[[1, 0]]
    with pytest.raises(RuntimeError, match="different indexes"):
        mine.check(table_rows)


def test_a_rank_whose_tree_drifted_fails_the_check_on_every_rank():
    run_ranks(check_drifted_tree)
