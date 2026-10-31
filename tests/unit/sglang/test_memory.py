"""The indexer's share of a GPU as sglang's pool sizing counts it - the arithmetic, and that it covers what is built.

The bytes a slot are checked against GLM-5.3-Flash's own figures, and the count against the tensors a layer's index
and a grown hierarchy actually hold, on a CPU.
"""

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.sglang import memory
from lsa.sglang.index import pisa
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.graph_tree import GraphTree, level_pages
from lsa.sglang.index.layer_index import KeySlot, KeySplit, LayerIndex
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings, PisaConfig
from tests.unit.sglang.test_qwen3_5 import QWEN38

# GLM-5.3-Flash on one node at tp 8 - its sparse layers, width and the pool sglang sized at 0.85 without the count
GLM_LAYERS = 11
GLM_MODEL_DIM = 4096
GLM_SLOTS = 7_429_568
# sglang's own cell under LSA - 1,024 B of KV a sparse layer
GLM_CELL = 1024 * GLM_LAYERS
# The attention input, a research option, stated - the default is the key and query, counted below
ATTENTION_INPUT = {"input": "attention_input"}
FLAT = DesignSettings.from_blocks(ATTENTION_INPUT, {})
FIXED = DesignSettings.from_blocks(
    {
        **ATTENTION_INPUT,
        "top_level": 1,
        "node_bucket_size": 4,
        "node_cluster_block_size": 1024,
        "beam_width": 2048,
    },
    {},
)
TREE_KNOBS = {
    **ATTENTION_INPUT,
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 4096,
    "beam_width": 2048,
    "start_nodes": 32768,
}
TREE = DesignSettings.from_blocks(TREE_KNOBS, {"node_centroid_precision": "fp8"})
SHAPE = {"rank": 2, "deviations": 2.0}
KEY_AND_QUERY = {"input": "key_and_query", "query_heads": "all"}
FLAT_KQ = DesignSettings.from_blocks(KEY_AND_QUERY, {})
FIXED_KQ = DesignSettings.from_blocks(
    {
        **KEY_AND_QUERY,
        "top_level": 1,
        "node_bucket_size": 4,
        "node_cluster_block_size": 1024,
        "beam_width": 2048,
    },
    {},
)
# GLM-5.3-Flash's key latent, which the key and query input projects
GLM_KEY_DIM = 512


def glm_share(settings, decode_cp=1, requests=65):
    return memory.share(settings, GLM_LAYERS, requests, 1_048_576 + 64, 8192, GLM_MODEL_DIM, 64, decode_cp)


def test_the_key_and_query_input_keeps_no_index_keys_and_bf16_maps_of_the_latent():
    attention_input, key_and_query = (
        glm_share(FLAT),
        memory.share(FLAT_KQ, GLM_LAYERS, 65, 1_048_576 + 64, 8192, GLM_KEY_DIM, 64),
    )
    # The index keys are projected from the pool's latents, so a slot costs the buckets alone
    assert key_and_query.index_keys == 0
    assert key_and_query.buckets == attention_input.buckets
    assert key_and_query.cell == math.ceil(GLM_LAYERS * (24.5 + 4 / PAGE_SIZE))
    # No index-key buffer past the pool, and bf16 maps of the 512-dim latent in place of fp32 ones of x
    maps = GLM_MODEL_DIM * 2 * 64 * 4 - GLM_KEY_DIM * 2 * 64 * 2
    assert attention_input.fixed - key_and_query.fixed == GLM_LAYERS * (maps + 64 * 128)


def test_the_pool_shrinks_by_the_share_and_the_slack_is_left_alone():
    counted = glm_share(FLAT)
    budget = GLM_SLOTS * GLM_CELL
    slots = (budget - counted.reserved) // (GLM_CELL + counted.cell)
    # What the pool and the indexer hold together is the budget, not a byte past it
    assert slots * GLM_CELL + counted.pool_bytes(slots) <= budget
    assert (slots + 1) * GLM_CELL + counted.pool_bytes(slots + 1) > budget
    shrink = GLM_CELL / (GLM_CELL + counted.cell) * (1 - counted.reserved / budget)
    assert slots / GLM_SLOTS == pytest.approx(shrink, rel=1e-6)


def test_a_hierarchy_counts_every_level_at_twice_its_rows():
    # One level of nodes over 4 buckets of 8 keys - a bf16 centroid and 4 children, open for good at the top
    assert glm_share(FIXED).hierarchy == pytest.approx(2 * (128 + 16) / 32)
    # Below the top, open with an fp32 key sum and closed without it
    expected = sum((80 + (256 + 80 if level < 16 else 0)) / (8 * 4**level) for level in range(1, 17))
    assert glm_share(TREE).hierarchy == pytest.approx(2 * expected)


def test_decode_context_parallel_counts_every_rank_s_buckets_and_its_own_keys():
    alone, split = glm_share(TREE), glm_share(TREE, decode_cp=8)
    assert split.index_keys == alone.index_keys
    assert split.buckets == pytest.approx(8 * alone.buckets)
    assert split.hierarchy == pytest.approx(8 * alone.hierarchy)
    # The key and query writes no index keys under it either, and holds every slot's buckets
    latent = memory.share(FLAT_KQ, GLM_LAYERS, 17, 2_100_064, 32768, GLM_KEY_DIM, 64, decode_cp=8)
    assert latent.index_keys == 0
    assert latent.buckets == pytest.approx(8 * alone.buckets)


DESCEND_KNOBS = {
    "window_size": 64,
    "index_dim": 16,
    "top_k": 64,
    "top_p": 16,
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 256,
    "beam_width": 2048,
}
DESCEND = DesignSettings.from_blocks(DESCEND_KNOBS, {"decode": "descend", "node_centroid_precision": "fp8"})


def test_a_descending_decode_counts_its_graph_tree_a_slot_on_every_rank_of_the_decode_group():
    kq = DesignSettings.from_blocks(
        {**TREE_KNOBS, **KEY_AND_QUERY}, {"decode": "descend", "node_centroid_precision": "fp8"}
    )
    single = memory.share(kq, GLM_LAYERS, 17, 2_100_064, 32768, GLM_KEY_DIM, 64)
    split = memory.share(kq, GLM_LAYERS, 17, 2_100_064, 32768, GLM_KEY_DIM, 64, decode_cp=8)
    # Each node bucket pools a 64 dim fp8 centroid and 4 children, and level `l` holds one for each 4**l bucket rows of 8 keys
    levels = len(level_pages(kq, 0, 0, 2_100_064)[0])
    assert 4 <= levels < 16
    assert single.graph_tree == pytest.approx(
        sum((64 + 16) / (8 * 4**level) for level in range(1, levels + 1))
    )
    assert split.graph_tree == pytest.approx(8 * single.graph_tree)
    assert split.slot - single.slot == pytest.approx(
        7 * (single.buckets + single.hierarchy + single.graph_tree)
    )
    # The pools are allocated whole, so the page of each level a request holds is counted beside the slots
    assert split.fixed == single.fixed


def test_a_design_that_builds_no_graph_tree_counts_none():
    for design in (TREE, FLAT, DesignSettings.from_blocks(TREE_KNOBS, {"decode": "flat"})):
        assert memory.share(design, GLM_LAYERS, 17, 2_100_064, 32768, 64, 64).graph_tree == 0
        assert memory.graph_tree_bytes(design, 17, 2_100_064) == (0.0, 0)
    shaped = DesignSettings.from_blocks(
        {**TREE_KNOBS, "node_bucket_shape": SHAPE}, {"decode": "descend", "node_centroid_precision": "fp8"}
    )
    assert memory.graph_tree_bytes(shaped, 17, 2_100_064) == (0.0, 0)


def test_the_counted_graph_tree_covers_what_the_cache_allocates_and_is_close_to_it():
    requests, context = 3, 5 * PAGE_SIZE + 300
    kv_size = 5 * PAGE_SIZE
    cache = BucketCache(DESCEND, kv_size, requests, context, "cpu")
    tree = cache.graph_tree
    assert tree is not None and tree.level_count >= 3
    tensors = [
        *tree.node_centroids,
        *tree.children,
        *tree.pages,
        tree.made,
        tree.closed,
        tree.start,
        tree.covered,
    ]
    built = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert built == GraphTree.pool_bytes(DESCEND, len(cache.bucket_sizes), requests, context)
    per_key, fixed = memory.graph_tree_bytes(DESCEND, requests, context)
    spare = requests + DESCEND.serving.spare_contexts * cache.max_pages
    counted = per_key * (kv_size + spare * PAGE_SIZE) + fixed
    assert built <= counted
    # Within the rounding up of a page of each level
    assert counted - built <= 2 * tree.level_count * 256 * sum(GraphTree.node_bucket_bytes(DESCEND).values())


def test_tensor_parallel_splitting_the_key_heads_counts_an_index_key_a_slot_beside_the_pool():
    # Qwen3.8-27B under tensor parallel 8 - 16 quadratic layers, 768 position-free dims, 1 KiB of K and V a layer
    qwen = (16, 3, 10_485_760 + 64, 32768, 768, 1)
    pooled = memory.share(FLAT_KQ, *qwen)
    held = memory.share(FLAT_KQ, *qwen, split_heads=True)
    assert pooled.index_keys == 0
    assert held.index_keys == 128
    assert held.buckets == pooled.buckets and held.hierarchy == pooled.hierarchy
    # 128 B a slot a layer over 16 layers, 21.5 GB at 10.5M slots
    assert held.cell - pooled.cell == 16 * 128
    assert (held.cell - pooled.cell) * 10_485_760 == pytest.approx(21.5e9, rel=1e-2)
    # The one spare row past the pool, which the attention input holds as well
    assert held.fixed - pooled.fixed == 16 * 128
    # The attention input's keys are counted whether or not heads are split
    assert memory.share(FLAT, *qwen, split_heads=True).index_keys == 128


def test_the_count_covers_the_index_keys_a_split_layer_holds(monkeypatch):
    width, requests, context, chunk, page = 96, 3, 3 * PAGE_SIZE, 512, 64
    slots = 2 * PAGE_SIZE + 100
    table = torch.zeros(requests, context, dtype=torch.int32)
    maps = torch.randn(width, 64), torch.randn(width, 64)
    split = KeySplit(0, 24, True, lambda term: term)
    index = LayerIndex(FLAT_KQ, maps, slots + page, slots, lambda: table, context, chunk, "cpu", split=split)
    built = sum(tensors_of(index).values())
    counted = memory.share(FLAT_KQ, 1, requests, context, chunk, width, page, split_heads=True).pool_bytes(
        slots
    )
    assert built <= counted
    # The buffer is in what is counted - without it the layer would hold more than the share
    assert built > memory.share(FLAT_KQ, 1, requests, context, chunk, width, page).pool_bytes(slots)


def tensors_of(holder, found=None):
    """Every distinct storage an object holds through its attributes, lists and dicts, to its bytes"""
    found = {} if found is None else found
    if isinstance(holder, torch.Tensor):
        found[holder.untyped_storage().data_ptr()] = holder.untyped_storage().nbytes()
    elif isinstance(holder, dict):
        for value in holder.values():
            tensors_of(value, found)
    elif isinstance(holder, list | tuple):
        for value in holder:
            tensors_of(value, found)
    elif hasattr(holder, "__dict__"):
        tensors_of(vars(holder), found)
    return found


@pytest.mark.parametrize(
    "settings",
    [FLAT, FIXED, FLAT_KQ, FIXED_KQ],
    ids=["flat", "fixed", "flat-key-and-query", "fixed-key-and-query"],
)
def test_the_count_covers_every_tensor_a_layer_s_index_is_built_with(settings):
    width, requests, context, chunk, page = 96, 3, 3 * PAGE_SIZE, 512, 64
    slots = 2 * PAGE_SIZE + 100
    table = torch.zeros(requests, context, dtype=torch.int32)
    maps = torch.randn(width, 64), torch.randn(width, 64)
    # The KV pool's own buffer, which sglang counts - the key and query reads its latents
    pool = torch.zeros(slots + page, 1, width + 8, dtype=torch.bfloat16)
    index = LayerIndex(
        settings, maps, slots + page, slots, lambda: table, context, chunk, "cpu", key_buffer=lambda: pool
    )
    built = sum(tensors_of(index).values())
    counted = memory.share(settings, 1, requests, context, chunk, width, page).pool_bytes(slots)
    # The beam's select and every listing are built on first use
    lazy = chunk * 2048 * 4 if settings.indexer.top_level else 0
    assert built <= counted - lazy
    # Over by the hierarchy not grown yet and a page of buckets of rounding at most
    per_key = memory.bucket_bytes(settings.indexer) + memory.hierarchy_bytes(settings.indexer, "bf16")
    slack = slots * memory.hierarchy_bytes(settings.indexer, "bf16") + 2 * PAGE_SIZE * per_key
    assert counted - lazy - built <= slack + 1


def test_one_layer_s_selection_peak_is_counted_once_and_so_is_what_the_layers_share():
    counted = memory.share(FIXED, GLM_LAYERS, 65, 1_048_640, 8192, GLM_MODEL_DIM, 64, transient=3 << 30)
    alone = memory.share(FIXED, GLM_LAYERS, 65, 1_048_640, 8192, GLM_MODEL_DIM, 64)
    assert counted.reserved - alone.reserved == 3 << 30
    # A fused select's output and a beam select a width, once however many layers
    one = memory.share(FIXED, 1, 65, 1_048_640, 8192, GLM_MODEL_DIM, 64)
    many = memory.share(FIXED, 16, 65, 1_048_640, 8192, GLM_MODEL_DIM, 64)
    assert many.shared == one.shared == memory.shared_bytes(FIXED, 8192) > 0
    assert many.fixed == 16 * one.fixed


def test_the_reservation_does_not_grow_with_the_layers_by_a_listing():
    chunk = 32_768
    one = memory.share(FLAT, 1, 65, 1_048_640, chunk, GLM_MODEL_DIM, 64)
    many = memory.share(FLAT, 16, 65, 1_048_640, chunk, GLM_MODEL_DIM, 64)
    # What a layer adds is its projections and tables, far under the 1 GiB a chunk's listing of 4 x 1,024 int64
    # keys a row would have cost each
    assert (many.reserved - one.reserved) / 15 < (chunk * 4 * 1024 * 8) / 2
    assert memory.shared_bytes(FLAT, chunk) == chunk * 1024 * 4


@pytest.mark.gpu
@pytest.mark.parametrize("settings", [FLAT, FIXED], ids=["flat", "fixed"])
def test_a_layer_s_listings_are_not_held_after_a_read(settings):
    context, chunk = 40 * PAGE_SIZE, 1024
    table = torch.arange(context, device="cuda", dtype=torch.int32)[None]
    maps = torch.randn(96, 64), torch.randn(96, 64)
    index = LayerIndex(settings, maps, context, context, lambda: table, context, chunk, "cuda")
    index.buffer.keys[:] = F.normalize(torch.randn(context, 64, device="cuda"), dim=-1).bfloat16()
    for end in (8 * PAGE_SIZE, 20 * PAGE_SIZE, context):
        index.maintain(["request"], [0], [end - chunk])
        positions = torch.arange(end - chunk, end, device="cuda", dtype=torch.int32)
        index.extend(
            torch.randn(chunk, 96, device="cuda").bfloat16(),
            positions,
            positions.long(),
            [0],
            [chunk],
            [end - chunk],
        )
    selector = index.selector
    top_ks = [selector.prefill_top_k, selector.tree_top_k, *selector.scratch.beam_top_ks.values()]
    kept = sum(
        tensors_of([[vars(top_k).get("keys"), vars(top_k).get("counts")] for top_k in top_ks]).values()
    )
    assert kept == 0


@pytest.mark.gpu
def test_the_selection_probe_measures_a_peak_and_leaves_nothing_behind():
    # The first reads set up the libraries' own workspaces, which stay
    memory.probe_transient(FIXED, 40 * PAGE_SIZE, 1024, 96)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    flat = memory.probe_transient(FLAT, 40 * PAGE_SIZE, 1024, 96)
    fixed = memory.probe_transient(FIXED, 40 * PAGE_SIZE, 1024, 96)
    assert torch.cuda.memory_allocated() == before
    # At least a chunk's read of its top-p listing, and a tree's level of nodes on top
    assert flat >= 1024 * 4 * 1024 * 8
    assert fixed > flat


@pytest.mark.gpu
def test_the_selection_probe_reads_the_key_and_query_from_a_pool_of_its_own():
    memory.probe_transient(FLAT_KQ, 40 * PAGE_SIZE, 1024, 96, 4, KeySlot.latent(96, 8))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    flat = memory.probe_transient(FLAT_KQ, 40 * PAGE_SIZE, 1024, 96, 4, KeySlot.latent(96, 8))
    fixed = memory.probe_transient(FIXED_KQ, 40 * PAGE_SIZE, 1024, 96, 4, KeySlot.latent(96, 8))
    assert torch.cuda.memory_allocated() == before
    assert flat >= 1024 * 4 * 1024 * 8
    assert fixed > flat


def test_a_grown_hierarchy_holds_no_more_than_its_count():
    # Node cluster blocks of 256, the least whose first growth fills a level's first rows, so five pages grow three levels
    settings = DesignSettings.from_blocks(
        {
            "window_size": 64,
            "index_dim": 16,
            "top_k": 64,
            "top_p": 16,
            "top_level": 16,
            "node_bucket_size": 4,
            "node_cluster_block_size": 256,
            "beam_width": 2048,
        },
        {},
    )
    keys = F.normalize(torch.randn(5 * PAGE_SIZE + 300, 16), dim=-1).bfloat16()
    cache = BucketCache(settings, len(keys), 3, len(keys), "cpu")
    cache.ensure(1, "request", 1)
    for end in range(3000, len(keys) + 3000, 3000):
        cache.advance(1, min(end, len(keys)), lambda positions: keys[positions.long()], {1})
    hierarchy = cache.hierarchies[1]
    assert hierarchy.levels >= 3
    held = sum(tensors_of([hierarchy.open_node_buckets, hierarchy.closed_node_buckets]).values())
    counted = memory.hierarchy_bytes(settings.indexer, "bf16") * len(keys)
    assert held <= counted
    # A bound, not a guess - within a small factor of what the rows hold
    assert held >= counted / 4


@pytest.mark.gpu
def test_the_selection_probe_reads_a_grouped_query_pool_shaped_as_the_model_s():
    # Qwen3.5's slot scaled down - four key heads of 32 dims past a rotated prefix of 8, one summed query row a token
    slot = KeySlot(heads=4, head_dim=40, offset=8, width=24)
    memory.probe_transient(FLAT_KQ, 40 * PAGE_SIZE, 1024, slot.key_dim, 1, slot)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    flat = memory.probe_transient(FLAT_KQ, 40 * PAGE_SIZE, 1024, slot.key_dim, 1, slot)
    assert torch.cuda.memory_allocated() == before
    assert flat >= 1024 * 4 * 1024 * 8


def test_spare_contexts_are_counted_as_the_pages_the_cache_holds_for_them():
    context = 1_048_576 + 64
    spare = DesignSettings.from_blocks(ATTENTION_INPUT, {"spare_contexts": 4})
    plain = memory.share(FLAT, GLM_LAYERS, 65, context, 8192, GLM_MODEL_DIM, 64)
    counted = memory.share(spare, GLM_LAYERS, 65, context, 8192, GLM_MODEL_DIM, 64)
    pages = 4 * -(-context // PAGE_SIZE)
    assert counted.cell == plain.cell
    assert counted.fixed - plain.fixed == GLM_LAYERS * math.ceil(
        pages * PAGE_SIZE * memory.bucket_bytes(FLAT.indexer)
    )


def grown_tree_cache():
    settings = DesignSettings.from_blocks(
        {
            "window_size": 64,
            "index_dim": 16,
            "top_k": 64,
            "top_p": 16,
            "top_level": 16,
            "node_bucket_size": 4,
            "node_cluster_block_size": 256,
            "beam_width": 2048,
        },
        {},
    )
    keys = F.normalize(torch.randn(5 * PAGE_SIZE + 300, 16), dim=-1).bfloat16()
    cache = BucketCache(settings, len(keys), 3, len(keys), "cpu")
    cache.ensure(1, "request", 1)
    for end in range(3000, len(keys) + 3000, 3000):
        cache.advance(1, min(end, len(keys)), lambda positions: keys[positions.long()], {1})
    return cache, len(keys)


def test_a_request_s_held_node_buckets_are_the_tensors_its_hierarchy_holds():
    cache, _ = grown_tree_cache()
    hierarchy = cache.hierarchies[1]
    held = cache.held(1)
    nodes = sum(count for name, count in held.items() if name.startswith("node_"))
    assert nodes == sum(tensors_of([hierarchy.open_node_buckets, hierarchy.closed_node_buckets]).values())
    assert held["node_centroids"] > 0


def test_a_flat_request_holds_the_bucket_rows_of_its_pages():
    cache = BucketCache(FLAT, 4 * PAGE_SIZE, 3, 4 * PAGE_SIZE, "cpu")
    keys = F.normalize(torch.randn(2 * PAGE_SIZE + 10, FLAT.indexer.index_dim), dim=-1).bfloat16()
    cache.ensure(0, "request", 1)
    cache.advance(0, len(keys), lambda positions: keys[positions.long()], {0})
    rows = len(cache.pages[0]) * cache.rows_per_page
    held = cache.held(0)
    assert held["bucket_centroids"] == rows * FLAT.indexer.index_dim * 2
    assert held["bucket_members"] == rows * (FLAT.indexer.bucket_cap + 1) * 4
    assert not any(name.startswith("node_") for name in held)


def test_a_released_request_is_reported_once_a_new_longest(capsys, monkeypatch):
    monkeypatch.setattr(memory, "COUNTED", {})
    cache, keys = grown_tree_cache()
    cache.release(1)
    first = capsys.readouterr().out
    assert f"a request of {keys} keys held" in first
    assert "node_centroids" in first
    memory.report_held(keys, {"bucket_centroids": keys})
    memory.report_held(keys - 1, {"bucket_centroids": keys})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_qwen_under_pisa_keeps_a_leaf_mean_a_key_head_a_rank_holds_and_gathers_no_query(monkeypatch, tp):
    config = PisaConfig()
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: SimpleNamespace(attn_tp_size=tp)
    runtime.get_schedule = lambda: SimpleNamespace(max_running_requests=16)
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    context = 262_144
    kvc = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(lsa=config.body()),
            hf_text_config=QWEN38["text_config"],
            context_len=context,
        ),
        layer_info=SimpleNamespace(start_layer=0, end_layer=64),
    )
    share = memory.pisa_share(kvc)
    leaves = math.ceil((context + memory.CONTEXT_ALLOWANCE) / config.block_size)
    blocks = sum(pisa.level_widths(leaves, config.branching))
    assert blocks < 2 * leaves + 20
    assert share.fixed == 16 * 17 * blocks * max(1, 4 // tp) * 256 * memory.FP32
    assert share.transient == memory.PISA_TILE_COPIES * pisa.TILE_BYTES
