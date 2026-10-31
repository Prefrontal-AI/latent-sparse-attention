"""Qwen3.5 under the design, on a CPU - the geometry, the key and query input against distillation's, and the backend glue.

The input a server reads off the heads attention is handed has to be the one distillation trained the maps on -
`regroup_partial` over `position_free` pre-rotation heads - or a map that scored one way offline selects other keys
served. The rotation is a prefix, so anything done to it must leave the input alone. The backend's hooks are checked
against fakes of the backend and the batch: what they hand on unchanged, what they arm, and that the read attends over
exactly the slots a list names.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.core import RotaryLayout, position_free, regroup_partial
from lsa.sglang.index import registry, weights
from lsa.sglang.index.layer_index import (
    KeyAndQueryProjection,
    KeyBuffer,
    KeySlot,
    KeySplit,
    LayerIndex,
    PoolKeys,
    Shard,
)
from lsa.sglang.models.qwen3_5 import backend, index_space
from lsa.sglang.settings.design import DesignSettings

# Qwen3.8-27B's text config as its checkpoint publishes it
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16
QWEN38 = {
    "text_config": {
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "full_attention_interval": 4,
        "layer_types": LAYER_TYPES,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_type": "yarn", "factor": 40.0},
    }
}
# GLM-5.3-Flash's, in the fields that tell the two apart
GLM = SimpleNamespace(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    num_attention_heads=64,
    layer_types=["linear_attention", "deepseek_sparse_attention"],
)
GEOMETRY = index_space.geometry_of(QWEN38)


def test_qwen38_indexes_its_sixteen_full_attention_layers_in_768_position_free_dims():
    assert GEOMETRY.layers == tuple(range(3, 64, 4))
    assert (GEOMETRY.query_head_count, GEOMETRY.kv_head_count, GEOMETRY.head_dim) == (24, 4, 256)
    assert GEOMETRY.rotary_dim == 64
    assert GEOMETRY.key_dim == 768
    assert GEOMETRY.key_slot == KeySlot(heads=4, head_dim=256, offset=64, width=192)


def test_a_servers_text_config_reads_the_same_as_the_published_one():
    # sglang hands over the text config itself, with the rotary factor lifted onto it and no `layer_types` -
    # its config class drops them and places the layers by the interval
    fields = {
        **QWEN38["text_config"],
        "rope_parameters": {},
        "partial_rotary_factor": 0.25,
        "layer_types": None,
    }
    text = SimpleNamespace(**fields)
    assert index_space.is_grouped_query(text)
    assert index_space.geometry_of(text) == GEOMETRY
    assert weights.indexed_layers(text) == list(GEOMETRY.layers)


def test_only_a_grouped_query_stack_is_served_through_the_backend():
    assert index_space.is_grouped_query(QWEN38)
    assert index_space.is_grouped_query(QWEN38["text_config"])
    assert not index_space.is_grouped_query(GLM)


def test_a_rotation_over_the_whole_head_leaves_nothing_to_index():
    text = {**QWEN38["text_config"], "rope_parameters": {"partial_rotary_factor": 1.0}}
    with pytest.raises(ValueError, match="nothing is\\s+position-free"):
        index_space.geometry_of(text)


def test_the_weights_helpers_read_a_grouped_query_model_without_sglang():
    text = QWEN38["text_config"]
    assert weights.indexed_layers(text) == list(GEOMETRY.layers)
    assert weights.input_dim(text, "key_and_query") == 768
    assert weights.input_dim(SimpleNamespace(**text), "attention_input") == 5120
    assert weights.key_slot(text) == GEOMETRY.key_slot
    assert weights.query_rows(text, 1) == 1
    # GLM's latent sits at the front of one head, before its rope dims
    assert weights.key_slot(GLM) == KeySlot.latent(512, 64)
    assert weights.query_rows(GLM, 8) == 8


def heads(token_count, seed=0):
    """Pre-rotation query and key heads [T, H, dh] bf16, as the norms leave them"""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(token_count, 24, 256, generator=generator).bfloat16()
    k = torch.randn(token_count, 4, 256, generator=generator).bfloat16()
    return q, k


def rotate_prefix(heads_, seed=1):
    """Do anything to the first 64 dims of every head - the server's rotation never reaches past them"""
    generator = torch.Generator().manual_seed(seed)
    rotated = heads_.clone()
    rotated[..., :64] = torch.randn(rotated[..., :64].shape, generator=generator).bfloat16()
    return rotated


def test_the_served_input_is_distillations_regrouped_position_free_heads():
    q, k = heads(37)
    inputs = index_space.key_and_query(rotate_prefix(q).flatten(1), rotate_prefix(k), GEOMETRY)
    # Distillation's read, on [B, H, T, dh] pre-rotation heads
    summed_query, key = regroup_partial(
        *position_free(q.transpose(0, 1)[None], k.transpose(0, 1)[None], RotaryLayout("prefix", 64))
    )
    assert inputs.query.shape == (37, 1, 768) and inputs.query.dtype == torch.float32
    assert torch.equal(inputs.query[:, 0], summed_query[0])
    assert inputs.key.dtype == torch.bfloat16
    assert torch.equal(inputs.key, key[0].bfloat16())


def maps(seed=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(768, 64, generator=generator), torch.randn(768, 64, generator=generator)


def test_the_index_vectors_are_the_trained_maps_applied_as_distillation_applies_them():
    query_projection, key_projection = maps()
    projection = KeyAndQueryProjection(query_projection, key_projection, "cpu")
    q, k = heads(53)
    inputs = index_space.key_and_query(q, k, GEOMETRY)
    summed_query, key = regroup_partial(
        *position_free(q.transpose(0, 1)[None], k.transpose(0, 1)[None], RotaryLayout("prefix", 64))
    )
    # The trainer's `normalize(input @ projection)` in fp32, against the server's bf16 products
    trained_queries = F.normalize(summed_query[0] @ query_projection, dim=-1)
    trained_keys = F.normalize(key[0].bfloat16().float() @ key_projection, dim=-1)
    assert torch.allclose(projection.index_queries(inputs.query), trained_queries, atol=2e-2)
    assert torch.allclose(projection.index_keys(inputs.key).float(), trained_keys, atol=2e-2)


def test_the_pool_keys_are_every_heads_position_free_dims_read_out_of_its_slot():
    q, k = heads(40)
    pool = torch.zeros(64, 4, 256, dtype=torch.bfloat16)
    slots = torch.randperm(64)[:40]
    # Attention caches the rotated key - the index reads past the prefix
    pool[slots] = rotate_prefix(k)
    projection = KeyAndQueryProjection(*maps(), "cpu")
    keys = PoolKeys(lambda: pool, projection, key_slot=GEOMETRY.key_slot)
    expected = projection.index_keys(index_space.key_and_query(q, k, GEOMETRY).key)
    assert torch.equal(keys.read(slots), expected)


def test_writing_the_key_input_leaves_the_rotated_prefix_and_other_slots_alone():
    _, k = heads(10)
    pool = torch.full((16, 4, 256), 7.0, dtype=torch.bfloat16)
    slots = torch.arange(3, 13)
    GEOMETRY.key_slot.write(pool, slots, index_space.key_and_query(heads(10)[0], k, GEOMETRY).key)
    assert torch.equal(pool[slots, :, 64:], k[..., 64:])
    assert bool((pool[slots, :, :64] == 7).all())
    assert bool((pool[:3] == 7).all()) and bool((pool[13:] == 7).all())


def test_a_pool_read_whose_dims_are_not_the_maps_is_refused():
    projection = KeyAndQueryProjection(*maps(), "cpu")
    with pytest.raises(ValueError, match="768"):
        PoolKeys(lambda: None, projection, key_slot=KeySlot(heads=4, head_dim=256, offset=0, width=256))
    with pytest.raises(ValueError, match="does not fit"):
        KeySlot(heads=4, head_dim=256, offset=64, width=256)


def test_a_rank_holds_the_heads_the_models_own_split_deals_it():
    # Every rank holds every head with one GPU, or under decode context parallel across the group, which gathers the queries
    assert index_space.head_shard(GEOMETRY, 1, 0) is None
    assert index_space.head_shard(GEOMETRY, 8, 3, 8) is None
    assert index_space.head_shard(GEOMETRY, 4, 1, 4) is None
    # Tensor parallel 8 - one key head a rank, replicated on two, and 3 of its group's 6 query heads
    eight = [index_space.head_shard(GEOMETRY, 8, rank) for rank in range(8)]
    assert [shard.kv_first for shard in eight] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [shard.counts_key for shard in eight] == [True, False] * 4
    assert {(shard.kv_count, shard.query_count) for shard in eight} == {(1, 3)}
    assert eight[5].dims(GEOMETRY) == (384, 576)
    # Tensor parallel 4 - one key head and a whole group of 6 queries, no replica
    four = [index_space.head_shard(GEOMETRY, 4, rank) for rank in range(4)]
    assert [shard.kv_first for shard in four] == [0, 1, 2, 3]
    assert all(shard.counts_key and (shard.kv_count, shard.query_count) == (1, 6) for shard in four)
    # Tensor parallel 2 - two key heads and two groups
    two = [index_space.head_shard(GEOMETRY, 2, rank) for rank in range(2)]
    assert [(shard.kv_first, shard.kv_count, shard.query_count) for shard in two] == [(0, 2, 12), (2, 2, 12)]
    assert two[1].dims(GEOMETRY) == (384, 768)


def test_every_key_head_is_counted_once_across_the_group_and_every_query_head_is_held_once():
    for tp_size in (2, 4, 8):
        shards = [index_space.head_shard(GEOMETRY, tp_size, rank) for rank in range(tp_size)]
        counted = [
            h
            for shard in shards
            if shard.counts_key
            for h in range(shard.kv_first, shard.kv_first + shard.kv_count)
        ]
        assert sorted(counted) == [0, 1, 2, 3]
        assert sum(shard.query_count for shard in shards) == 24
        # A rank's query heads are the ones its key heads serve
        for rank, shard in enumerate(shards):
            first_query = rank * shard.query_count
            groups = range(first_query // 6, (first_query + shard.query_count - 1) // 6 + 1)
            assert list(groups) == list(range(shard.kv_first, shard.kv_first + shard.kv_count))


def test_a_mesh_the_heads_do_not_divide_over_is_refused():
    # 24 query heads over 16 ranks, 5 ranks over 4 key heads, 6 ranks over 4, and 3 ranks over 4
    with pytest.raises(NotImplementedError, match="splits 24 query heads unevenly"):
        index_space.head_shard(GEOMETRY, 16, 0)
    with pytest.raises(NotImplementedError, match="neither divides 4 key-value heads"):
        index_space.head_shard(GEOMETRY, 6, 0)
    with pytest.raises(NotImplementedError, match="neither divides 4 key-value heads"):
        index_space.head_shard(GEOMETRY, 3, 0)
    # Decode context parallel over part of the group splits the pool and the heads at once
    with pytest.raises(NotImplementedError, match="decode context parallel 4 under tensor parallel 8"):
        index_space.head_shard(GEOMETRY, 8, 0, 4)
    with pytest.raises(NotImplementedError, match="--dcp-size 1 or equal to --tp"):
        index_space.head_shard(GEOMETRY, 4, 0, 2)


def test_a_rank_holding_every_head_reads_the_input_as_it_always_did():
    q, k = heads(29)
    whole = index_space.key_and_query(q.flatten(1), k, GEOMETRY)
    for shard in (None, index_space.HeadShard(0, 4, 24, True)):
        again = index_space.key_and_query(q.flatten(1), k, GEOMETRY, shard)
        assert torch.equal(again.key, whole.key) and torch.equal(again.query, whole.query)


class Fabric:
    """N simulated tensor-parallel ranks in one process, the group's all-reduce replaced by adding every rank's term

    A first pass over the ranks only records each one's term and hands it back. Once every rank has called, the terms
    are added in rank order, and every call of the second pass returns that sum - what a collective hands each rank,
    the same bits on all of them.
    """

    def __init__(self):
        self.terms = []
        self.total = None

    def reduce(self, term):
        if self.total is not None:
            return self.total
        self.terms.append(term)
        return term

    def close(self):
        self.total = self.terms[0]
        for term in self.terms[1:]:
            self.total = self.total + term


def rank_inputs(q, k, tp_size):
    """What each rank of a tensor-parallel group is handed - its own query and key heads, and the dims they are"""
    ranks = []
    for rank in range(tp_size):
        shard = index_space.head_shard(GEOMETRY, tp_size, rank)
        first = rank * shard.query_count
        local_q = q[:, first : first + shard.query_count].flatten(1)
        local_k = k[:, shard.kv_first : shard.kv_first + shard.kv_count]
        ranks.append((shard, index_space.key_and_query(local_q, local_k, GEOMETRY, shard)))
    return ranks


def split_of(shard, fabric):
    return KeySplit(*shard.dims(GEOMETRY), shard.counts_key, fabric.reduce)


def sharded_index(tp_size, q, k, projection_maps):
    """Every rank's index queries and index keys from its own heads, through the fake all-reduce"""
    fabric = Fabric()
    ranks = rank_inputs(q, k, tp_size)
    projections = [
        KeyAndQueryProjection(*projection_maps, "cpu", split=split_of(shard, fabric)) for shard, _ in ranks
    ]
    for _ in range(2):
        index = [
            p.split_index(inputs.key, inputs.query) for p, (_, inputs) in zip(projections, ranks, strict=True)
        ]
        fabric.close()
    return index, fabric


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_sharded_index_keys_and_queries_are_the_unsharded_ones(tp_size):
    q, k = heads(41, seed=3)
    projection_maps = maps(5)
    whole = KeyAndQueryProjection(*projection_maps, "cpu")
    inputs = index_space.key_and_query(q.flatten(1), k, GEOMETRY)
    index, _ = sharded_index(tp_size, q, k, projection_maps)
    # The fp32 products of the same bf16 weights over the whole input - what a sum of partials is
    query_projection, key_projection = (weight.bfloat16().float() for weight in projection_maps)
    exact_queries = F.normalize(inputs.query[:, 0] @ query_projection, dim=-1)
    exact_keys = F.normalize(inputs.key.float() @ key_projection, dim=-1)
    for index_queries, index_keys in index:
        assert index_queries.dtype == torch.float32 and index_keys.dtype == torch.bfloat16
        assert torch.allclose(index_queries, exact_queries, atol=1e-5)
        assert torch.allclose(index_keys.float(), exact_keys, atol=1e-2)
        # And the unsharded server's bf16 products, which round where the partials do not
        assert torch.allclose(index_queries, whole.index_queries(inputs.query), atol=2e-2)
        assert torch.allclose(index_keys.float(), whole.index_keys(inputs.key).float(), atol=2e-2)
    # Every rank holds the same one
    for index_queries, index_keys in index[1:]:
        assert torch.equal(index_queries, index[0][0]) and torch.equal(index_keys, index[0][1])


@pytest.mark.parametrize("tp_size", [4, 8])
def test_a_key_head_replicated_on_two_ranks_is_added_to_the_sum_once(tp_size):
    q, k = heads(11, seed=4)
    projection_maps = maps(6)
    _, fabric = sharded_index(tp_size, q, k, projection_maps)
    # Each rank's term is its index-dim queries then its index-dim keys. The replica that does not count adds zeros
    key_terms = [term[:, 64:] for term in fabric.terms]
    zero = [bool((term == 0).all()) for term in key_terms]
    assert zero == ([False, True] * 4 if tp_size == 8 else [False] * 4)
    # Adding them counts each head once - a replica counted twice would double that head's rows
    full = index_space.key_and_query(q.flatten(1), k, GEOMETRY)
    query_projection, key_projection = (weight.bfloat16().float() for weight in projection_maps)
    assert torch.allclose(sum(key_terms), full.key.float() @ key_projection, atol=1e-3, rtol=1e-4)
    # Every rank's queries are added, whichever replica holds them
    assert torch.allclose(
        sum(term[:, :64] for term in fabric.terms), full.query[:, 0] @ query_projection, atol=1e-3, rtol=1e-4
    )


def test_a_projection_with_no_split_cannot_split_an_index():
    projection = KeyAndQueryProjection(*maps(), "cpu")
    q, k = heads(2)
    inputs = index_space.key_and_query(q.flatten(1), k, GEOMETRY)
    with pytest.raises(ValueError, match="KeySplit"):
        projection.split_index(inputs.key, inputs.query)


def test_the_selection_the_sharded_index_makes_is_the_unsharded_one():
    # Keys along the query's direction stand clear of noise, so a top-k is the same through bf16 rounding.
    # One map for both sides makes a key along a query's direction an index key along its index query
    key_count, planted_count = 600, 20
    generator = torch.Generator().manual_seed(11)
    _, k = heads(key_count, seed=12)
    query = torch.randn(1, 24, 256, generator=generator).bfloat16().expand(key_count, 24, 256)
    planted = torch.randperm(key_count, generator=generator)[:planted_count]
    direction = query[0].view(4, 6, 256).float().sum(1)
    k[planted] = (4 * direction).bfloat16()
    shared = torch.randn(768, 64, generator=generator)
    whole = KeyAndQueryProjection(shared, shared, "cpu")
    full = index_space.key_and_query(query.flatten(1), k, GEOMETRY)
    scores = whole.index_keys(full.key).float() @ whole.index_queries(full.query)[0]
    wanted = sorted(scores.topk(planted_count).indices.tolist())
    assert wanted == sorted(planted.tolist())
    index, _ = sharded_index(8, query, k, (shared, shared))
    for index_queries, index_keys in index:
        picked = index_keys.float() @ index_queries[0]
        assert sorted(picked.topk(planted_count).indices.tolist()) == wanted


@pytest.fixture
def key_and_query_design():
    return DesignSettings.parse({"projections": "maps.pt"})


def test_a_split_layer_writes_each_new_tokens_index_key_at_its_kv_slot_on_every_rank(key_and_query_design):
    tp_size, token_count, slot_count = 8, 24, 64
    table = torch.zeros(1, 64, dtype=torch.int32)
    q, k = heads(token_count, seed=8)
    slots = torch.randperm(slot_count - 8)[:token_count]
    fabric = Fabric()
    ranks = rank_inputs(q, k, tp_size)
    layers = [
        LayerIndex(
            key_and_query_design,
            maps(7),
            slot_count,
            slot_count - 8,
            lambda: table,
            64,
            512,
            "cpu",
            split=split_of(shard, fabric),
        )
        for shard, _ in ranks
    ]
    for _ in range(2):
        written = [layer.write(inputs, slots) for layer, (_, inputs) in zip(layers, ranks, strict=True)]
        fabric.close()
    whole = KeyAndQueryProjection(*maps(7), "cpu")
    full = index_space.key_and_query(q.flatten(1), k, GEOMETRY)
    untouched = torch.ones(slot_count, dtype=torch.bool)
    untouched[slots] = False
    for layer, index_queries in zip(layers, written, strict=True):
        # The keys are held in a buffer of the layer's own, at the slots the tokens were given
        assert isinstance(layer.buffer, KeyBuffer)
        assert layer.index_keys.shape == (slot_count, 64) and layer.index_keys.dtype == torch.bfloat16
        assert torch.allclose(layer.buffer.read(slots).float(), whole.index_keys(full.key).float(), atol=2e-2)
        assert torch.allclose(index_queries, whole.index_queries(full.query), atol=2e-2)
        assert not bool(layer.index_keys[untouched].any())
    # Every rank stored the same keys
    for layer in layers[1:]:
        assert torch.equal(layer.index_keys, layers[0].index_keys)


def test_a_layer_holding_every_key_head_still_reads_its_keys_from_the_pool(key_and_query_design):
    table = torch.zeros(1, 64, dtype=torch.int32)
    pool = torch.zeros(64, 4, 256, dtype=torch.bfloat16)
    layer = LayerIndex(
        key_and_query_design,
        maps(7),
        64,
        56,
        lambda: table,
        64,
        512,
        "cpu",
        key_buffer=lambda: pool,
        key_slot=GEOMETRY.key_slot,
    )
    # No index keys of its own, and a write puts the key input in the pool as it did
    assert isinstance(layer.buffer, PoolKeys) and layer.projection.split is None
    assert layer.index_keys.shape == (0, 64)
    q, k = heads(5, seed=9)
    slots = torch.tensor([9, 3, 40, 41, 7])
    layer.write(index_space.key_and_query(q.flatten(1), k, GEOMETRY), slots)
    assert torch.equal(pool[slots, :, 64:], k[..., 64:])
    # A key-and-query layer is handed a pool unless the mesh splits the heads
    with pytest.raises(ValueError, match="needs the pool's key buffer"):
        LayerIndex(key_and_query_design, maps(7), 64, 56, lambda: table, 64, 512, "cpu")


def test_a_split_beside_decode_context_parallel_is_refused(key_and_query_design):
    table = torch.zeros(1, 64, dtype=torch.int32)
    shard = index_space.head_shard(GEOMETRY, 8, 0)
    with pytest.raises(ValueError, match="decode context parallel splits the slots"):
        LayerIndex(
            key_and_query_design,
            maps(7),
            64,
            56,
            lambda: table,
            64,
            512,
            "cpu",
            Shard(2, 0, lambda term: term),
            split=split_of(shard, Fabric()),
        )


class Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_idle(self):
        return self.kind == "idle"

    def is_extend(self):
        return self.kind == "extend"

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, q_rope=None, **kwargs):
    """Stand in for the backend's own forward, recording that it ran"""
    self.dense.append(layer.layer_id)
    return "dense"


def test_a_layer_outside_the_design_and_an_idle_batch_reach_the_backend_unchanged():
    armed = SimpleNamespace(dense=[], **{backend.INDEXERS: {3: object()}})
    linear = SimpleNamespace(layer_id=2)
    batch = SimpleNamespace(forward_mode=Mode("extend"))
    assert backend.select_and_read(forward_extend, armed, "q", "k", "v", linear, batch) == "dense"
    quadratic = SimpleNamespace(layer_id=3)
    idle = SimpleNamespace(forward_mode=Mode("idle"))
    assert backend.select_and_read(forward_extend, armed, "q", "k", "v", quadratic, idle) == "dense"
    # A backend that was never armed - another model's - reads everything densely
    plain = SimpleNamespace(dense=[])
    assert backend.select_and_read(forward_extend, plain, "q", "k", "v", quadratic, batch) == "dense"
    assert armed.dense == [2, 3] and plain.dense == [3]


def test_another_models_backend_and_a_draft_are_left_unarmed(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "ARMED", {})
    glm = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=GLM))
    plain = SimpleNamespace()
    backend.arm(None, plain, glm)
    assert not hasattr(plain, backend.INDEXERS)
    qwen = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=QWEN38["text_config"]))
    draft = SimpleNamespace(is_draft_runner=True)
    backend.arm(None, draft, qwen)
    assert not hasattr(draft, backend.INDEXERS)


def test_an_unarmed_backend_maintains_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(registry, "maintain", called.append)
    backend.maintain(SimpleNamespace(), "batch")
    backend.maintain_out_graph(SimpleNamespace(), "batch", False)
    armed = SimpleNamespace(**{backend.INDEXERS: {3: object()}})
    backend.maintain(armed, "batch")
    # A capture records device work - maintenance is host work, done before every replay
    backend.maintain_out_graph(armed, "capture", True)
    backend.maintain_out_graph(armed, "replay", in_capture=False)
    assert called == ["batch", "replay"]


def fake_flash_attention(
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k_new, **kwargs
):
    """FlashAttention's paged read at a page size of 1, each query its own sequence of `cache_seqlens` slots

    With `return_softmax_lse` it also returns the log-sum-exp, [heads, tokens] as the varlen kernel lays it out.
    """
    groups = q.shape[1] // k_cache.shape[2]
    out, lse = [], []
    for row in range(q.shape[0]):
        assert int(cu_seqlens_q[row + 1] - cu_seqlens_q[row]) == 1
        assert int(cu_seqlens_k_new[row + 1] - cu_seqlens_k_new[row]) == int(cache_seqlens[row])
        slots = page_table[row, : int(cache_seqlens[row])].long()
        keys = k_cache[slots, 0].repeat_interleave(groups, dim=1).transpose(0, 1).float()
        values = v_cache[slots, 0].repeat_interleave(groups, dim=1).transpose(0, 1).float()
        logits = q[row, :, None].float() @ keys.transpose(1, 2) * kwargs["softmax_scale"]
        out.append(logits.softmax(-1) @ values)
        lse.append(logits.logsumexp(-1))
    out = torch.stack(out)[:, :, 0].to(q.dtype)
    if kwargs.get("return_softmax_lse"):
        return out, torch.stack(lse)[:, :, 0].transpose(0, 1)
    return out


def test_the_read_attends_each_query_over_exactly_the_slots_its_list_names():
    generator = torch.Generator().manual_seed(4)
    keys = torch.randn(32, 1, 4, 256, generator=generator).bfloat16()
    values = torch.randn(32, 1, 4, 256, generator=generator).bfloat16()
    attention = SimpleNamespace(
        flash_attn_with_kvcache=fake_flash_attention,
        get_paged_mha_kv_cache=lambda layer: (keys, values),
        num_splits=0,
    )
    layer = SimpleNamespace(tp_q_head_num=24, head_dim=256, scaling=256**-0.5, logit_cap=0.0)
    q = torch.randn(3, 24 * 256, generator=generator).bfloat16()
    lists = torch.tensor([[5, 9, 2, -1, -1], [0, -1, -1, -1, -1], [31, 30, 7, 8, 1]], dtype=torch.int32)
    out = backend.sparse_read(attention, q, layer, lists)
    assert out.shape == (3, 24 * 256)
    for row in range(3):
        slots = lists[row][lists[row] >= 0].long()
        # The torch reference - each query head over its key head's selected keys and values
        k = keys[slots, 0].repeat_interleave(6, dim=1).transpose(0, 1).float()
        v = values[slots, 0].repeat_interleave(6, dim=1).transpose(0, 1).float()
        expected = F.scaled_dot_product_attention(q[row].view(24, 1, 256).float(), k, v, scale=256**-0.5)
        assert torch.allclose(out[row].view(24, 256).float(), expected[:, 0], atol=2e-2)


def test_a_design_on_the_attention_input_is_refused_for_a_dense_model(monkeypatch):
    design = DesignSettings.parse({"input": "attention_input", "projections": "maps.pt"})
    monkeypatch.setattr(backend, "REQUIRED", ())
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = runtime.get_spec = None
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    with pytest.raises(ValueError, match="reads the key and query"):
        backend.refuse_unservable(SimpleNamespace(), GEOMETRY, design)


class Pool:
    """A KV pool a fake backend reads through `get_kv_buffer`, flat [rows, Hkv, dh] as sglang holds it"""

    def __init__(self, keys, values):
        self.keys, self.values = keys, values

    def get_kv_buffer(self, layer_id):
        return self.keys, self.values

    def get_key_buffer(self, layer_id):
        return self.keys

    def get_value_buffer(self, layer_id):
        return self.values


def merge_reference(partials):
    """The decode group's merge, every rank's partial weighed by its share of the whole list's softmax"""
    lses = torch.stack([lse for _, lse in partials])
    total = lses.logsumexp(0)
    weights = torch.nan_to_num(torch.exp(lses - total), nan=0.0)
    return sum(weight[..., None] * out for weight, (out, _) in zip(weights, partials, strict=True))


def test_a_decode_group_s_partial_reads_merge_into_the_read_of_the_whole_list():
    size, slot_count = 8, 64
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(slot_count, 4, 256, generator=generator).bfloat16()
    values = torch.randn(slot_count, 4, 256, generator=generator).bfloat16()
    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=24,
        tp_k_head_num=4,
        tp_v_head_num=4,
        head_dim=256,
        v_head_dim=256,
        scaling=256**-0.5,
        logit_cap=0.0,
    )
    q = torch.randn(4, 24, 256, generator=generator).bfloat16()
    # Spread over every rank, over two ranks alone, one key, and slots past a rank's last one
    lists = torch.tensor(
        [
            [5, 9, 2, 63, 40, 17, 33, 0, -1, -1],
            [8, 16, 24, 1, -1, -1, -1, -1, -1, -1],
            [61, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 15, 23, 31, 39, 47, 55, 62, 6, 14],
        ],
        dtype=torch.int32,
    )
    whole = SimpleNamespace(
        flash_attn_with_kvcache=fake_flash_attention,
        get_paged_mha_kv_cache=lambda layer: (keys[:, None], values[:, None]),
        num_splits=0,
    )
    expected = backend.sparse_read(whole, q.view(4, -1), layer, lists).view(4, 24, 256).float()
    # Each rank's pool - slot s at row s // size, for the slots it owns
    ranks = [
        SimpleNamespace(
            flash_attn_with_kvcache=fake_flash_attention,
            token_to_kv_pool=Pool(keys[rank::size], values[rank::size]),
        )
        for rank in range(size)
    ]
    partials = []
    for rank, rank_backend in enumerate(ranks):
        shard = Shard(size, rank, lambda term: term)
        rows, counts = backend.owned_lists(lists, shard)
        owned = lists[(lists >= 0) & (lists % size == rank)]
        assert int(counts.sum()) == len(owned)
        for row in range(len(lists)):
            assert sorted(rows[row, : counts[row]].tolist()) == sorted(
                (lists[row][(lists[row] >= 0) & (lists[row] % size == rank)] // size).tolist()
            )
            assert (rows[row, counts[row] :] == -1).all()
        partials.append(backend.partial_read(rank_backend, q, layer, rows, counts))
    # A rank holding none of a list adds nothing to its softmax
    assert torch.isneginf(partials[0][1][2]).all()
    merged = merge_reference(partials)
    assert torch.allclose(merged, expected, atol=2e-2)
    # Each rank keeps its own three heads of the merge
    for rank, rank_backend in enumerate(ranks):
        heads = slice(3 * rank, 3 * rank + 3)
        out = backend.group_read(
            rank_backend,
            q,
            layer,
            lists,
            Shard(size, rank, lambda term: term),
            merge=lambda out, lse, heads=heads: merged[:, heads],
        )
        assert torch.equal(out, merged[:, heads].to(q.dtype).reshape(4, -1))


class Group:
    """What `get_parallel` answers for a mesh, under a fake sglang"""

    def __init__(self, tp, dcp, prefill_cp=False, dp=1):
        self.attn_tp_size, self.attn_dcp_size, self.dcp_enabled = tp, dcp, dcp > 1
        self.attn_dcp_rank = 0
        self.attn_tp_rank = 0
        self.attn_dp_size = dp
        self.attn_cp_size = tp if prefill_cp else 1
        self.enable_prefill_cp = prefill_cp


def test_a_rank_holds_a_key_head_replicated_once_the_mesh_outgrows_the_heads():
    assert [index_space.kv_heads_held(GEOMETRY, tp) for tp in (1, 2, 4, 8)] == [4, 2, 1, 1]


def test_tensor_parallel_and_decode_context_parallel_across_the_group_are_served_and_other_meshes_are_not(
    monkeypatch,
):
    design = DesignSettings.parse({"projections": "maps.pt"})
    monkeypatch.setattr(backend, "REQUIRED", ())
    monkeypatch.setattr(backend, "REQUIRED_DECODE_GROUP", ())
    monkeypatch.setattr(backend.plugin, "refuse_metadata_glue", lambda: None)
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_spec = lambda: SimpleNamespace(speculative_algorithm=None)
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    attention = SimpleNamespace(page_size=1, kv_index_translator=SimpleNamespace(is_translating=False))
    # Every rank holds every key head with one GPU and under decode context parallel across the group
    for tp, dcp in ((1, 1), (8, 8), (4, 4)):
        runtime.get_parallel = lambda tp=tp, dcp=dcp: Group(tp, dcp)
        assert backend.refuse_unservable(attention, GEOMETRY, design) is None
    # Plain tensor parallel splits them, and a rank is told which it holds
    for tp in (8, 4, 2):
        runtime.get_parallel = lambda tp=tp: Group(tp, 1)
        assert backend.refuse_unservable(attention, GEOMETRY, design) == index_space.head_shard(
            GEOMETRY, tp, 0
        )
    for tp, dcp, match in (
        (8, 4, "decode context parallel 4 under tensor parallel 8"),
        (16, 1, "splits 24 query heads unevenly"),
        (6, 1, "neither divides 4 key-value heads"),
    ):
        runtime.get_parallel = lambda tp=tp, dcp=dcp: Group(tp, dcp)
        with pytest.raises(NotImplementedError, match=match):
            backend.refuse_unservable(attention, GEOMETRY, design)
    runtime.get_parallel = lambda: Group(8, 1, dp=2)
    with pytest.raises(NotImplementedError, match="data-parallel attention"):
        backend.refuse_unservable(attention, GEOMETRY, design)
    runtime.get_parallel = lambda: Group(8, 8, prefill_cp=True)
    with pytest.raises(NotImplementedError, match="prefill context parallel"):
        backend.refuse_unservable(attention, GEOMETRY, design)


@pytest.mark.gpu
def test_the_triton_read_attends_over_exactly_the_slots_a_list_names():
    pytest.importorskip("sglang")
    torch.manual_seed(0)
    slots, kv_heads, q_heads, head_dim, width = 4096, 4, 24, 256, 96
    keys = torch.randn(slots, kv_heads, head_dim, device="cuda").bfloat16()
    values = torch.randn(slots, kv_heads, head_dim, device="cuda").bfloat16()
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: keys, get_value_buffer=lambda layer_id: values)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=q_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=head_dim**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    for token_count in (8, 5000):
        lengths = torch.randint(1, width + 1, (token_count,), device="cuda")
        lists = torch.stack([torch.randperm(slots, device="cuda")[:width] for _ in range(token_count)]).int()
        lists[torch.arange(width, device="cuda")[None, :] >= lengths[:, None]] = -1
        q = torch.randn(token_count, q_heads * head_dim, device="cuda").bfloat16()
        out = backend.triton_sparse_read(SimpleNamespace(token_to_kv_pool=pool), q, layer, lists)
        group = q_heads // kv_heads
        for row in (0, token_count // 2, token_count - 1):
            picked = lists[row, : lengths[row]].long()
            k = keys[picked].float().repeat_interleave(group, dim=1)
            v = values[picked].float().repeat_interleave(group, dim=1)
            weights_ = torch.softmax(
                torch.einsum("hd,khd->hk", q[row].view(q_heads, head_dim).float(), k) * layer.scaling, -1
            )
            reference = torch.einsum("hk,khd->hd", weights_, v).reshape(-1)
            torch.testing.assert_close(out[row].float(), reference, atol=2e-2, rtol=2e-2)


def test_the_triton_read_indexes_slots_in_int64_once_the_pool_passes_2_to_the_31_elements():
    """An int32 slot times its row stride wraps past 2**31 - every 10M context of the one-card harness read out of bounds"""
    below = torch.empty(2**21 - 1, 4, 256, device="meta")
    at = torch.empty(2**21, 4, 256, device="meta")
    assert backend.slot_index_dtype(below, below) == torch.int32
    assert backend.slot_index_dtype(at, below) == torch.int64
    assert backend.slot_index_dtype(below, at) == torch.int64


@pytest.mark.gpu
def test_the_triton_read_reads_the_last_slots_of_a_pool_past_2_to_the_31_elements():
    pytest.importorskip("sglang")
    torch.manual_seed(0)
    kv_heads, q_heads, head_dim, width, token_count = 4, 24, 256, 64, 8
    slots = 2**21 + 4096
    keys = torch.randn(slots, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn(slots, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: keys, get_value_buffer=lambda layer_id: values)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=q_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=head_dim**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    # Every list names slots past 2**31 elements of the pool
    lists = torch.stack(
        [slots - 1 - torch.randperm(4096, device="cuda")[:width] for _ in range(token_count)]
    ).int()
    q = torch.randn(token_count, q_heads * head_dim, device="cuda").bfloat16()
    out = backend.triton_sparse_read(SimpleNamespace(token_to_kv_pool=pool), q, layer, lists)
    group = q_heads // kv_heads
    for row in range(token_count):
        picked = lists[row].long()
        k = keys[picked].float().repeat_interleave(group, dim=1)
        v = values[picked].float().repeat_interleave(group, dim=1)
        weights_ = torch.softmax(
            torch.einsum("hd,khd->hk", q[row].view(q_heads, head_dim).float(), k) * layer.scaling, -1
        )
        reference = torch.einsum("hk,khd->hd", weights_, v).reshape(-1)
        torch.testing.assert_close(out[row].float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
def test_a_decode_group_s_triton_partial_reads_merge_into_the_read_of_the_whole_list(monkeypatch):
    """Off Hopper every rank reads its share through the Triton kernel, its log-sum-exp weighing the merge"""
    pytest.importorskip("sglang")
    monkeypatch.setattr(backend, "flash_reads", lambda device: False)
    torch.manual_seed(0)
    size, slot_count, kv_heads, q_heads, head_dim, width, token_count = 8, 4096, 1, 24, 256, 512, 16
    keys = torch.randn(slot_count, kv_heads, head_dim, device="cuda").bfloat16()
    values = torch.randn(slot_count, kv_heads, head_dim, device="cuda").bfloat16()
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=head_dim**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    q = torch.randn(token_count, q_heads, head_dim, device="cuda").bfloat16()
    lengths = torch.randint(1, width + 1, (token_count,), device="cuda")
    lists = torch.stack([torch.randperm(slot_count, device="cuda")[:width] for _ in range(token_count)]).int()
    lists[torch.arange(width, device="cuda")[None, :] >= lengths[:, None]] = -1
    # Row 0 lives on rank 0 alone, so every other rank holds nothing of it
    lists[0] = -1
    lists[0, :3] = torch.tensor([0, 8, 16], device="cuda")
    partials = []
    for rank in range(size):
        rank_backend = SimpleNamespace(token_to_kv_pool=Pool(keys[rank::size], values[rank::size]))
        rows, counts = backend.owned_lists(lists, Shard(size, rank, lambda term: term))
        partials.append(backend.partial_read(rank_backend, q, layer, rows, counts))
    assert torch.isneginf(partials[1][1][0]).all()
    merged = merge_reference(partials)
    group = q_heads // kv_heads
    for row in range(token_count):
        picked = lists[row][lists[row] >= 0].long()
        k = keys[picked].float().repeat_interleave(group, dim=1)
        v = values[picked].float().repeat_interleave(group, dim=1)
        weights_ = torch.softmax(torch.einsum("hd,khd->hk", q[row].float(), k) * layer.scaling, -1)
        reference = torch.einsum("hk,khd->hd", weights_, v)
        torch.testing.assert_close(merged[row], reference, atol=2e-2, rtol=2e-2)
