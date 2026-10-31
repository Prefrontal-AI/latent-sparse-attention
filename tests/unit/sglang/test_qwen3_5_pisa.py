"""Qwen3.5's quadratic layers under PISA, behind stand-ins for the backend - a list a key head, read by that head's queries.

What is worth testing without sglang or a GPU: that each key head selects what the reference selects for it from the keys
and queries in the pool, that each query head reads its own key head's list and nothing else, and that a mesh splitting a
group of heads is refused.
"""

import dataclasses
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.baselines import pisa as reference
from lsa.sglang.index import pisa as pisa_index
from lsa.sglang.index import registry
from lsa.sglang.models.qwen3_5 import backend, index_space
from lsa.sglang.models.qwen3_5 import pisa as grouped_query_pisa
from lsa.sglang.settings.debug import PisaCheck
from lsa.sglang.settings.design import PisaConfig
from tests.unit.sglang.test_qwen3_5 import GEOMETRY, GLM, QWEN38, Group, Mode, fake_flash_attention

CONFIG = PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5)
LOGIT_SCALE = 256**-0.5
LAYER = 3


class Backend:
    """What the indexer reads off a FlashAttention backend - the request table and the layer's KV pool"""

    def __init__(self, kv_head_count=4, request_count=2, context=64, slot_count=160, seed=0):
        generator = torch.Generator().manual_seed(seed)
        slots = torch.randperm(slot_count, generator=generator)[: request_count * context]
        self.req_to_token = slots.view(request_count, context).int()
        self.keys = torch.randn(slot_count, kv_head_count, 256, generator=generator).bfloat16()
        self.values = torch.randn(slot_count, kv_head_count, 256, generator=generator).bfloat16()
        self.token_to_kv_pool = SimpleNamespace(
            size=slot_count,
            get_key_buffer=lambda layer: self.keys,
            get_kv_buffer=lambda layer: (self.keys, self.values),
        )
        self.flash_attn_with_kvcache = fake_flash_attention
        self.num_splits = 0


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(registry, "INDEXERS", [])
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(grouped_query_pisa.memory, "check_built", lambda *args: None)
    monkeypatch.setattr(pisa_index, "TALLY", pisa_index.CheckTally())
    attention = Backend()
    indexer = grouped_query_pisa.PisaGroupedQueryIndexer(
        LAYER, GEOMETRY, CONFIG, PisaCheck(), attention, kv_head_count=4
    )
    return attention, indexer


def batch(extend, rows, positions, seq_lens, lengths=None):
    return SimpleNamespace(
        forward_mode=Mode("extend" if extend else "decode"),
        lsa_rows=rows,
        req_pool_indices=torch.tensor(rows),
        rids=[f"r{row}" for row in rows],
        positions=torch.tensor(positions),
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens_cpu=lengths,
    )


def reference_sets(attention, row, key_count, queries, positions):
    """The reference's lists, a set of positions per key head, over the request's keys in the pool"""
    keys = attention.keys[attention.req_to_token[row, :key_count].long()]
    selection = reference.select(
        keys.float(),
        queries.float(),
        torch.tensor(positions),
        CONFIG.block_size,
        CONFIG.keep_count,
        CONFIG.branching,
        CONFIG.beam_width,
        LOGIT_SCALE,
        CONFIG.forced_blocks,
    ).selected_key_ids
    return [[set(head[head >= 0].tolist()) for head in line] for line in selection]


def served_sets(attention, row, lists):
    """A served list's KV slots back to positions in the request, a set per key head"""
    inverse = {int(slot): position for position, slot in enumerate(attention.req_to_token[row].tolist())}
    return [[{inverse[int(slot)] for slot in head if slot >= 0} for head in line] for line in lists]


def queries_for(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 24, 256, generator=generator).bfloat16()


def test_each_key_head_selects_what_the_reference_selects_for_it_from_its_own_group(armed):
    attention, indexer = armed
    # A prefill of two requests, then a decode step of each
    queries = queries_for(48 + 40, seed=1)
    prefill = batch(True, [0, 1], list(range(48)) + list(range(40)), [48, 40], [48, 40])
    indexer.maintain(prefill)
    lists = indexer.select(queries.view(88, -1), prefill, LOGIT_SCALE)
    assert lists.shape == (88, 4, CONFIG.width) and lists.dtype == torch.int32
    assert served_sets(attention, 0, lists[:48]) == reference_sets(attention, 0, 48, queries[:48], range(48))
    assert served_sets(attention, 1, lists[48:]) == reference_sets(attention, 1, 40, queries[48:], range(40))
    step = queries_for(2, seed=2)
    decode = batch(False, [0, 1], [48, 40], [49, 41])
    indexer.maintain(decode)
    lists = indexer.select(step.view(2, -1), decode, LOGIT_SCALE)
    assert served_sets(attention, 0, lists[:1]) == reference_sets(attention, 0, 49, step[:1], [48])
    assert served_sets(attention, 1, lists[1:]) == reference_sets(attention, 1, 41, step[1:], [40])


def test_the_key_heads_select_apart_where_their_keys_differ(armed):
    attention, indexer = armed
    prefill = batch(True, [0], list(range(64)), [64], [64])
    indexer.maintain(prefill)
    lists = served_sets(
        attention, 0, indexer.select(queries_for(64, seed=3).view(64, -1), prefill, LOGIT_SCALE)
    )
    assert any(len({frozenset(head) for head in line}) > 1 for line in lists)


def test_each_query_head_reads_only_its_own_key_heads_list():
    attention = Backend(request_count=1, context=32, slot_count=32)
    layer = SimpleNamespace(
        layer_id=LAYER,
        tp_q_head_num=24,
        tp_k_head_num=4,
        tp_v_head_num=4,
        head_dim=256,
        v_head_dim=256,
        scaling=LOGIT_SCALE,
        logit_cap=0.0,
    )
    generator = torch.Generator().manual_seed(4)
    q = torch.randn(2, 24 * 256, generator=generator).bfloat16()
    lists = torch.full((2, 4, 5), -1, dtype=torch.int32)
    lists[0, 0, :3] = torch.tensor([5, 9, 2])
    lists[0, 1, :1] = torch.tensor([0])
    lists[0, 2, :5] = torch.tensor([31, 30, 7, 8, 1])
    lists[0, 3, :2] = torch.tensor([4, 5])
    lists[1, :, :4] = torch.tensor([[1, 2, 3, 4], [9, 10, 11, 12], [20, 21, 22, 23], [6, 7, 8, 9]])
    out = grouped_query_pisa.head_read(attention, q, layer, lists)
    assert out.shape == (2, 24 * 256)
    for token in range(2):
        for head in range(4):
            slots = lists[token, head][lists[token, head] >= 0].long()
            k = attention.keys[slots, head].float()
            v = attention.values[slots, head].float()
            group = q[token].view(24, 256)[head * 6 : (head + 1) * 6].float()
            expected = F.scaled_dot_product_attention(group[:, None], k[None], v[None], scale=LOGIT_SCALE)
            got = out[token].view(24, 256)[head * 6 : (head + 1) * 6].float()
            assert torch.allclose(got, expected[:, 0], atol=2e-2)


def test_a_key_heads_slot_is_its_slot_in_the_pool_read_as_one_head():
    lists = torch.tensor([[[3, -1], [3, 0]]], dtype=torch.int32)
    assert grouped_query_pisa.head_pages(lists).tolist() == [[6, 0], [7, 1]]


def fake_runtime(monkeypatch, parallel):
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_spec = lambda: SimpleNamespace(speculative_algorithm=None)
    runtime.get_parallel = lambda: parallel
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)


def test_tensor_parallel_up_to_twice_the_key_heads_is_served_and_a_split_group_or_pool_is_not(monkeypatch):
    monkeypatch.setattr(backend, "REQUIRED", ())
    monkeypatch.setattr(grouped_query_pisa.plugin, "refuse_metadata_glue", lambda: None)
    attention = SimpleNamespace(page_size=1, kv_index_translator=SimpleNamespace(is_translating=False))
    for tp, held in ((1, 4), (2, 2), (4, 1), (8, 1)):
        fake_runtime(monkeypatch, Group(tp, 1))
        assert grouped_query_pisa.refuse_unservable(attention, GEOMETRY).kv_count == held
    # Rank 5 of 8 holds the third key head, a share of its group
    parallel = Group(8, 1)
    parallel.attn_tp_rank = 5
    fake_runtime(monkeypatch, parallel)
    assert grouped_query_pisa.refuse_unservable(attention, GEOMETRY) == index_space.HeadShard(2, 1, 3, False)
    for tp in (3, 6, 12, 16):
        fake_runtime(monkeypatch, Group(tp, 1))
        with pytest.raises(NotImplementedError, match=f"tensor parallel {tp} splits 4 key heads.*or at 8"):
            grouped_query_pisa.refuse_unservable(attention, GEOMETRY)
    for parallel in (Group(8, 8), Group(4, 1, prefill_cp=True)):
        fake_runtime(monkeypatch, parallel)
        with pytest.raises(NotImplementedError, match="tensor parallel only"):
            grouped_query_pisa.refuse_unservable(attention, GEOMETRY)
    fake_runtime(monkeypatch, Group(1, 1))
    with pytest.raises(NotImplementedError, match="--page-size 1"):
        grouped_query_pisa.refuse_unservable(
            SimpleNamespace(page_size=16, kv_index_translator=attention.kv_index_translator), GEOMETRY
        )


def test_a_mesh_that_splits_the_query_heads_unevenly_is_refused():
    geometry = dataclasses.replace(GEOMETRY, query_head_count=20)
    grouped_query_pisa.refuse_mesh(geometry, 4)
    with pytest.raises(NotImplementedError, match="tensor parallel 8 splits 20 query heads unevenly"):
        grouped_query_pisa.refuse_mesh(geometry, 8)


def test_only_a_qwen_backend_is_armed_and_with_a_list_a_key_head_it_holds(monkeypatch):
    monkeypatch.setattr(grouped_query_pisa, "ARMED", {})
    monkeypatch.setattr(registry, "INDEXERS", [])
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    monkeypatch.setattr(
        grouped_query_pisa,
        "refuse_unservable",
        lambda attention, geometry: index_space.HeadShard(0, 2, 12, True),
    )
    block = SimpleNamespace(lsa={**CONFIG.body(), "debug": {"check_pisa": {"rows": 3}}})
    glm = SimpleNamespace(model_config=SimpleNamespace(hf_config=block, hf_text_config=GLM))
    plain = SimpleNamespace()
    grouped_query_pisa.arm(None, plain, glm)
    assert not hasattr(plain, backend.INDEXERS)
    qwen = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=block, hf_text_config=QWEN38["text_config"])
    )
    served = SimpleNamespace()
    grouped_query_pisa.arm(None, served, qwen)
    indexers = getattr(served, backend.INDEXERS)
    assert sorted(indexers) == list(index_space.geometry_of(QWEN38).layers)
    assert {indexer.kv_head_count for indexer in indexers.values()} == {2}
    assert {indexer.replicas for indexer in indexers.values()} == {None}
    assert all(indexer.config == CONFIG for indexer in indexers.values())
    # The check's rows come from the block's debug sub-block
    assert {indexer.check_rows for indexer in indexers.values()} == {3}


def test_a_captured_prefill_is_refused_with_the_way_out(armed, monkeypatch):
    attention, indexer = armed
    setattr(attention, backend.INDEXERS, {LAYER: indexer})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    layer = SimpleNamespace(layer_id=LAYER)
    prefill = batch(True, [0], list(range(9)), [9], [9])

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return "dense"

    with pytest.raises(RuntimeError, match="prefill CUDA graph disabled"):
        grouped_query_pisa.select_and_read(forward_extend, attention, "q", "k", "v", layer, prefill)


def test_the_debugging_check_starts_once_the_request_holds_the_keys_it_is_set_to(monkeypatch, capsys):
    monkeypatch.setattr(registry, "INDEXERS", [])
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(grouped_query_pisa.memory, "check_built", lambda *args: None)
    monkeypatch.setattr(pisa_index, "TALLY", pisa_index.CheckTally())
    attention = Backend()
    check = PisaCheck(rows=2, from_keys=40)
    indexer = grouped_query_pisa.PisaGroupedQueryIndexer(
        LAYER, GEOMETRY, CONFIG, check, attention, kv_head_count=4
    )
    short = batch(True, [0], list(range(32)), [32], [32])
    indexer.maintain(short)
    indexer.select(queries_for(32, seed=5).view(32, -1), short, LOGIT_SCALE)
    assert "pisa check" not in capsys.readouterr().out
    long = batch(True, [0], list(range(32, 48)), [48], [16])
    indexer.select(queries_for(16, seed=6).view(16, -1), long, LOGIT_SCALE)
    printed = capsys.readouterr().out.splitlines()
    rows = [line for line in printed if "pisa check" in line and "position" in line]
    assert len(rows) == 4 * 2
    # The served lists agree with the reference on every checked row and head
    for line in rows:
        held, wanted = line.split(" - ")[1].split(" of the reference's ")
        assert held == wanted.split()[0] and ", exact at gap" in line
    assert printed[-1].startswith("lsa-lsa: pisa check tally rows 8 exact 8 ") and printed[-1].endswith(
        "verdict PASS"
    )


class Ranks:
    """The attention group's collectives, run by one thread a rank"""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    def exchange(self, tensor):
        self.slots[self.local.rank] = tensor.clone()
        self.barrier.wait()
        everyone = list(self.slots)
        self.barrier.wait()
        return everyone

    def all_reduce(self, tensor):
        # Summed in a different order on each rank, as a ring does
        everyone = self.exchange(tensor)
        order = everyone[self.local.rank :] + everyone[: self.local.rank]
        return sum(order[1:], order[0])

    def gather(self, tensor):
        return torch.cat(self.exchange(tensor))

    def run(self, work):
        """Run `work(rank)` on every rank at once and return what each returned"""
        results, failures = [None] * self.size, []

        def rank_body(rank):
            self.local.rank = rank
            try:
                results[rank] = work(rank)
            except BaseException as failure:
                failures.append(failure)
                self.barrier.abort()

        threads = [threading.Thread(target=rank_body, args=(rank,)) for rank in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]
        return results


def replicas_of_rank(ranks, rank):
    """Rank `rank` of eight, holding key head `rank // 2` with another rank"""
    return grouped_query_pisa.Replicas(
        rank // 2, 4, 2, ranks.all_reduce, lambda tensor: ranks.gather(tensor).view(8, *tensor.shape)
    )


def test_the_replicas_of_a_key_head_sum_their_scores_to_the_same_bits_and_no_other_rank_adds_in():
    ranks = Ranks(8)
    scores = torch.randn(8, 5, 7, generator=torch.Generator().manual_seed(0)) * torch.logspace(-3, 3, 7)
    summed = ranks.run(lambda rank: replicas_of_rank(ranks, rank).sum_scores(scores[rank]))
    for rank in range(8):
        pair = rank // 2 * 2
        assert torch.equal(summed[rank], scores[pair] + scores[pair + 1])


def test_the_replicas_gather_their_key_heads_whole_group_in_head_order_and_compare_their_lists():
    ranks = Ranks(8)
    queries = torch.randn(8, 3, 6, 4, generator=torch.Generator().manual_seed(1))
    grouped = ranks.run(lambda rank: replicas_of_rank(ranks, rank).group_queries(queries[rank]))
    for rank in range(8):
        pair = rank // 2 * 2
        assert torch.equal(grouped[rank], torch.cat([queries[pair], queries[pair + 1]], dim=1))
    lists = torch.arange(4 * 2 * 5, dtype=torch.int32).view(4, 2, 5).repeat_interleave(2, dim=0)
    assert ranks.run(lambda rank: replicas_of_rank(ranks, rank).agree(lists[rank])) == [True] * 8
    lists[5, 1, 2] = -1
    assert ranks.run(lambda rank: replicas_of_rank(ranks, rank).agree(lists[rank])) == [False] * 8


def test_a_key_head_on_two_ranks_selects_the_list_a_rank_holding_its_whole_group_selects(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(grouped_query_pisa.memory, "check_built", lambda *args: None)
    monkeypatch.setattr(registry, "INDEXERS", [])
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    attention = Backend()
    queries = queries_for(48, seed=7)
    prefill = batch(True, [0], list(range(48)), [48], [48])
    decode = batch(False, [0], [48], [49])
    step = queries_for(1, seed=8)
    whole = grouped_query_pisa.PisaGroupedQueryIndexer(
        LAYER, GEOMETRY, CONFIG, PisaCheck(), attention, kv_head_count=4
    )
    whole.maintain(prefill)
    wanted = whole.select(queries.view(48, -1), prefill, LOGIT_SCALE)
    whole.maintain(decode)
    wanted_step = whole.select(step.view(1, -1), decode, LOGIT_SCALE)
    ranks = Ranks(8)

    def serve(rank):
        # A pool of the key head this rank holds, and the half of its group's queries
        local = Backend()
        local.keys, local.values = attention.keys[:, rank // 2 : rank // 2 + 1], attention.values[:, :1]
        local.req_to_token = attention.req_to_token
        indexer = grouped_query_pisa.PisaGroupedQueryIndexer(
            LAYER, GEOMETRY, CONFIG, PisaCheck(), local, 1, replicas_of_rank(ranks, rank)
        )
        share = slice(rank % 2 * 3, rank % 2 * 3 + 3)
        mine = queries.view(48, 4, 6, 256)[:, rank // 2, share]
        indexer.maintain(prefill)
        lists = indexer.select(mine.reshape(48, -1), prefill, LOGIT_SCALE)
        indexer.maintain(decode)
        mine_step = step.view(1, 4, 6, 256)[:, rank // 2, share]
        return lists, indexer.select(mine_step.reshape(1, -1), decode, LOGIT_SCALE)

    served = ranks.run(serve)
    for rank, (lists, step_lists) in enumerate(served):
        assert torch.equal(lists[:, 0], wanted[:, rank // 2])
        assert torch.equal(step_lists[:, 0], wanted_step[:, rank // 2])


def test_replicas_are_built_only_for_a_key_head_on_several_ranks(monkeypatch):
    monkeypatch.setattr(backend, "REQUIRED", ())
    monkeypatch.setattr(grouped_query_pisa, "REQUIRED_REPLICAS", ())
    assert grouped_query_pisa.replicas_of(GEOMETRY, index_space.HeadShard(0, 4, 24, True)) is None
    assert grouped_query_pisa.replicas_of(GEOMETRY, index_space.HeadShard(1, 2, 12, True)) is None
    assert grouped_query_pisa.replicas_of(GEOMETRY, index_space.HeadShard(3, 1, 6, True)) is None
    dp_attention = ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.attn_tp_all_reduce = lambda tensor: tensor
    dp_attention.attn_tp_all_gather_into_tensor = lambda out, tensor: out.copy_(
        tensor.repeat(8, *[1] * (tensor.dim() - 1))
    )
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.dp_attention", dp_attention)
    replicas = grouped_query_pisa.replicas_of(GEOMETRY, index_space.HeadShard(2, 1, 3, False))
    assert (replicas.head, replicas.head_count, replicas.size) == (2, 4, 2)
    assert replicas.gather(torch.ones(5, 3)).shape == (8, 5, 3)
