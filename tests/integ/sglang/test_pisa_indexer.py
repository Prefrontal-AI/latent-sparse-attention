"""PISA's indexer behind stand-ins for the engine - the handover from the attention layer to a list of KV slots, with no sglang or GPU.

What is worth testing here is what the indexer reads off the batch and does with it: that keys are written where the batch's
tokens live, that the ranks of the attention group, each scoring with its own heads, keep what the reference keeps over every
head, that a prefill and a decode step select it through the request table, and that it refuses a server it cannot serve.
The ranks run one thread each, meeting at the group's collectives.
"""

import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lsa.baselines import pisa as reference
from lsa.sglang import settings
from lsa.sglang.index import pisa as pisa_index
from lsa.sglang.index import registry
from lsa.sglang.settings.design import PisaConfig

KEY_DIM = 8
GROUP_SIZE = 2
HEADS_LOCAL = 2
LOGIT_SCALE = 0.7
CONFIG = PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5)


class FakeIndexer:
    """The checkpoint's indexer, which the model builds and whose weights the real one loads"""

    def __init__(self, **kwargs):
        self.index_topk = kwargs["config"].index_topk


class Group:
    """The attention group's collectives, run by one thread a rank"""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    def exchange(self, tensor):
        self.slots[self.local.rank] = tensor
        self.barrier.wait()
        everyone = list(self.slots)
        self.barrier.wait()
        return everyone

    def gather(self, output, input):
        output.view(self.size, *input.shape)[:] = torch.stack(self.exchange(input))

    def all_reduce(self, tensor):
        return torch.stack(self.exchange(tensor)).sum(0)

    def all_to_all_single(self, output, input):
        # Share r of every rank's input lands on rank r, in rank order
        rank, each = self.local.rank, input.shape[0] // self.size
        everyone = self.exchange(input)
        output.view(self.size, each, *input.shape[1:])[:] = torch.stack(
            [tensor.view(self.size, each, *input.shape[1:])[rank] for tensor in everyone]
        )

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


class Engine:
    """What the indexer reads off the engine - the pool, the request table, the group and the flags"""

    def __init__(self, request_count=2, context=64, slot_count=160):
        generator = torch.Generator().manual_seed(0)
        slots = torch.randperm(slot_count, generator=generator)[: request_count * context]
        self.table = slots.view(request_count, context).int()
        self.buffer = torch.zeros(slot_count, 1, KEY_DIM, dtype=torch.bfloat16)
        self.group_size = GROUP_SIZE
        self.group = Group(GROUP_SIZE)
        self.dp_attention = False
        self.graph_surface = False
        self.glue_graph = False
        self.context_parallel = False

    def modules(self):
        def gather(output, input):
            # The other ranks' heads follow this rank's, in rank order
            self.group.gather(output, input)

        def all_reduce(tensor):
            return self.group.all_reduce(tensor)

        return {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.environ": SimpleNamespace(
                envs=SimpleNamespace(
                    SGLANG_ENABLE_METADATA_GLUE_GRAPH=SimpleNamespace(get=lambda: self.glue_graph)
                )
            ),
            "sglang.srt.layers": ModuleType("sglang.srt.layers"),
            "sglang.srt.layers.attention": ModuleType("sglang.srt.layers.attention"),
            "sglang.srt.layers.attention.dsa": ModuleType("sglang.srt.layers.attention.dsa"),
            "sglang.srt.layers.attention.dsa.dsa_indexer": SimpleNamespace(Indexer=FakeIndexer),
            "sglang.srt.layers.attention.dsa.utils": SimpleNamespace(
                is_graph_dsa_split_op_surface=lambda forward_batch: self.graph_surface
            ),
            "sglang.srt.layers.dp_attention": SimpleNamespace(
                attn_tp_all_gather_into_tensor=gather,
                attn_tp_all_reduce=all_reduce,
                is_dp_attention_enabled=lambda: self.dp_attention,
            ),
            "sglang.srt.model_executor": ModuleType("sglang.srt.model_executor"),
            "sglang.srt.model_executor.forward_context": SimpleNamespace(
                get_attn_backend=lambda: SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(req_to_token=self.table)
                ),
                get_token_to_kv_pool=lambda: SimpleNamespace(
                    size=self.buffer.shape[0], get_key_buffer=lambda layer: self.buffer
                ),
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_parallel=lambda: SimpleNamespace(
                    attn_tp_rank=getattr(self.group.local, "rank", 0),
                    attn_tp_group=self.group,
                    dcp_enabled=False,
                    attn_dcp_size=1,
                    attn_dcp_rank=0,
                    attn_tp_size=self.group_size,
                    enable_prefill_cp=self.context_parallel,
                    attn_cp_size=2 if self.context_parallel else 1,
                    cp_strategy="contiguous",
                )
            ),
        }


@pytest.fixture
def engine(monkeypatch):
    fake = Engine()
    for name, module in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "lsa.sglang.models.glm5_next.pisa_indexer", raising=False)
    monkeypatch.setattr(registry, "INDEXERS", [])
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    fake.module = importlib.import_module("lsa.sglang.models.glm5_next.pisa_indexer")
    monkeypatch.setattr(fake.module.memory, "check_built", lambda *args: None)
    yield fake
    sys.modules.pop("lsa.sglang.models.glm5_next.pisa_indexer", None)


CHECKED = {"check_pisa": {"rows": 2}}


def model_config(width=CONFIG.width, debug=None):
    block = {**CONFIG.body(), **({"debug": debug} if debug else {})}
    return SimpleNamespace(
        index_topk=width, index_kpool=1, kv_lora_rank=KEY_DIM, qk_rope_head_dim=0, lsa=block
    )


def build(engine, width=CONFIG.width, layer=3, debug=None):
    return engine.module.PisaIndexer(config=model_config(width, debug), layer_id=layer)


def batch(mode_extend, rows, rids, out_cache_loc, seq_lens, lengths=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False, is_extend=lambda: mode_extend),
        lsa_rows=rows,
        req_pool_indices=torch.tensor(rows),
        rids=rids,
        out_cache_loc=out_cache_loc,
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens_cpu=lengths,
    )


def slots_of(engine, row, start, stop):
    return engine.table[row, start:stop].long()


def positions_of(engine, row, lists):
    inverse = {int(slot): position for position, slot in enumerate(engine.table[row].tolist())}
    return [{inverse[int(slot)] for slot in line if slot >= 0} for line in lists]


def reference_sets(keys, queries, positions):
    selection = reference.select(
        keys[:, None],
        queries,
        positions,
        CONFIG.block_size,
        CONFIG.keep_count,
        CONFIG.branching,
        CONFIG.beam_width,
        LOGIT_SCALE,
        CONFIG.forced_blocks,
    ).selected_key_ids[:, 0]
    return [set(row[row >= 0].tolist()) for row in selection]


def group_queries(count, seed):
    """Every rank's absorbed queries for `count` tokens - this rank's first, then the other's"""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(count, HEADS_LOCAL, KEY_DIM, generator=generator).bfloat16() for _ in range(GROUP_SIZE)
    ]


def test_a_prefill_writes_its_keys_and_every_rank_selects_what_the_reference_does_over_every_heads_query(
    engine,
):
    indexers = [build(engine) for _ in range(GROUP_SIZE)]
    keys = torch.randn(40, KEY_DIM).bfloat16()
    queries = group_queries(40, seed=1)
    forward = batch(True, [1], ["a"], slots_of(engine, 1, 0, 40), [40], [40])

    def prefill(rank):
        indexers[rank].maintain(forward)
        return indexers[rank].forward_cuda(
            None, None, torch.arange(40), forward, 3, key=keys, query=queries[rank], logit_scale=LOGIT_SCALE
        )

    lists = engine.group.run(prefill)
    assert lists[0].dtype == torch.int32 and lists[0].shape == (40, CONFIG.width)
    # Written where the batch's tokens live
    assert torch.equal(engine.buffer[slots_of(engine, 1, 0, 40), 0], keys)
    expected = reference_sets(keys.float(), torch.cat(queries, 1).float(), torch.arange(40))
    assert [positions_of(engine, 1, rank_lists) for rank_lists in lists] == [expected] * GROUP_SIZE


def test_a_prefill_of_rows_the_group_does_not_divide_shares_them_and_selects_what_the_reference_does(engine):
    indexers = [build(engine) for _ in range(GROUP_SIZE)]
    keys = torch.randn(37, KEY_DIM).bfloat16()
    queries = group_queries(37, seed=3)
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 37), [37], [37])

    def prefill(rank):
        indexers[rank].maintain(forward)
        return indexers[rank].forward_cuda(
            None, None, torch.arange(37), forward, 3, key=keys, query=queries[rank], logit_scale=LOGIT_SCALE
        )

    lists = engine.group.run(prefill)
    assert lists[0].shape == (37, CONFIG.width) and torch.equal(lists[0], lists[1])
    expected = reference_sets(keys.float(), torch.cat(queries, 1).float(), torch.arange(37))
    assert positions_of(engine, 0, lists[0]) == expected


def test_a_decode_step_after_a_chunked_prefill_selects_on_every_rank_what_the_reference_does(engine):
    indexers = [build(engine) for _ in range(GROUP_SIZE)]
    keys = torch.randn(31, KEY_DIM).bfloat16()
    queries = group_queries(31, seed=2)

    def prefill_then_decode(rank):
        indexer = indexers[rank]
        for start, stop in ((0, 20), (20, 30)):
            forward = batch(True, [0], ["a"], slots_of(engine, 0, start, stop), [stop], [stop - start])
            indexer.maintain(forward)
            indexer.forward_cuda(
                None,
                None,
                torch.arange(start, stop),
                forward,
                3,
                key=keys[start:stop],
                query=queries[rank][start:stop],
                logit_scale=LOGIT_SCALE,
            )
        forward = batch(False, [0], ["a"], slots_of(engine, 0, 30, 31), [31])
        indexer.maintain(forward)
        return indexer.forward_cuda(
            None,
            None,
            torch.tensor([30]),
            forward,
            3,
            key=keys[30:],
            query=queries[rank][30:],
            logit_scale=LOGIT_SCALE,
        )

    lists = engine.group.run(prefill_then_decode)
    expected = reference_sets(
        keys.float(), torch.cat([rank[30:] for rank in queries], 1).float(), torch.tensor([30])
    )
    assert [positions_of(engine, 0, rank_lists) for rank_lists in lists] == [expected] * GROUP_SIZE


def test_a_short_prompts_keys_are_written_and_no_list_is_made_when_none_is_asked_for(engine):
    indexer = build(engine)
    keys = torch.randn(5, KEY_DIM).bfloat16()
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 5), [5], [5])
    indexer.maintain(forward)
    assert (
        indexer.forward_cuda(
            None, None, torch.arange(5), forward, 3, return_indices=False, key=keys, query=keys[:, None]
        )
        is None
    )
    assert torch.equal(engine.buffer[slots_of(engine, 0, 0, 5), 0], keys)


def test_an_idle_batch_selects_nothing(engine):
    indexer = build(engine)
    forward = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: True))
    assert indexer.forward_cuda(None, None, torch.zeros(0), forward, 3) is None


def test_the_indexer_announces_itself_and_registers_for_maintenance(engine, capsys):
    indexer = build(engine)
    assert registry.INDEXERS == [indexer]
    assert "lsa-lsa: indexing" in capsys.readouterr().out


def test_a_list_width_that_is_not_the_designs_is_refused(engine):
    with pytest.raises(ValueError, match="list width"):
        build(engine, width=CONFIG.width + 1)


@pytest.mark.parametrize(
    ("attribute", "error", "message"),
    [
        ("glue_graph", ValueError, "metadata glue graph"),
        ("dp_attention", RuntimeError, "data-parallel|dp-attention"),
        ("context_parallel", ValueError, "tensor parallel only"),
    ],
)
def test_a_server_pisa_cannot_serve_is_refused_when_the_indexer_is_built(engine, attribute, error, message):
    setattr(engine, attribute, True)
    with pytest.raises(error, match=message):
        build(engine)


def test_a_latent_with_rope_dims_is_refused(engine):
    config = model_config()
    config.qk_rope_head_dim = 64
    with pytest.raises(ValueError, match="rope"):
        engine.module.PisaIndexer(config=config, layer_id=3)


def test_a_captured_prefill_is_refused_with_the_way_out(engine):
    indexer = build(engine)
    engine.graph_surface = True
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 5), [5], [5])
    with pytest.raises(RuntimeError, match="prefill CUDA graph"):
        indexer.forward_cuda(None, None, torch.arange(5), forward, 3)


def test_a_call_without_the_softmax_scale_is_refused(engine):
    indexer = build(engine)
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 5), [5], [5])
    keys = torch.randn(5, KEY_DIM).bfloat16()
    with pytest.raises(ValueError, match="softmax scale"):
        indexer.forward_cuda(None, None, torch.arange(5), forward, 3, key=keys, query=keys[:, None])
    with pytest.raises(ValueError, match="key latent"):
        indexer.forward_cuda(None, None, torch.arange(5), forward, 3)


def test_every_layer_checks_its_lists_and_the_tally_passes(engine, monkeypatch, capsys):
    monkeypatch.setattr(pisa_index, "TALLY", pisa_index.CheckTally())
    # One rank, so the check's gathers have no other to wait for
    engine.group_size = 1
    first, second = build(engine, layer=3, debug=CHECKED), build(engine, layer=7, debug=CHECKED)
    keys = torch.randn(30, KEY_DIM).bfloat16()
    mine = group_queries(30, seed=3)[0]
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 30), [30], [30])
    first.maintain(forward)
    for indexer in (second, first):
        indexer.forward_cuda(
            None, None, torch.arange(30), forward, 3, key=keys, query=mine, logit_scale=LOGIT_SCALE
        )
    printed = capsys.readouterr().out.splitlines()
    rows = [line for line in printed if "pisa check" in line and "position" in line]
    assert [line.split("pisa check ")[1].split(" position")[0] for line in rows] == ["layer 7"] * 2 + [
        "layer 3"
    ] * 2
    assert all(", exact at gap" in line for line in rows)
    assert printed[-1].startswith("lsa-lsa: pisa check tally rows 4 exact 4 ") and printed[-1].endswith(
        "verdict PASS"
    )


def test_the_ranks_lists_are_compared_on_every_check(engine, monkeypatch, capsys):
    monkeypatch.setattr(pisa_index, "TALLY", pisa_index.CheckTally())
    indexers = [build(engine, debug=CHECKED) for _ in range(GROUP_SIZE)]
    keys = torch.randn(30, KEY_DIM).bfloat16()
    queries = group_queries(30, seed=4)
    forward = batch(True, [0], ["a"], slots_of(engine, 0, 0, 30), [30], [30])

    def prefill(rank):
        indexers[rank].maintain(forward)
        indexers[rank].forward_cuda(
            None, None, torch.arange(30), forward, 3, key=keys, query=queries[rank], logit_scale=LOGIT_SCALE
        )

    engine.group.run(prefill)
    assert pisa_index.TALLY.rank_checks == GROUP_SIZE and pisa_index.TALLY.rank_differences == 0
    assert pisa_index.TALLY.line().endswith("verdict PASS")


def test_the_settings_the_indexer_reads_are_the_designs(engine):
    assert build(engine).config == settings.pisa_of(model_config()) == CONFIG
