"""The hooks that arm a stock sglang with the design's indexer, exercised without sglang or a GPU.

What is worth testing here is the handover: that every target the plugin patches is named, that each hook
does to its arguments what it should, that a model config with a block arms the process with the indexer and the
elision and one without registers nothing, and that the projection file is checked against the model before it loads.
The selection past that point is tested in test_selector, and the indexer's own path in test_layer_index.
"""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lsa.sglang import context_parallel, plugin, settings
from lsa.sglang.constants import Banner
from lsa.sglang.index import pisa, registry
from lsa.sglang.settings import PluginSettings
from lsa.sglang.settings.design import DECODE_CONTEXT_LIMITS

# On the attention input, whose maps are `model_dim` wide - the key and query tests set their input
DESIGN = {
    "window_size": 512,
    "sink_size": 4,
    "top_k": 2048,
    "top_p": 512,
    "index_dim": 8,
    "input": "attention_input",
}


def block_of(tmp_path, **overrides):
    """Return the settings block a model config carries."""
    return {**DESIGN, "projections": str(tmp_path / "maps.pt"), **overrides}


def config_with(block, text=None):
    """Return a model config that carries `block` on its top-level config, as an override puts it."""
    return SimpleNamespace(hf_config=SimpleNamespace(lsa=block), hf_text_config=text or SimpleNamespace())


def served(block):
    return PluginSettings.parse(block)


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Every test starts from a process nothing has armed."""
    monkeypatch.setattr(plugin, "ARMED", [])


@pytest.fixture
def unarmed(monkeypatch):
    """Leave arming to its own tests, so the config hook is seen alone."""
    monkeypatch.setattr(plugin, "arm", lambda served: None)


def write_maps(tmp_path, layers=(3, 7), model_dim=16, index_dim=8):
    maps = {}
    for layer in layers:
        maps[f"{layer}.W_Q_idx"] = torch.randn(model_dim, index_dim)
        maps[f"{layer}.W_K_idx"] = torch.randn(model_dim, index_dim)
    torch.save(maps, tmp_path / "maps.pt")


def fake_hook_registry(monkeypatch):
    registered = []
    registry_module = ModuleType("sglang.srt.plugins.hook_registry")
    registry_module.HookRegistry = SimpleNamespace(
        register=lambda target, hook, kind: registered.append((target, hook, kind)),
        apply_hooks=lambda: None,
    )
    registry_module.HookType = SimpleNamespace(
        AROUND="around", AFTER="after", BEFORE="before", REPLACE="replace"
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.plugins.hook_registry", registry_module)
    return registered


def fake_indexer_module(monkeypatch):
    indexer_module = ModuleType("lsa.sglang.models.glm5_next.indexer")
    indexer_module.LSAIndexer = type("LSAIndexer", (), {})
    monkeypatch.setitem(sys.modules, "lsa.sglang.models.glm5_next.indexer", indexer_module)


def test_a_server_that_asks_for_lsa_without_a_projection_file_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="projections"):
        plugin.arm(served(block_of(tmp_path, projections="")))


def test_a_target_that_no_longer_resolves_is_named_and_nothing_is_registered(monkeypatch, tmp_path):
    registered = fake_hook_registry(monkeypatch)

    def resolve(target):
        if target.endswith("_forward_fa3"):
            raise AttributeError("renamed")
        return object()

    monkeypatch.setattr(plugin, "resolve", resolve)
    with pytest.raises(RuntimeError, match="_forward_fa3"):
        plugin.arm(served(block_of(tmp_path)))
    assert registered == []


def test_a_model_config_with_no_block_arms_nothing(monkeypatch):
    registered = fake_hook_registry(monkeypatch)
    config = SimpleNamespace(hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(index_topk=2048))
    plugin.configure(None, config)
    # The native baseline is the absence of a setting - no hook, and the config as it was
    assert registered == []
    assert plugin.ARMED == []
    assert config.hf_text_config.index_topk == 2048


def test_the_standing_hooks_are_the_model_config_and_nothing_else(monkeypatch):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    plugin.register()
    assert [(target, hook) for target, hook, _ in registered] == [(plugin.MODEL_CONFIG, plugin.configure)]


def test_a_sglang_whose_model_config_moved_is_refused_by_name(monkeypatch):
    registered = fake_hook_registry(monkeypatch)

    def resolve(target):
        raise AttributeError("renamed")

    monkeypatch.setattr(plugin, "resolve", resolve)
    with pytest.raises(RuntimeError, match="ModelConfig.__init__"):
        plugin.register()
    assert registered == []


def test_a_process_armed_once_refuses_a_second_block_that_says_otherwise(monkeypatch, tmp_path):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    monkeypatch.setattr(plugin, "refuse_metadata_glue", lambda: None)
    monkeypatch.setattr(plugin, "elide_index_keys", lambda: None)
    fake_indexer_module(monkeypatch)
    first = served(block_of(tmp_path))
    plugin.arm(first)
    count = len(registered)
    # The same block again is the same server's other config - the main process builds more than one
    plugin.arm(first)
    assert len(registered) == count
    with pytest.raises(RuntimeError, match="a server runs one block"):
        plugin.arm(served(block_of(tmp_path, top_k=1024)))


def test_the_elision_answers_yes_on_the_two_pool_modules_and_leaves_the_models_logic_alone(monkeypatch):
    def original(config, layer_id):
        return layer_id % 4 != 3

    pools = [ModuleType("pool_a"), ModuleType("pool_b")]
    for module in pools:
        module.dsa_layer_skips_topk = original
        monkeypatch.setitem(sys.modules, module.__name__, module)
    model = ModuleType("model_side")
    model.dsa_layer_skips_topk = original
    monkeypatch.setattr(plugin, "ELIDED", ("pool_a.dsa_layer_skips_topk", "pool_b.dsa_layer_skips_topk"))
    plugin.elide_index_keys()
    for module in pools:
        assert all(module.dsa_layer_skips_topk(None, layer) for layer in range(8))
    # The model's own skip logic keeps its answer
    assert [model.dsa_layer_skips_topk(None, layer) for layer in range(4)] == [True, True, True, False]


def test_resolve_finds_a_method_and_names_what_it_cannot():
    assert plugin.resolve("json.JSONDecoder.decode").__name__ == "decode"
    assert plugin.resolve("json.dumps") is json.dumps
    with pytest.raises(AttributeError):
        plugin.resolve("json.JSONDecoder.no_such_method")


def test_request_ids_reach_the_batch_views():
    view = SimpleNamespace()
    plugin.carry_request_ids(view, SimpleNamespace(rids=["a", "b"]))
    assert view.rids == ["a", "b"]


def test_a_batch_without_request_ids_says_so_rather_than_raising():
    view = SimpleNamespace()
    plugin.carry_request_ids(view, SimpleNamespace())
    assert view.rids is None


def test_bind_names_arguments_however_they_were_passed():
    def target(self, first, second, third=3):
        return None

    positional = plugin.bind(target, "self", 1, 2)
    by_name = plugin.bind(target, "self", second=2, first=1)
    assert positional.arguments["second"] == 2
    assert by_name.arguments["first"] == 1
    # Defaults are filled, so a hook reading one never sees it missing
    assert positional.arguments["third"] == 3


class FakeText:
    index_topk = 2048
    index_kpool = 4
    index_kpool_compress = True
    hidden_size = 16
    num_hidden_layers = 8
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ] * 2


@pytest.fixture
def model_config_module(monkeypatch):
    module = ModuleType("sglang.srt.configs.model_config")
    module.dsa_layer_skips_topk = lambda config, layer: False
    for name in ("sglang", "sglang.srt", "sglang.srt.configs"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, ModuleType(name)))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", module)
    return module


def test_the_list_width_is_set_where_the_model_reads_it(tmp_path, model_config_module, unarmed):
    write_maps(tmp_path)
    config = config_with(block_of(tmp_path), FakeText())
    plugin.configure(None, config)
    # top_k + window_size + sink_size, and no pooled index keys left to compress
    assert config.hf_text_config.index_topk == 2048 + 512 + 4
    assert config.hf_text_config.index_kpool == 1
    assert config.hf_text_config.index_kpool_compress is False


def test_the_block_is_put_on_the_text_config_the_indexer_is_built_from(
    tmp_path, model_config_module, unarmed
):
    write_maps(tmp_path)
    config = config_with(block_of(tmp_path), FakeText())
    plugin.configure(None, config)
    assert settings.design_of(config.hf_text_config).width == 2048 + 512 + 4


def test_a_block_that_arrived_on_the_text_config_is_found_there(tmp_path, model_config_module, unarmed):
    write_maps(tmp_path)
    text = FakeText()
    text.lsa = block_of(tmp_path)
    # A nested checkpoint's text config is the top-level config's own `text_config`
    plugin.configure(None, SimpleNamespace(hf_config=SimpleNamespace(text_config=text), hf_text_config=text))
    assert text.index_topk == 2048 + 512 + 4


def test_a_config_is_armed_from_its_block_before_it_is_sized(monkeypatch, tmp_path, model_config_module):
    write_maps(tmp_path)
    armed = []
    monkeypatch.setattr(plugin, "arm", armed.append)
    plugin.configure(None, config_with(block_of(tmp_path), FakeText()))
    assert [item.design.indexer.top_k for item in armed] == [2048]


def test_a_refused_block_stops_the_launch_with_what_arrived(tmp_path, unarmed):
    with pytest.raises(ValueError, match="bucket_shape"):
        plugin.configure(None, config_with(block_of(tmp_path, bucket_shape=5)))


def test_a_projection_file_missing_a_sparse_layer_stops_the_launch(tmp_path, model_config_module, unarmed):
    write_maps(tmp_path, layers=(3,))
    config = config_with(block_of(tmp_path), FakeText())
    with pytest.raises(ValueError, match=r"missing \[7\]"):
        plugin.configure(None, config)
    assert config.hf_text_config.index_topk == 2048


def test_a_layer_that_reuses_the_top_k_before_it_needs_no_map(tmp_path, model_config_module, unarmed):
    model_config_module.dsa_layer_skips_topk = lambda config, layer: layer == 7
    write_maps(tmp_path, layers=(3,))
    plugin.configure(None, config_with(block_of(tmp_path), FakeText()))


def test_a_config_without_indexer_fields_is_left_alone(tmp_path, unarmed):
    config = config_with(block_of(tmp_path))
    plugin.configure(None, config)
    assert not hasattr(config.hf_text_config, "index_topk")


class FakeQwenText:
    hidden_size = 32
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 16
    partial_rotary_factor = 0.25
    layer_types = ["linear_attention", "full_attention"] * 2


def test_a_grouped_query_models_map_is_checked_against_its_quadratic_layers(tmp_path, unarmed):
    # Two key heads of 12 position-free dims - the map reads them side by side
    write_maps(tmp_path, layers=(1, 3), model_dim=24)
    config = config_with(block_of(tmp_path, input="key_and_query"), FakeQwenText())
    plugin.configure(None, config)
    # No list width to set - the backend's hooks read the lists
    assert not hasattr(config.hf_text_config, "index_topk")
    write_maps(tmp_path, layers=(1,), model_dim=24)
    with pytest.raises(ValueError, match=r"missing \[3\]"):
        plugin.configure(None, config)
    write_maps(tmp_path, layers=(1, 3), model_dim=32)
    with pytest.raises(ValueError, match=r"\[24, 8\]"):
        plugin.configure(None, config)


def test_a_draft_model_is_left_alone(tmp_path, unarmed):
    config = config_with(block_of(tmp_path), FakeText())
    config.is_draft_model = True
    plugin.configure(None, config)
    assert config.hf_text_config.index_kpool == 4


def test_a_banner_is_printed_once_whichever_layer_earns_it(capsys, monkeypatch):
    """Readiness waits for each banner, and every sparse layer after the first builds the same thing"""
    monkeypatch.setattr(registry, "ANNOUNCED", set())
    registry.announce_once(Banner.BUILT, 3)
    registry.announce_once(Banner.BUILT, 7)
    assert capsys.readouterr().out.splitlines() == [
        f"{Banner.BUILT} layer 3 and every sparse layer after",
    ]


def design_of(tmp_path, **overrides):
    return served(block_of(tmp_path, **overrides)).design


def test_a_server_running_context_parallel_is_refused_on_an_image_without_it(monkeypatch, tmp_path):
    def resolve(target):
        if target in plugin.REQUIRED_CP:
            raise ModuleNotFoundError("no context parallel here")
        return object()

    monkeypatch.setattr(plugin, "resolve", resolve)
    monkeypatch.setattr(context_parallel, "is_served", lambda: True)
    with pytest.raises(RuntimeError, match="is_cp_active"):
        plugin.require_context_parallel(design_of(tmp_path))


def test_a_server_without_context_parallel_does_not_need_its_symbols(monkeypatch, tmp_path):
    wanted = []
    monkeypatch.setattr(plugin, "resolve", lambda target: wanted.append(target))
    monkeypatch.setattr(context_parallel, "is_served", lambda: False)
    plugin.require_context_parallel(design_of(tmp_path))
    assert wanted == []


@pytest.mark.parametrize("served_cp", [True, False])
def test_the_key_and_query_input_under_context_parallel_needs_the_forks_chunk_key_gather(
    monkeypatch, tmp_path, served_cp
):
    wanted = []
    monkeypatch.setattr(plugin, "resolve", lambda target: wanted.append(target))
    monkeypatch.setattr(context_parallel, "is_served", lambda: served_cp)
    plugin.require_context_parallel(design_of(tmp_path, input="key_and_query", query_heads="all"))
    assert (set(plugin.REQUIRED_CP_KEY_AND_QUERY) <= set(wanted)) == served_cp


def test_the_attention_input_under_context_parallel_leaves_the_chunk_key_gather_alone(monkeypatch, tmp_path):
    wanted = []
    monkeypatch.setattr(plugin, "resolve", lambda target: wanted.append(target))
    monkeypatch.setattr(context_parallel, "is_served", lambda: True)
    plugin.require_context_parallel(design_of(tmp_path))
    assert set(wanted) == set(plugin.REQUIRED_CP)


def test_a_server_is_context_parallel_only_once_sglangs_own_state_says_so(monkeypatch):
    published = SimpleNamespace(value=False)
    parallel = SimpleNamespace(enable_prefill_cp=False, dcp_enabled=False)
    module = ModuleType("sglang.srt.runtime_context")
    module.get_context = lambda: SimpleNamespace(is_config_namespace_published=lambda name: published.value)
    module.get_parallel = lambda: parallel
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", module)
    # Before publication there is nothing to ask
    parallel.dcp_enabled = True
    assert not context_parallel.is_served()
    published.value = True
    assert context_parallel.is_served()
    parallel.dcp_enabled = False
    assert not context_parallel.is_served()
    parallel.enable_prefill_cp = True
    assert context_parallel.is_served()


def test_the_attention_input_registers_neither_key_and_query_hook(monkeypatch, tmp_path):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    monkeypatch.setattr(plugin, "refuse_metadata_glue", lambda: None)
    monkeypatch.setattr(plugin, "elide_index_keys", lambda: None)
    fake_indexer_module(monkeypatch)
    plugin.arm(served(block_of(tmp_path)))
    targets = {target for target, _, _ in registered}
    assert not targets & set(plugin.KEY_AND_QUERY_TARGETS)


class FakeIndexer:
    is_lsa = True

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "selection"


def test_the_gate_stops_an_lsa_layers_indexer_and_remembers_it_wanted_to():
    layer = SimpleNamespace(indexer=FakeIndexer())
    assert plugin.gate(lambda self, *args: True, layer) is False
    assert getattr(layer, plugin.WANTED_INDEXER) is True
    native = SimpleNamespace(indexer=SimpleNamespace())
    assert plugin.gate(lambda self, *args: True, native) is True


def test_prepare_selects_from_the_key_and_the_absorbed_query_and_puts_the_lists_where_attention_reads():
    indexer = FakeIndexer()
    layer = SimpleNamespace(indexer=indexer, layer_id=3)
    setattr(layer, plugin.WANTED_INDEXER, True)
    query, key = torch.randn(5, 2, 16), torch.randn(5, 1, 16)
    result = ("q_pe", "k_pe", query, key, "batch", "allocator", "positions", None, "scaling", None, "gate")
    prepared = plugin.prepare(result, layer)
    assert prepared[plugin.SELECTION] == "selection"
    assert prepared[: plugin.SELECTION] == result[: plugin.SELECTION]
    assert prepared[plugin.SELECTION + 1 :] == result[plugin.SELECTION + 1 :]
    call = indexer.calls[0]
    assert call["query"] is query and torch.equal(call["key"], key.squeeze(1))
    assert call["positions"] == "positions" and call["forward_batch"] == "batch" and call["layer_id"] == 3


def test_prepare_leaves_a_layer_whose_model_did_not_want_its_indexer():
    layer = SimpleNamespace(indexer=FakeIndexer(), layer_id=3)
    setattr(layer, plugin.WANTED_INDEXER, False)
    assert plugin.prepare(tuple(range(10)), layer) is None


def test_prepare_refuses_a_fused_bmm_path():
    layer = SimpleNamespace(indexer=FakeIndexer(), layer_id=3)
    setattr(layer, plugin.WANTED_INDEXER, True)
    result = (None, None, torch.zeros(1), torch.zeros(1, 1), None, None, None, None, None, "plan")
    with pytest.raises(RuntimeError, match="fused bmm"):
        plugin.prepare(result, layer)


def test_the_key_and_query_input_checks_its_maps_at_the_keys_dim(tmp_path, model_config_module, unarmed):
    write_maps(tmp_path, model_dim=12)
    text = FakeText()
    text.kv_lora_rank = 12
    config = config_with(block_of(tmp_path, input="key_and_query"), text)
    plugin.configure(None, config)
    assert text.index_topk == 2048 + 512 + 4
    write_maps(tmp_path, model_dim=16)
    with pytest.raises(ValueError, match=r"\[12, 8\]"):
        plugin.configure(None, config)


def decode_runner(block, context_length, decoding=True, variants=None):
    config = SimpleNamespace(context_len=context_length, hf_config=SimpleNamespace(lsa=block))
    return SimpleNamespace(
        attention_graph_variants=variants,
        capture_forward_mode=SimpleNamespace(is_decode=lambda: decoding),
        model_runner=SimpleNamespace(model_config=config),
    )


def test_the_decode_runner_is_handed_the_context_ranges_narrower_than_the_server(tmp_path):
    from lsa.sglang import context_buckets

    runner = decode_runner(block_of(tmp_path), 2_000_000)
    seen = []

    def capture(self):
        seen.append(self.attention_graph_variants)
        return "captured"

    assert plugin.arm_context_graphs(capture, runner) == "captured"
    narrower = tuple(limit for limit in DECODE_CONTEXT_LIMITS if limit < 2_000_000)
    assert seen == [context_buckets.ContextBuckets(narrower)]


def test_the_decode_runner_keeps_variants_it_has_and_a_server_of_one_range(tmp_path):
    held = object()
    runner = decode_runner(block_of(tmp_path), 2_000_000, variants=held)
    plugin.arm_context_graphs(lambda self: None, runner)
    assert runner.attention_graph_variants is held
    short = decode_runner(block_of(tmp_path), 100_000)
    plugin.arm_context_graphs(lambda self: None, short)
    assert short.attention_graph_variants is None


def test_a_descending_decode_is_captured_as_one_graph_whatever_the_context(tmp_path):
    block = block_of(
        tmp_path,
        decode="descend",
        top_level=1,
        node_bucket_size=4,
        node_cluster_block_size=1024,
        beam_width=2048,
    )
    runner = decode_runner(block, 2_000_000)
    plugin.arm_context_graphs(lambda self: None, runner)
    assert runner.attention_graph_variants is None


def test_a_runner_that_does_not_decode_is_left_with_one_graph(tmp_path):
    runner = decode_runner(block_of(tmp_path), 2_000_000, decoding=False)
    plugin.arm_context_graphs(lambda self: None, runner)
    assert runner.attention_graph_variants is None


def test_a_design_with_no_ranges_is_left_with_one_graph(tmp_path):
    runner = decode_runner(block_of(tmp_path, decode_context_limits=[]), 2_000_000)
    plugin.arm_context_graphs(lambda self: None, runner)
    assert runner.attention_graph_variants is None


def test_the_context_graph_hook_is_registered_where_the_runner_has_variants(monkeypatch):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    assert plugin.register_context_graphs()
    assert registered == [(plugin.CONTEXT_GRAPH_TARGETS[0], plugin.arm_context_graphs, "around")]


def test_an_image_without_the_runners_variants_serves_one_graph_and_registers_nothing(monkeypatch):
    registered = fake_hook_registry(monkeypatch)

    def resolve(target):
        raise AttributeError("no such attribute")

    monkeypatch.setattr(plugin, "resolve", resolve)
    assert not plugin.register_context_graphs()
    assert registered == []


def pisa_block(**overrides):
    return {"selector": "pisa", "block_size": 64, "keep_count": 72, **overrides}


def pisa_plugin(monkeypatch, resolve=lambda target: object()):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", resolve)
    monkeypatch.setattr(plugin, "refuse_metadata_glue", lambda: None)
    monkeypatch.setattr(plugin, "elide_index_keys", lambda: None)
    indexer_module = ModuleType("lsa.sglang.models.glm5_next.pisa_indexer")
    indexer_module.PisaIndexer = type("PisaIndexer", (), {})
    monkeypatch.setitem(sys.modules, "lsa.sglang.models.glm5_next.pisa_indexer", indexer_module)
    return registered, indexer_module.PisaIndexer


def test_the_block_names_the_selector_and_a_block_with_none_is_lsas(tmp_path):
    pisa_settings = served(pisa_block())
    assert pisa_settings.selector == "pisa"
    assert pisa_settings.pisa.width == 72 * 64
    assert served(block_of(tmp_path)).selector == "lsa"


def test_pisa_needs_no_projection_file_and_registers_its_own_indexer_in_lsas_place(monkeypatch, capsys):
    registered, indexer = pisa_plugin(monkeypatch)
    plugin.arm(served(pisa_block()))
    hooks = {target: (hook, kind) for target, hook, kind in registered}
    assert hooks[plugin.TARGETS[1]] == (indexer, "replace")
    assert hooks[plugin.KEY_AND_QUERY_TARGETS[0]] == (plugin.gate, "around")
    assert hooks[plugin.KEY_AND_QUERY_TARGETS[1]] == (plugin.prepare, "after")
    # The config hook is the standing one, and is not registered a second time
    assert plugin.MODEL_CONFIG not in hooks
    assert hooks[plugin.REPOINT_TARGET] == (pisa.reopen_repointed, "before")
    banner = capsys.readouterr().out
    assert banner.startswith(Banner.READY) and "pisa" in banner and f"width {72 * 64}" in banner


def test_pisa_arms_the_grouped_query_backends_with_its_own_selection_and_no_decode_graphs(monkeypatch):
    from lsa.sglang.models.qwen3_5 import backend
    from lsa.sglang.models.qwen3_5 import pisa as grouped_query_pisa

    registered, _ = pisa_plugin(monkeypatch)
    plugin.arm(served(pisa_block()))
    hooks = {target: hook for target, hook, _ in registered}
    for attention in plugin.GROUPED_QUERY_BACKENDS:
        assert hooks[f"{attention}.__init__"] is grouped_query_pisa.arm
        assert hooks[f"{attention}.forward_extend"] is grouped_query_pisa.select_and_read
        assert hooks[f"{attention}.forward_decode"] is grouped_query_pisa.select_and_read
        assert hooks[f"{attention}.init_forward_metadata"] is backend.maintain
    assert not set(hooks) & set(plugin.CONTEXT_GRAPH_TARGETS)


def test_pisa_resolves_the_gather_it_reads_ranks_queries_through(monkeypatch):
    wanted = []

    def resolve(target):
        wanted.append(target)
        return object()

    pisa_plugin(monkeypatch, resolve)
    plugin.arm(served(pisa_block()))
    assert set(plugin.REQUIRED_PISA) <= set(wanted)
    assert not set(plugin.REQUIRED_CP) & set(wanted)


def test_a_target_pisa_reads_that_no_longer_resolves_is_named_and_nothing_is_registered(monkeypatch):
    def resolve(target):
        if target in plugin.REQUIRED_PISA:
            raise AttributeError("renamed")
        return object()

    registered, _ = pisa_plugin(monkeypatch, resolve)
    with pytest.raises(RuntimeError, match="attn_tp_all_gather_into_tensor"):
        plugin.arm(served(pisa_block()))
    assert registered == []


def test_the_list_width_pisa_sets_is_the_blocks_kept_times_their_keys_and_no_map_is_read(unarmed):
    config = config_with(pisa_block(keep_count=40), FakeText())
    plugin.configure(None, config)
    assert config.hf_text_config.index_topk == 40 * 64
    assert config.hf_text_config.index_kpool == 1
    assert config.hf_text_config.index_kpool_compress is False


def test_a_block_with_only_debug_arms_the_dump_of_the_native_indexer_and_nothing_else(monkeypatch, tmp_path):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    dump = {"directory": str(tmp_path), "layers": [3]}
    plugin.arm(served({"debug": {"dump": dump}}))
    assert {target for target, _, kind in registered if kind == "after"} == {
        *(f"{indexer}.forward_cuda" for indexer in plugin.NATIVE_INDEXERS),
        plugin.QSA_INDEXER,
    }
    assert len(registered) == len(plugin.NATIVE_INDEXERS) + 1


class FakePisaIndexer(FakeIndexer):
    is_lsa = False
    is_pisa = True


def prepared_result():
    return (
        "q_pe",
        "k_pe",
        torch.randn(5, 2, 16),
        torch.randn(5, 1, 16),
        "batch",
        "allocator",
        "positions",
        None,
        "scaling",
        None,
    )


def test_prepare_hands_pisa_the_attention_layers_softmax_scale():
    indexer = FakePisaIndexer()
    layer = SimpleNamespace(indexer=indexer, layer_id=3, scaling=0.125)
    setattr(layer, plugin.WANTED_INDEXER, True)
    prepared = plugin.prepare(prepared_result(), layer)
    assert prepared[plugin.SELECTION] == "selection"
    assert indexer.calls[0]["logit_scale"] == 0.125


def test_lsa_is_not_handed_a_scale_it_never_asked_for():
    indexer = FakeIndexer()
    layer = SimpleNamespace(indexer=indexer, layer_id=3, scaling=0.125)
    setattr(layer, plugin.WANTED_INDEXER, True)
    plugin.prepare(prepared_result(), layer)
    assert "logit_scale" not in indexer.calls[0]


def test_the_gate_stops_a_pisa_layers_indexer_too():
    layer = SimpleNamespace(indexer=FakePisaIndexer())
    assert plugin.gate(lambda self, *args: True, layer) is False
    assert getattr(layer, plugin.WANTED_INDEXER) is True


def test_a_block_with_only_a_check_registers_nothing_for_the_checkpoints_own_indexer(monkeypatch):
    registered = fake_hook_registry(monkeypatch)
    monkeypatch.setattr(plugin, "resolve", lambda target: object())
    plugin.arm(served({"debug": {"check_index": True}}))
    assert registered == []
