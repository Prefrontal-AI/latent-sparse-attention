"""The design's settings - every design builds, and every refusal names what arrived and what was wanted."""

import json

import pytest

from lsa.budget import EarlyBudget
from lsa.sglang.settings.design import DesignSettings, IndexerConfig, PisaConfig, ServingChoices, Shape

FLAT = {"window_size": 512, "sink_size": 4, "top_k": 4096, "top_p": 1024, "index_dim": 64, "pool_size": 4}
FLAT = {**FLAT, "bucket_size": 8, "bucket_cap": 16, "top_level": 0}
FIXED = {
    **FLAT,
    "top_level": 1,
    "node_bucket_size": 4,
    "node_cluster_block_size": 1024,
    "beam_width": 2048,
}
TREE = {
    **FLAT,
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 4096,
    "beam_width": 2048,
    "start_nodes": 32768,
}
FP8 = {"node_centroid_precision": "fp8"}
SHAPE = {"rank": 2, "deviations": 2}


def design(indexer: dict, serving: dict | None = None) -> DesignSettings:
    return DesignSettings.from_blocks(indexer, serving or {})


@pytest.mark.parametrize(
    ("name", "indexer", "serving"),
    [
        ("flat", FLAT, {}),
        ("fixed", FIXED, {}),
        ("tree", TREE, FP8),
        ("flat k2048", {**FLAT, "top_k": 2048, "top_p": 512}, {}),
        ("flat no pooling", {**FLAT, "pool_size": None}, {}),
        ("tree descend", TREE, {**FP8, "decode": "descend"}),
        ("tree node shape", {**TREE, "node_bucket_shape": SHAPE}, FP8),
        ("flat bucket shape", {**FLAT, "bucket_shape": SHAPE}, {}),
    ],
)
def test_every_design_builds(name, indexer, serving):
    settings = design(indexer, serving)

    assert settings.indexer.top_k == indexer["top_k"], name
    assert settings.width == indexer["top_k"] + 512 + 4


def test_scalar_per_level_knob_means_every_level():
    settings = design(TREE, FP8).indexer

    assert [settings.at("beam_width", level) for level in (1, 8, 16)] == [2048, 2048, 2048]


def test_list_per_level_knob_starts_at_level_one():
    settings = design(
        {
            **FIXED,
            "top_level": 2,
            "beam_width": [512, 1024],
            "node_bucket_size": 4,
            "node_cluster_block_size": [1024, 4096],
        }
    ).indexer

    assert settings.at("beam_width", 1) == 512
    assert settings.at("beam_width", 2) == 1024
    assert settings.at("node_cluster_block_size", 2) == 4096
    assert isinstance(settings.beam_width, tuple)


def test_at_refuses_level_zero_and_a_knob_that_does_not_vary():
    settings = design(FIXED).indexer

    with pytest.raises(ValueError, match="levels 1 to 1 - asked for level 0"):
        settings.at("beam_width", 0)
    with pytest.raises(ValueError, match="does not vary by level"):
        settings.at("top_k", 1)


def test_per_level_list_must_be_top_level_long():
    with pytest.raises(ValueError, match="got 2 for top_level 1"):
        design({**FIXED, "beam_width": [512, 1024]})


def test_per_level_value_must_be_positive_whole():
    with pytest.raises(ValueError, match="beam_width must be at least 1 - got 0"):
        design({**FIXED, "beam_width": 0})
    with pytest.raises(ValueError, match="must be an integer"):
        design({**FIXED, "beam_width": 2.5})


def test_bucket_cap_at_least_twice_bucket_size():
    design({**FLAT, "bucket_cap": 16})
    with pytest.raises(
        ValueError,
        match=r"bucket_cap must be at least twice bucket_size.*bucket_cap 15 against bucket_size 8",
    ):
        design({**FLAT, "bucket_cap": 15})


def test_bucket_size_divides_window_size():
    with pytest.raises(ValueError, match="bucket_size must divide window_size.*7 does not divide 512"):
        design({**FLAT, "bucket_size": 7, "bucket_cap": 14, "pool_size": None})


def test_pool_size_divides_bucket_size():
    with pytest.raises(ValueError, match="pool_size must divide bucket_size.*3 does not divide 8"):
        design({**FLAT, "window_size": 504, "pool_size": 3})


def test_pool_size_divides_bucket_cap():
    with pytest.raises(ValueError, match="pool_size must divide bucket_cap.*4 does not divide 10"):
        design({**FLAT, "bucket_size": 4, "bucket_cap": 10, "pool_size": 4})


def test_pool_size_divides_window_size():
    with pytest.raises(ValueError, match="pool_size must divide window_size.*8 does not"):
        design({**FLAT, "window_size": 516, "bucket_size": 4, "bucket_cap": 8, "pool_size": 8, "top_p": 1024})


def test_pool_size_none_is_no_pooling():
    assert design({**FLAT, "pool_size": None}).indexer.pool_size is None
    with pytest.raises(ValueError, match="pool_size must be at least 1 - got 0"):
        design({**FLAT, "pool_size": 0})


def test_top_p_must_be_able_to_fill_top_k():
    design({**FLAT, "top_p": 256})
    with pytest.raises(ValueError, match=r"top_p 255 times bucket_cap 16 is 4080, under top_k 4096"):
        design({**FLAT, "top_p": 255})


def test_node_knobs_set_iff_top_level_is_one_or_more():
    with pytest.raises(ValueError, match=r"\['beam_width'\] are node knobs and top_level is 0"):
        design({**FLAT, "beam_width": 2048})
    with pytest.raises(ValueError, match=r"top_level 1 needs the node knobs \['beam_width'\]"):
        design({**FIXED, "beam_width": None})
    with pytest.raises(ValueError, match="needs the node knobs"):
        design({**FLAT, "top_level": 1})


def test_start_nodes_only_with_top_level_above_one():
    with pytest.raises(ValueError, match="start_nodes needs a hierarchy of more than one level.*top_level 1"):
        design({**FIXED, "start_nodes": 32768})
    with pytest.raises(ValueError, match="start_nodes needs a hierarchy of more than one level.*top_level 0"):
        design({**FLAT, "start_nodes": 32768})
    with pytest.raises(ValueError, match="start_nodes must be at least 1"):
        design({**TREE, "start_nodes": 0})


def test_decode_tree_only_with_a_hierarchy():
    design(FIXED, {"decode": "descend"})
    with pytest.raises(ValueError, match="decode descend needs a hierarchy.*top_level 0"):
        design(FLAT, {"decode": "descend"})


def test_fp8_node_centroids_only_with_a_hierarchy():
    design(FIXED, FP8)
    with pytest.raises(ValueError, match="fp8 node centroids need node buckets.*top_level 0"):
        design(FLAT, FP8)


def test_serving_options_are_refused_when_unknown():
    with pytest.raises(ValueError, match="node_centroid_precision must be one of.*got 'fp4'"):
        ServingChoices(node_centroid_precision="fp4")
    with pytest.raises(ValueError, match="decode must be one of.*got 'beam'"):
        ServingChoices(decode="beam")
    with pytest.raises(ValueError, match="kernels must be one of.*got 'cuda'"):
        ServingChoices(kernels="cuda")


def test_the_kernels_are_triton_unless_a_design_asks_for_cutedsl():
    assert ServingChoices().kernels == "triton"
    assert ServingChoices(kernels="cutedsl").kernels == "cutedsl"


def test_unknown_keys_are_refused_in_both_blocks():
    with pytest.raises(ValueError, match=r"model.indexer has no knobs \['centroid_precision'\]"):
        IndexerConfig.from_mapping({"centroid_precision": "fp8"})
    with pytest.raises(ValueError, match=r"model.serving has no knobs \['scan'\]"):
        ServingChoices.from_mapping({"scan": "fp8"})


@pytest.mark.parametrize("name", ["window_size", "top_k", "top_p", "index_dim", "bucket_size", "bucket_cap"])
def test_sizes_must_be_positive(name):
    with pytest.raises(ValueError, match=f"{name} must be at least 1 - got 0"):
        design({**FLAT, name: 0})


def test_sink_size_may_be_zero_but_not_negative():
    assert design({**FLAT, "sink_size": 0}).indexer.sink_size == 0
    with pytest.raises(ValueError, match="sink_size must be at least 0 - got -1"):
        design({**FLAT, "sink_size": -1})


def test_the_settings_block_is_both_blocks_flattened_and_reads_back_whole():
    made = design(TREE, {**FP8, "decode": "descend", "projections": "/maps/x.pt"})
    body = made.body()
    assert body["top_k"] == 4096 and body["decode"] == "descend" and body["projections"] == "/maps/x.pt"
    # A json round trip turns the per-level tuples into lists, which read back the same
    assert DesignSettings.parse(json.loads(json.dumps(body))) == made


def test_a_settings_block_with_a_key_neither_block_has_is_refused():
    with pytest.raises(ValueError, match=r"\['recent'\], which are neither knobs of model.indexer"):
        DesignSettings.parse({**design(FLAT).body(), "recent": 508})


def test_a_top_p_or_beam_past_what_the_engines_top_k_selects_is_refused():
    with pytest.raises(ValueError, match="top_p is at most the engine's top-k limit 2048 - got 4096"):
        design({**FLAT, "top_p": 4096})
    with pytest.raises(ValueError, match="beam_width is at most the engine's top-k limit 2048 - got 4096"):
        design({**FIXED, "beam_width": 4096})
    with pytest.raises(ValueError, match="beam_width is at most the engine's top-k limit 2048 - got 3000"):
        design({**TREE, "beam_width": [2048, 3000, *[1024] * 14]})


def test_a_shape_is_a_rank_and_deviations_read_from_a_mapping():
    settings = design({**TREE, "node_bucket_shape": SHAPE, "bucket_shape": {"rank": 0}}, FP8).indexer
    assert settings.node_bucket_shape == Shape(rank=2, deviations=2.0)
    assert settings.bucket_shape == Shape(rank=0, deviations=2.0)
    assert isinstance(settings.node_bucket_shape.deviations, float)
    assert design(FLAT).indexer.bucket_shape is None and design(FLAT).indexer.node_bucket_shape is None


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ({"rank": -1}, "rank must be at least 0 - got -1"),
        ({"rank": 9}, "at most 8 directions - got rank 9"),
        ({"rank": 1.5}, "rank must be an integer"),
        ({"deviations": 0}, "deviations must be positive - got 0"),
        ({"deviations": "two"}, "deviations must be a number"),
        ({"rank": 2, "z": 2}, r"bucket_shape has no knobs \['z'\]"),
        (2, "bucket_shape is a mapping of rank and deviations, or null - got 2"),
    ],
)
def test_a_shape_the_kernels_cannot_hold_is_refused(shape, message):
    with pytest.raises(ValueError, match=message):
        design({**FLAT, "bucket_shape": shape})


def test_a_bucket_shape_keeps_no_more_directions_than_a_bucket_holds_keys():
    small = {**FLAT, "bucket_size": 2, "bucket_cap": 4, "pool_size": None}
    design({**small, "bucket_shape": {"rank": 4}})
    with pytest.raises(
        ValueError, match="at most bucket_cap 4 directions, the keys a bucket holds - got rank 8"
    ):
        design({**small, "bucket_shape": {"rank": 8}})


def test_a_node_bucket_shape_needs_node_buckets():
    with pytest.raises(ValueError, match="node_bucket_shape shapes node buckets and top_level is 0"):
        design({**FLAT, "node_bucket_shape": SHAPE})


def test_a_shape_needs_an_index_dim_its_packed_basis_reads_in_words():
    with pytest.raises(ValueError, match="index_dim 48 is not a multiple of 32"):
        design({**FLAT, "index_dim": 48, "bucket_shape": SHAPE})
    with pytest.raises(ValueError, match="keeps at most index_dim 4 directions - got rank 8"):
        design({**FLAT, "index_dim": 4, "bucket_shape": {"rank": 8}})


def test_a_shape_basis_the_design_lacks_is_refused():
    with pytest.raises(ValueError, match="shape_basis must be one of.*got 'bf16'"):
        ServingChoices(shape_basis="bf16")


def test_a_shaped_settings_block_reads_back_whole():
    made = design({**TREE, "node_bucket_shape": SHAPE, "bucket_shape": {"rank": 1, "deviations": 1}}, FP8)
    body = json.loads(json.dumps(made.body()))
    assert body["node_bucket_shape"] == {"rank": 2, "deviations": 2.0}
    assert DesignSettings.parse(body) == made


def test_the_indexer_input_reads_back_as_set():
    settings = DesignSettings.from_blocks({"input": "attention_input"}, {})
    assert DesignSettings.parse(settings.body()).indexer.input == "attention_input"


def test_an_indexer_input_the_design_lacks_is_refused():
    with pytest.raises(ValueError, match="input must be one of"):
        IndexerConfig(input="latent")


def test_query_heads_read_back_as_set_and_only_the_key_and_query_has_heads():
    settings = DesignSettings.from_blocks({"query_heads": "rank"}, {})
    assert DesignSettings.parse(settings.body()).indexer.query_heads == "rank"
    x = DesignSettings.from_blocks({"input": "attention_input"}, {})
    assert x.indexer.query_heads is None
    assert DesignSettings.parse(x.body()) == x
    for heads in ("all", "rank"):
        with pytest.raises(ValueError, match="has no heads"):
            IndexerConfig(input="attention_input", query_heads=heads)
    with pytest.raises(ValueError, match="query_heads must be one of"):
        IndexerConfig(query_heads="local")


def test_prefix_reuse_is_off_by_default_and_a_flag():
    assert ServingChoices().prefix_reuse is False
    assert ServingChoices.from_mapping({"prefix_reuse": True}).prefix_reuse is True
    with pytest.raises(ValueError, match="prefix_reuse"):
        ServingChoices(prefix_reuse="no")


def test_pisa_defaults_are_the_papers_pyramid_at_lsas_budget():
    config = PisaConfig()
    assert (config.block_size, config.keep_count, config.branching, config.forced_blocks) == (64, 72, 2, True)
    # A beam left unset is the keep count, and a list is the blocks kept times their keys
    assert (config.beam, config.width) == (72, 4608)
    assert PisaConfig(beam_width=40).beam == 40


def test_a_pisa_settings_block_names_its_selector_and_reads_back_whole():
    config = PisaConfig(block_size=32, keep_count=100, branching=4, beam_width=50, forced_blocks=False)
    body = json.loads(json.dumps(config.body()))
    assert body["selector"] == "pisa"
    assert PisaConfig.parse(body) == config


def test_a_pisa_settings_block_with_a_key_it_lacks_or_another_selector_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        PisaConfig.parse({**PisaConfig().body(), "top_k": 1})
    with pytest.raises(ValueError, match="not 'pisa'"):
        PisaConfig.parse({"block_size": 64})
    with pytest.raises(ValueError, match="no knobs"):
        PisaConfig.from_mapping({"window_size": 512})


def test_an_lsa_reader_refuses_a_pisa_settings_block_by_name():
    with pytest.raises(ValueError, match="selector 'pisa'"):
        DesignSettings.parse(PisaConfig().body())


def test_spare_contexts_are_a_whole_number_and_none_by_default():
    assert ServingChoices().spare_contexts == 0
    assert ServingChoices.from_mapping({"spare_contexts": 16}).spare_contexts == 16
    for refused in (-1, 1.5, True):
        with pytest.raises(ValueError, match="spare_contexts"):
            ServingChoices(spare_contexts=refused)


EARLY = {"position": 1048576, "top_k": 2048, "top_p": 512}


def test_an_early_budget_is_none_by_default_and_reads_back_whole():
    assert design(FLAT).indexer.early_budget is None
    made = design({**FLAT, "early_budget": EARLY})
    assert made.indexer.early_budget == EarlyBudget(position=1048576, top_k=2048, top_p=512)
    assert made.width == 4096 + 512 + 4
    assert DesignSettings.parse(json.loads(json.dumps(made.body()))) == made


def test_an_early_budget_may_equal_the_designs_own():
    design({**FLAT, "early_budget": {**EARLY, "top_k": 4096, "top_p": 1024}})


@pytest.mark.parametrize("name", ["position", "top_k", "top_p"])
@pytest.mark.parametrize("value", [0, -1, 2.5, True, "8"])
def test_an_early_budget_field_must_be_a_positive_whole_number(name, value):
    with pytest.raises(ValueError, match=f"early_budget {name} must be"):
        design({**FLAT, "early_budget": {**EARLY, name: value}})


def test_an_early_budget_has_exactly_a_position_a_top_k_and_a_top_p():
    with pytest.raises(ValueError, match=r"early_budget has the keys \['position', 'top_k', 'top_p'\]"):
        design({**FLAT, "early_budget": {"position": 1024, "top_k": 2048}})
    with pytest.raises(ValueError, match="early_budget has the keys"):
        design({**FLAT, "early_budget": {**EARLY, "window_size": 8}})
    with pytest.raises(ValueError, match="early_budget is a mapping"):
        design({**FLAT, "early_budget": 2048})


def test_an_early_budget_over_the_designs_own_is_refused():
    with pytest.raises(
        ValueError, match="early_budget top_k is at most the design's own top_k - got 8192 against 4096"
    ):
        design({**FLAT, "early_budget": {**EARLY, "top_k": 8192}})
    with pytest.raises(
        ValueError, match="early_budget top_p is at most the design's own top_p - got 2048 against 1024"
    ):
        design({**FLAT, "early_budget": {**EARLY, "top_p": 2048}})


def test_an_early_budget_top_p_must_be_able_to_fill_its_top_k():
    design({**FLAT, "early_budget": {**EARLY, "top_k": 2048, "top_p": 128}})
    with pytest.raises(ValueError, match=r"top_p 127 times bucket_cap 16 is 2032, under top_k 2048"):
        design({**FLAT, "early_budget": {**EARLY, "top_k": 2048, "top_p": 127}})


@pytest.mark.parametrize("indexer", [FIXED, TREE])
def test_an_early_budget_on_a_hierarchy_is_refused(indexer):
    with pytest.raises(
        ValueError,
        match="early_budget is for the flat design, which a hierarchy does not read - got early_budget with top_level",
    ):
        design({**indexer, "early_budget": EARLY}, FP8)
