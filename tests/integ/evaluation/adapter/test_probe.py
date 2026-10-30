"""The config-level screen - what a checkpoint says about itself before any weights are loaded."""

from types import SimpleNamespace

import pytest

from lsa.evaluation.adapter import probe


def test_head_dim_comes_from_the_config_s_head_dim_when_there_is_one():
    assert probe.head_dim_of(SimpleNamespace(head_dim=256, hidden_size=4096)) == 256


def test_head_dim_falls_back_to_the_hidden_size_split_across_the_heads():
    assert probe.head_dim_of(SimpleNamespace(hidden_size=4096, num_attention_heads=32)) == 128


def test_a_config_that_names_no_shape_at_all_is_rejected():
    with pytest.raises(ValueError, match="neither head_dim"):
        probe.head_dim_of(SimpleNamespace())


def test_a_partial_rotary_factor_names_the_rotated_dims():
    assert probe.rotary_dim_of(SimpleNamespace(partial_rotary_factor=0.25), 256) == 64


def test_a_rope_head_dim_names_them_outright():
    assert probe.rotary_dim_of(SimpleNamespace(qk_rope_head_dim=64), 512) == 64


def test_a_config_that_names_no_fraction_rotates_the_whole_head():
    assert probe.rotary_dim_of(SimpleNamespace(), 128) == 128


def test_a_model_that_declares_no_layer_types_is_uniform():
    assert probe.layer_stack(SimpleNamespace(num_hidden_layers=32)) == (32, 32, [])


def test_a_hybrid_model_counts_only_the_layers_that_would_be_swapped():
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 8
    assert probe.layer_stack(SimpleNamespace(layer_types=layer_types)) == (
        32,
        8,
        ["full_attention", "linear_attention"],
    )


def test_a_stack_with_its_own_vocabulary_matches_nothing_by_default():
    layer_types = ["linear_attention"] * 3 + ["qwen_sparse_attention"]
    count, matched, declared = probe.layer_stack(SimpleNamespace(layer_types=layer_types))
    assert (count, matched) == (4, 0)
    assert declared == ["linear_attention", "qwen_sparse_attention"]


def test_naming_the_type_matches_it():
    layer_types = ["linear_attention"] * 3 + ["qwen_sparse_attention"]
    assert probe.layer_stack(SimpleNamespace(layer_types=layer_types), ("qwen_sparse_attention",))[1] == 1


def test_a_latent_cache_marks_a_model_the_generic_read_does_not_serve():
    assert probe.is_mla(SimpleNamespace(kv_lora_rank=512))
    assert not probe.is_mla(SimpleNamespace(num_key_value_heads=4))
