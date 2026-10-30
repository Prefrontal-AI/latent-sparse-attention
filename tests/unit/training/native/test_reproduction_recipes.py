import math
from pathlib import Path

import pytest

from lsa.training.native.config import load

CONFIGS = Path(__file__).resolve().parents[4] / "configs/training/native"


@pytest.mark.parametrize(
    ("name", "tokens", "warmup_tokens", "global_batch", "scheduler"),
    [
        ("qwen35_0_8b_gpt2/pretrain_long", 18_000_000_000, 900_000_000, 1048576, "wsd"),
        ("qwen35_0_8b_gpt2/pretrain_fineweb", 100_000_000_000, 1_000_000_000, 1048576, "wsd"),
        ("qwen35_0_8b_gpt2/decays", 3_600_000_000, 100_000_000, 1048576, "wsd"),
        ("qwen35_0_8b_gpt2/decays_fineweb", 18_400_000_000, 100_000_000, 1048576, "wsd"),
        ("qwen35_0_8b_gpt2/decay_fineweb128k", 12_300_000_000, 100_000_000, 2097152, "wsd"),
        ("dense_340m_gpt2/dsa", 20_000_000_000, 420_000_000, 1048576, "cosine"),
    ],
)
def test_source_token_budgets_resolve_by_ceil(name, tokens, warmup_tokens, global_batch, scheduler):
    for config in load(CONFIGS / f"{name}.yaml"):
        assert config.run.optim.steps == math.ceil(tokens / global_batch)
        assert config.run.optim.warmup_steps == math.ceil(warmup_tokens / global_batch)
        assert config.run.optim.tokens_per_step == global_batch
        assert config.run.optim.schedule == scheduler
        assert config.run.runtime.seed == 42
        assert config.data.seed == (1042 if "decay" in name else 42)


@pytest.mark.parametrize(
    "arm", ["nsam8", "nsam8_selection", "pisam8", "pisam8_historical_budgets", "pisam8_window_budgets"]
)
def test_nsa_and_pisa_select_the_source_fineweb_trunk(arm):
    for config in load(CONFIGS / f"qwen35_0_8b_gpt2/{arm}.yaml"):
        assert config.data.name == "fineweb_edu_100bt"
        assert (config.run.optim.steps, config.run.optim.warmup_steps, config.run.optim.decay_ratio) == (
            95368,
            954,
            0.01,
        )
        assert config.data.seed == config.run.runtime.seed == 42


def test_mqa_and_learned_lsa_use_their_distinct_upstream_head_shapes():
    for config in load(CONFIGS / "qwen35_0_8b_gpt2/lsa_mqa8.yaml"):
        assert (
            config.model.query_head_count,
            config.model.kv_head_count,
            config.model.head_dim,
            config.model.rotary_dim,
        ) == (8, 1, 128, 0)
        assert config.attention.query_head_reduction == "max"
    for config in load(CONFIGS / "qwen35_0_8b_gpt2/lsa_indexer.yaml"):
        assert (
            config.model.query_head_count,
            config.model.kv_head_count,
            config.model.head_dim,
            config.model.rotary_dim,
        ) == (8, 2, 256, 64)
        assert config.attention.index_dim == 64
        assert config.attention.index_query_heads == "concat"
        assert config.data.name == "gpt2"


@pytest.mark.parametrize("name", ["lsa_onehead", "lsa_onehead_indexer"])
def test_onehead_summaries_and_indexer_preserve_source_shapes(name):
    configs = load(CONFIGS / f"qwen35_0_8b_gpt2/{name}.yaml")
    assert len(configs) == 4
    assert {config.attention.summaries for config in configs} == {False, True}
    for config in configs:
        assert (
            config.model.query_head_count,
            config.model.kv_head_count,
            config.model.head_dim,
            config.model.rotary_dim,
        ) == (1, 1, 128, 0)
        if name.endswith("indexer"):
            assert (
                config.attention.index_dim,
                config.attention.index_training,
                config.attention.index_norm,
                config.attention.index_rotary_dim,
            ) == (64, "ste", "l2", 0)


@pytest.mark.parametrize(
    ("pretrain", "decay", "knobs"),
    [
        ("pretrain_fineweb_window", "decay_fineweb128k_window", {"kind": "window", "window_size": 2560}),
        ("nsam8", "decay_fineweb128k_nsa", {"kind": "nsa", "select_block_count": 32}),
        ("pisam8", "decay_fineweb128k_pisa", {"kind": "pisa", "level_block_count": 40}),
        ("pretrain_fineweb_lsa", "decay_fineweb128k_lsa", {"kind": "lsa", "top_k": 2048}),
    ],
)
def test_each_arm_decays_at_128k_from_its_own_pretrain(pretrain, decay, knobs):
    (trunk,) = load(CONFIGS / f"qwen35_0_8b_gpt2/{pretrain}.yaml")
    (branch,) = load(CONFIGS / f"qwen35_0_8b_gpt2/{decay}.yaml")
    assert trunk.run.init_from is None
    assert branch.run.init_from == f"{trunk.out}/checkpoint/step-94414"
    assert (branch.run.optim.steps, branch.run.optim.warmup_steps, branch.run.optim.tokens_per_step) == (
        5866,
        48,
        2097152,
    )
    assert (branch.data.name, branch.data.seq_len, branch.data.seed) == ("gpt2", 131072, 1042)
    for knob, value in knobs.items():
        assert getattr(branch.attention, knob) == value
    assert trunk.attention.kind == branch.attention.kind
