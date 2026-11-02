from pathlib import Path

import pytest

from lsa.training.native.config import load

CONFIGS = Path(__file__).resolve().parents[4] / "configs/training/native/qwen35_0_8b_gpt2"


@pytest.mark.parametrize(
    ("name", "corpus", "step"),
    [
        ("pretrain_long", "gpt2", 13733),
        ("pretrain_fineweb", "fineweb_edu_100bt", 94414),
    ],
)
def test_pretrain_phases_save_the_shared_decay_start(name, corpus, step):
    [config] = load(CONFIGS / f"{name}.yaml")
    assert config.data.name == corpus
    assert config.run.checkpoint.decay_start
    assert config.run.optim.decay_start_step == step
    assert config.run.optim.decay_shape == "sqrt"


@pytest.mark.parametrize("name", ["decays", "decays_fineweb", "decay_fineweb128k"])
def test_the_decay_sweep_keeps_a_step_fixed_while_context_and_accumulation_change(name):
    configs = load(CONFIGS / f"{name}.yaml")
    expected = {
        "decays": [32768, 65536, 131072],
        "decays_fineweb": [32768, 65536],
        "decay_fineweb128k": [131072],
    }
    assert [config.data.seq_len for config in configs] == expected[name]
    assert len({config.run.name for config in configs}) == len(expected[name])
    for config in configs:
        assert config.data.name == "gpt2"
        assert config.run.optim.tokens_per_rank == config.data.seq_len
        assert config.run.optim.tokens_per_step == (2097152 if name == "decay_fineweb128k" else 1048576)
        assert config.data.seed == 1042
        assert config.run.runtime.activation_checkpoint == "full"
        assert not config.run.checkpoint.decay_start
        assert config.run.init_from.endswith("step-13733" if name == "decays" else "step-94414")


@pytest.mark.parametrize(("phase", "length"), [("32k", 32768), ("64k", 65536), ("128k", 131072)])
def test_each_decay_overlay_composes_independently(phase, length):
    [config] = load(
        CONFIGS / f"phase/decay{phase}.yaml",
        ["run.output=/tmp/native", "run.name=decay", "run.dp_shard=8", "model.flavor=0.8B-gpt2"],
    )
    assert config.data.seq_len == length
    assert config.run.optim.tokens_per_rank == length
    assert config.run.optim.steps == 3434


@pytest.mark.parametrize("keys", [1024, 2048, 4096])
def test_a_released_n_lsa_preset_names_its_checkpoint_and_the_keys_it_reads(keys):
    [config] = load(CONFIGS / f"evaluate_released_lsa_k{keys}.yaml")
    assert config.attention.kind == "lsa"
    assert config.attention.top_k == keys
    assert config.data.seq_len == 131072
    assert config.run.name == f"released-decay128k-lsa-{keys}_seq_len131072"
    assert config.scored_checkpoint().endswith(
        f"training/native/released/decay128k-lsa-{keys}/checkpoint/step-5866"
    )
    assert config.scored_checkpoint("/elsewhere/step-1") == "/elsewhere/step-1"


@pytest.mark.parametrize(
    ("preset", "kind", "budget"),
    [
        ("dense", "dense", {}),
        ("swa_2560", "window", {"window_size": 2560}),
        ("swa_4608", "window", {"window_size": 4608}),
        ("nsa_2048", "nsa", {"select_block_count": 32}),
        ("nsa_4096", "nsa", {"select_block_count": 64}),
        ("pisa_2368", "pisa", {"level_block_count": 40}),
        ("pisa_4416", "pisa", {"level_block_count": 72}),
    ],
)
def test_a_released_baseline_preset_names_its_checkpoint_and_the_budget_it_reads(preset, kind, budget):
    [config] = load(CONFIGS / f"evaluate_released_{preset}.yaml")
    arm = preset.replace("_", "-")
    assert config.attention.kind == kind
    for key, value in budget.items():
        assert getattr(config.attention, key) == value
    assert config.data.seq_len == 131072
    assert config.run.name == f"released-decay128k-{arm}_seq_len131072"
    assert config.scored_checkpoint().endswith(
        f"training/native/released/decay128k-{arm}/checkpoint/step-5866"
    )
