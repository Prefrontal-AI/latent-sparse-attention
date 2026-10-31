from types import SimpleNamespace

import pytest

from lsa.training.native import run
from lsa.training.native.config import AttentionConfig, OptimConfig


@pytest.mark.parametrize(
    "attention,details",
    [
        (
            AttentionConfig(kind="nsa", select_block_count=32, compress_hidden_dim=64),
            [
                "compression 32/16",
                "compress_hidden_dim 64",
                "selection 32 blocks of 64",
                "key_budget 2560 plus compressed tokens",
            ],
        ),
        (
            AttentionConfig(kind="pisa", level_block_count=41, local_block_count=8),
            [
                "leaf_block_size 64",
                "branching 2",
                "level_block_count 41",
                "local_block_count 8",
                "key_budget 2624",
            ],
        ),
    ],
)
def test_summary_names_sparse_branch_settings_and_actual_key_budget(monkeypatch, attention, details):
    experiment = SimpleNamespace(
        attention=attention,
        run=SimpleNamespace(
            name="sparse",
            init_from=None,
            optim=OptimConfig(steps=3, warmup_steps=0),
            runtime=SimpleNamespace(seed=42),
            evaluation=SimpleNamespace(enabled=False),
        ),
        model=SimpleNamespace(flavor="0.8B-gpt2"),
        data=SimpleNamespace(seq_len=4096, seed=1042),
        tokenizer_path="/fixture/tokenizer",
    )
    monkeypatch.setattr(run, "describe_corpora", lambda _: "fixture token shards")
    summary = run.describe(experiment)
    for detail in details:
        assert detail in summary
