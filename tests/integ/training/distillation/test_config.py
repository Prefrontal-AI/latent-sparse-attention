"""Distillation-specific constraints; common YAML and sweep rules are checked at the shared loader."""

from dataclasses import asdict
from pathlib import Path

import pytest

from lsa.training.distillation import config as distillation

MINIMAL = """
model:
  indexer:
    index_dim: 32
data:
  root: caches/teacher
  train:
    - {name: natural, source: tokens, split: train}
  evaluate:
    - {name: natural, source: tokens, split: eval}
run:
  name: unit
  output: results/training
"""


def write(tmp_path, text, name="run.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_file_describes_a_distillation_run(tmp_path):
    (experiment,) = distillation.load(write(tmp_path, MINIMAL))
    assert experiment.run.name == "unit"
    assert experiment.model.indexer.index_dim == 32


def test_a_run_that_does_not_say_where_it_writes_is_refused(tmp_path):
    text = MINIMAL.replace("  output: results/training\n", "")
    with pytest.raises(ValueError, match=r"RunConfig needs \['output'\]"):
        distillation.load(write(tmp_path, text))


def test_where_a_run_writes_is_its_output_and_its_name(tmp_path):
    (experiment,) = distillation.load(write(tmp_path, MINIMAL), ["run.output=elsewhere"])
    assert experiment.out == Path("elsewhere") / "unit"


def test_two_evaluation_sets_may_not_share_a_name(tmp_path):
    text = MINIMAL.replace(
        "    - {name: natural, source: tokens, split: eval}",
        "    - {name: natural, source: tokens, split: eval}\n"
        "    - {name: natural, source: ruler_niah_multikey_2}",
    )
    with pytest.raises(ValueError, match="share a name"):
        distillation.load(write(tmp_path, text))


def test_an_index_of_no_dims_is_refused(tmp_path):
    with pytest.raises(ValueError, match="index_dim must be at least 1"):
        distillation.load(write(tmp_path, MINIMAL), ["model.indexer.index_dim=0"])


def test_the_indexer_block_takes_the_designs_knobs_and_no_others(tmp_path):
    with pytest.raises(ValueError, match="no field"):
        distillation.load(write(tmp_path, MINIMAL), ["model.indexer.sparsity=2"])
    (experiment,) = distillation.load(write(tmp_path, MINIMAL), ["model.indexer.top_k=2048"])
    assert experiment.model.indexer.top_k == 2048


def test_a_window_of_no_documents_is_refused(tmp_path):
    with pytest.raises(ValueError, match="window must be positive"):
        distillation.load(write(tmp_path, MINIMAL), ["data.window=0"])


def test_a_run_may_take_no_steps_at_all(tmp_path):
    (experiment,) = distillation.load(write(tmp_path, MINIMAL), ["run.optim.steps=0"])
    assert experiment.run.optim.steps == 0


def test_a_negative_step_count_is_refused(tmp_path):
    with pytest.raises(ValueError, match="steps must not be negative"):
        distillation.load(write(tmp_path, MINIMAL), ["run.optim.steps=-1"])


def test_an_objective_target_the_loss_cannot_read_is_refused():
    from lsa.training.distillation.config import ObjectiveConfig

    with pytest.raises(ValueError, match="objective.target must be one of"):
        ObjectiveConfig(target="dsa")


def test_a_target_depth_on_a_target_not_read_off_the_dense_ranking_is_refused():
    from lsa.training.distillation.config import ObjectiveConfig
    from lsa.training.distillation.constants import Target

    with pytest.raises(ValueError, match="truncates the dense ranking"):
        ObjectiveConfig(target=Target.NATIVE_SELECTION, target_depth=16)


def test_a_run_on_the_key_and_query_trains_every_head_summed(tmp_path):
    text = MINIMAL.replace("    index_dim: 32\n", "    index_dim: 32\n    input: key_and_query\n")
    with pytest.raises(ValueError, match="trains query_heads: all"):
        distillation.load(write(tmp_path, text), ["model.indexer.query_heads=rank"])
    (run,) = distillation.load(write(tmp_path, text))
    assert (run.model.indexer.input, run.model.indexer.query_heads) == ("key_and_query", "all")


def test_a_run_reads_the_designs_input_unless_it_names_the_attention_input(tmp_path):
    (run,) = distillation.load(write(tmp_path, MINIMAL))
    assert (run.model.indexer.input, run.model.indexer.query_heads) == ("key_and_query", "all")
    text = MINIMAL.replace("    index_dim: 32\n", "    index_dim: 32\n    input: attention_input\n")
    (run,) = distillation.load(write(tmp_path, text))
    assert (run.model.indexer.input, run.model.indexer.query_heads) == ("attention_input", None)


def test_a_closed_block_names_a_cluster_block_of_keys(tmp_path):
    from lsa.training.distillation.config import ObjectiveConfig

    assert ObjectiveConfig().closed_block is None
    with pytest.raises(ValueError, match="closed_block must be positive"):
        ObjectiveConfig(closed_block=0)


def test_a_closed_block_on_a_target_not_read_off_the_cached_ranking_is_refused():
    from lsa.training.distillation.config import ObjectiveConfig
    from lsa.training.distillation.constants import Target

    with pytest.raises(ValueError, match="hides keys of the cached top-k ranking"):
        ObjectiveConfig(target=Target.NATIVE_ATTENTION, closed_block=8192)
    assert ObjectiveConfig(target=Target.DENSE_ON_NATIVE, closed_block=8192).closed_block == 8192


def test_a_closed_block_is_read_on_the_key_and_query(tmp_path):
    text = MINIMAL.replace(
        "  output: results/training\n", "  output: results/training\n  objective:\n    closed_block: 8192\n"
    )
    (run,) = distillation.load(write(tmp_path, text))
    assert run.run.objective.closed_block == 8192
    attention = text.replace("    index_dim: 32\n", "    index_dim: 32\n    input: attention_input\n")
    with pytest.raises(ValueError, match="closed_block is read on the key and query"):
        distillation.load(write(tmp_path, attention))


@pytest.mark.parametrize(
    "path",
    sorted(
        Path("configs/training/distillation/glm53_flash").glob("long_16k_1m_heavy_key_and_query_d2048*.yaml")
    ),
)
def test_every_key_and_query_recipe_composes_under_its_own_name(path):
    runs = distillation.load(path)
    assert [run.run.name for run in runs] == [path.stem]
    assert runs[0].model.indexer.input == "key_and_query"


def test_a_target_depth_on_the_full_dense_target_is_refused():
    from lsa.training.distillation.config import ObjectiveConfig
    from lsa.training.distillation.constants import Target

    with pytest.raises(ValueError, match="truncates the dense ranking.*'dense_full'"):
        ObjectiveConfig(target=Target.DENSE_FULL, target_depth=64)
    assert ObjectiveConfig(target=Target.DENSE_FULL).target_depth is None


def test_the_full_dense_target_needs_the_key_and_query_input(tmp_path):
    attention = MINIMAL.replace("    index_dim: 32\n", "    index_dim: 32\n    input: attention_input\n")
    with pytest.raises(
        ValueError, match="'dense_full'.*model.indexer.input 'attention_input'.*needs 'key_and_query'"
    ):
        distillation.load(write(tmp_path, attention), ["run.objective.target=dense_full"])
    (run,) = distillation.load(write(tmp_path, MINIMAL), ["run.objective.target=dense_full"])
    assert run.run.objective.target == "dense_full"


def test_the_full_arm_differs_from_the_truncate_arm_in_the_target_alone():
    runs = Path("configs/training/distillation/glm53_flash")
    for seed in range(3):
        (cut,) = distillation.load(runs / f"long_16k_1m_heavy_key_and_query_d2048_truncate_seed{seed}.yaml")
        (full,) = distillation.load(runs / f"long_16k_1m_heavy_key_and_query_dense_full_seed{seed}.yaml")
        assert (full.run.objective.target, full.run.objective.target_depth) == ("dense_full", None)
        assert (cut.run.objective.target, cut.run.objective.target_depth) == ("dense", 2048)
        assert full.run.name == f"long_16k_1m_heavy_key_and_query_dense_full_seed{seed}"
        # Every other field of the resolved run is the same, bar the names the two arms report under
        same = lambda run: {  # noqa: E731
            **asdict(run),
            "run": {
                **asdict(run.run),
                "name": None,
                "objective": {**asdict(run.run.objective), "target": None, "target_depth": None},
                "wandb": {**asdict(run.run.wandb), "prefix": None, "tags": None},
            },
        }
        assert same(full) == same(cut)
