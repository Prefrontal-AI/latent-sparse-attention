"""Every shipped config loads.

A file under `configs/` is a run someone will start, so a schema change that breaks one should fail here
rather than when the job is submitted.
Every tree has its own schema and each is loaded through it, which is what makes this a gate on the config
rather than on the loader - the three evaluation trees as much as the training ones.
The distillation and evaluation trees nest by model - `qwen35_4b/`, `glm53_flash/` - so a glob is what
finds them rather than a list.

The paths come off `__file__` rather than off the working directory.
A relative glob matches nothing from anywhere but the repository root, and a loop over nothing passes having
checked nothing - which is what this used to do.
"""

from pathlib import Path

import pytest

from lsa import site
from lsa.evaluation import config as evaluation
from lsa.evaluation.constants import Evaluation
from lsa.training.distillation import config as distillation
from lsa.training.distillation.data import teacher_config
from lsa.training.native import config as native

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def yamls(*parts):
    return sorted(CONFIGS.joinpath(*parts).rglob("*.yaml"))


DISTILLATION = yamls("training", "distillation")
# Phase overlays are composed by native runs and checked independently in the training domain
NATIVE = [path for path in yamls("training", "native") if "phase" not in path.relative_to(CONFIGS).parts]
TEACHER = yamls("data", "targets")


def is_partial(path):
    """A file laid over a configuration rather than run - its name starts with an underscore"""
    return path.name.startswith("_")


def on_b300(path):
    """A config for the B300 - its name closes with the card, or opens with it in the files not yet renamed"""
    return path.stem.endswith("_b300") or path.stem.startswith("b300_")


ACCURACY = yamls("evaluation", Evaluation.ACCURACY)
KERNELS = yamls("evaluation", Evaluation.KERNELS)
PRODUCTION = yamls("evaluation", Evaluation.PRODUCTION)

TREES = {
    "distillation": DISTILLATION,
    "native": NATIVE,
    "teacher": TEACHER,
    "accuracy": ACCURACY,
    "kernels": KERNELS,
    "production": PRODUCTION,
}


def config_id(path):
    return str(path.relative_to(CONFIGS))


@pytest.mark.parametrize("tree", TREES, ids=list(TREES))
def test_a_tree_ships_configs(tree):
    """An empty glob would delete every case below rather than fail one"""
    assert TREES[tree]


@pytest.mark.parametrize("path", DISTILLATION, ids=config_id)
def test_a_distillation_config_loads(path):
    assert distillation.load(path)


@pytest.mark.parametrize("path", NATIVE, ids=config_id)
def test_a_native_config_loads(path):
    assert native.load(path)


@pytest.mark.parametrize("path", TEACHER, ids=config_id)
def test_a_target_extraction_config_loads(path):
    assert teacher_config.load(path)


@pytest.mark.parametrize("path", [path for path in ACCURACY if not is_partial(path)], ids=config_id)
def test_an_accuracy_config_loads(path):
    assert all(
        isinstance(run, evaluation.AccuracyConfig) for run in evaluation.load(path, kind=Evaluation.ACCURACY)
    )


@pytest.mark.parametrize("path", [path for path in KERNELS if not is_partial(path)], ids=config_id)
def test_a_kernels_config_loads(path):
    assert all(
        isinstance(run, evaluation.KernelsConfig) for run in evaluation.load(path, kind=Evaluation.KERNELS)
    )


@pytest.mark.parametrize("path", [path for path in PRODUCTION if not is_partial(path)], ids=config_id)
def test_a_production_config_loads(path):
    runs = evaluation.load(path, kind=Evaluation.PRODUCTION)
    assert all(isinstance(run, evaluation.ProductionConfig) for run in runs)


@pytest.mark.parametrize("path", [*ACCURACY, *KERNELS, *PRODUCTION], ids=config_id)
def test_an_evaluation_config_says_it_is_not_a_run_exactly_when_its_name_starts_with_an_underscore(path):
    """The underscore is what the loading cases above skip, so a run named with one would go unloaded"""
    assert path.read_text().startswith("# Not a run") == is_partial(path)


def test_every_evaluation_config_sits_in_a_tree_of_its_kind():
    """A yaml beside the three trees would be loaded by none of the cases above"""
    assert set(yamls("evaluation")) == {*ACCURACY, *KERNELS, *PRODUCTION}


def runs_of(kind, path):
    return evaluation.load(path, kind=kind)


def every_run():
    """Every run the evaluation trees load, with the kind and the file it came from, base files left out"""
    for kind, paths in (
        (Evaluation.ACCURACY, ACCURACY),
        (Evaluation.KERNELS, KERNELS),
        (Evaluation.PRODUCTION, PRODUCTION),
    ):
        for path in paths:
            if is_partial(path):
                continue
            for run in runs_of(kind, path):
                yield path, run


def test_every_run_name_within_an_evaluation_tree_is_distinct():
    """A name is what a record is labelled with, so two configs of a kind sharing one would land in one group

    A kind keeps its records under a root of its own, so a scored run and a timed one of the same stem may share a name.
    """
    # The card closes a name at launch, so a B300 config and the H200 one it repeats share the name they launch under
    names = [(path.relative_to(CONFIGS).parts[1], run.run.name, on_b300(path)) for path, run in every_run()]
    assert "unnamed" not in [name for _, name, _ in names]
    assert len(names) == len(set(names)), sorted(key for key in set(names) if names.count(key) > 1)


def flag(serve_args, name):
    for position, argument in enumerate(serve_args):
        if argument.startswith(f"{name}="):
            return argument.partition("=")[2]
        if argument == name:
            return serve_args[position + 1]
    return None


def test_every_qwen38_lsa_server_holds_every_key_head_paged_a_token_at_a_time_with_an_eager_prefill():
    """The backend refuses anything else at startup - a config should not get that far

    One card, plain tensor parallel, or decode context parallel across the whole tensor-parallel group, prefilled on
    fa3 and decoded on triton as the image requires of that mesh. A B300 runs no fa3, so it serves triton in both
    phases.
    """
    served = [
        (on_b300(path), run)
        for path, run in every_run()
        if path.parent.name == "qwen38_27b" and "kernels" not in path.parts and run.model.attention == "lsa"
    ]
    assert {run.run.tp for _, run in served} == {1, 2, 4, 8}
    for b300, run in served:
        serve_args = run.server().serve_args
        assert flag(serve_args, "--page-size") == "1", run.run.name
        assert "--disable-prefill-cuda-graph" in serve_args, run.run.name
        if run.run.tp == 1:
            backend = "triton" if b300 else "fa3"
            assert flag(serve_args, "--attention-backend") == backend, run.run.name
            continue
        # Plain tensor parallel splits the key heads over the ranks and shards the index, which no decode group is
        dcp = flag(serve_args, "--dcp-size")
        assert dcp in (None, str(run.run.tp)), run.run.name
        if b300:
            assert flag(serve_args, "--attention-backend") == "triton", run.run.name
        else:
            assert flag(serve_args, "--prefill-attention-backend") == "fa3", run.run.name
            assert flag(serve_args, "--decode-attention-backend") == "triton", run.run.name
        assert "--enable-prefill-cp" not in serve_args, run.run.name


# The knobs each design sets away from the defaults
INDEXER = {
    "lsa_flat": {},
    "lsa_fixed": {"top_level": 1, "node_bucket_size": 4, "node_cluster_block_size": 1024, "beam_width": 2048},
    "lsa_tree": {
        "top_level": 16,
        "node_bucket_size": 4,
        "node_cluster_block_size": 4096,
        "beam_width": 2048,
        "start_nodes": 32768,
    },
    "lsa_flat_k2048": {"top_k": 2048, "top_p": 512},
    "lsa_flat_no_pooling": {"pool_size": None},
    "lsa_tree_descend": {
        "top_level": 16,
        "node_bucket_size": 4,
        "node_cluster_block_size": 4096,
        "beam_width": 2048,
        "start_nodes": 32768,
    },
    "lsa_tree_node_shape": {
        "top_level": 16,
        "node_bucket_size": 4,
        "node_cluster_block_size": 4096,
        "beam_width": 2048,
        "start_nodes": 32768,
        "node_bucket_shape": {"rank": 2, "deviations": 2.0},
    },
}
SERVING = {
    "lsa_tree": {"node_centroid_precision": "fp8"},
    "lsa_tree_descend": {"node_centroid_precision": "fp8", "decode": "descend"},
    "lsa_tree_node_shape": {"node_centroid_precision": "fp8", "shape_basis": "sign"},
}
# The design's input and its map - the key and every head's query summed, through the heavy-mix map re-exported for it
# The one map each model's designs read, by the model's config directory
PROJECTIONS = {
    "glm53_flash": "models/maps/glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt",
    "qwen38_27b": "models/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt",
}


def test_every_prefill_context_parallel_run_splits_a_chunk_contiguously():
    """LSA reads one contiguous span of each request a GPU, and the indexer refuses any other split

    Decode context parallel alone splits no chunk - Qwen's dense mesh shards only its KV pool.
    """
    context_parallel = [run for _, run in every_run() if "--enable-prefill-cp" in run.server().serve_args]
    assert context_parallel
    for run in context_parallel:
        serve_args = run.server().serve_args
        assert serve_args[serve_args.index("--cp-strategy") + 1] == "contiguous", run.run.name
        assert run.server().image in site.load()["images"].values(), run.run.name


def test_every_data_config_is_a_target_extraction_recipe():
    assert set(yamls("data")) == set(TEACHER)


def test_every_pisa_server_captures_its_decode_on_tensor_parallel_alone():
    """A decode step's search is captured but a prefill's is eager, and the plugin refuses context parallel, so a config
    should not get as far as the server to learn either"""
    served = [run for _, run in every_run() if run.model.attention == "pisa"]
    assert served
    for run in served:
        assert "--disable-cuda-graph" not in run.server().serve_args, run.run.name
        if "fa3" in run.server().serve_args:
            assert "--disable-prefill-cuda-graph" in run.server().serve_args, run.run.name
        assert not set(run.server().serve_args) & {"--enable-prefill-cp", "--attn-cp-size", "--dcp-size"}, (
            run.run.name
        )
        assert run.model.settings["selector"] == "pisa", run.run.name


def test_every_served_descending_decode_runs_under_the_decode_cuda_graph_unless_its_node_buckets_are_shaped():
    """The graph reads a mirror of the node buckets, which a shaped node bucket has none of - that design is served eagerly"""
    served = [
        run
        for path, run in every_run()
        if "kernels" not in path.parts
        and run.model.attention == "lsa"
        and run.model.settings["decode"] == "descend"
    ]
    assert served
    for run in served:
        shaped = run.model.settings.get("node_bucket_shape") is not None
        assert ("--disable-cuda-graph" in run.server().serve_args) == shaped, run.run.name
