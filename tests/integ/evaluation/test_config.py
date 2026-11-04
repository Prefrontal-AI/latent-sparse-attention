"""The evaluation schemas, the loader and the sweep - everything that runs without a model."""

import json
import re
import shutil
from pathlib import Path

import pytest

from lsa import site
from lsa.evaluation import config as experiment
from lsa.evaluation.constants import Evaluation

MINIMAL = """
model:
  attention: lsa
  indexer:
    window_size: 512
    sink_size: 4
    top_k: 2048
    index_dim: 64
  serving:
    projections: /maps/glm53_flash.pt
data:
  suite: full
run:
  name: smoke
  image: /images/sglang.img
  concurrency_scale: 2.0
"""


REAL_SITES = site.SITES
IMAGE_NAMES = ("h200", "b300", "early", "papertrail5m", "papertrail10m")


@pytest.fixture(autouse=True)
def fixture_site(tmp_path, monkeypatch):
    """Read the repository's configs against a site that names every image, whatever sites the checkout has"""
    sites = tmp_path / "sites"
    shutil.copytree(REAL_SITES, sites)
    images = "".join(f"  {name}: registry/lsa-test/{name}:1\n" for name in IMAGE_NAMES)
    (sites / "test.yaml").write_text(f"root: /test/root\nimages:\n{images}")
    monkeypatch.setattr(site, "SITES", sites)
    monkeypatch.setattr(site, "_selected", "test")
    return sites


def write(tmp_path, text, name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def load(tmp_path, text=MINIMAL, overrides=None, kind=Evaluation.ACCURACY):
    return experiment.load(write(tmp_path, text), overrides, kind=kind)


def test_a_file_without_a_sweep_is_one_experiment(tmp_path):
    experiments = load(tmp_path)
    assert len(experiments) == 1
    assert experiments[0].name == "smoke"
    assert experiments[0].model.indexer.top_k == 2048
    assert experiments[0].run.concurrency_scale == 2.0


def test_a_file_that_names_no_model_names_none(tmp_path):
    """No schema default stands in for the model, so a base run alone cannot serve one"""
    only = load(tmp_path)[0]
    assert only.model.name == ""
    assert only.model.asset is None
    assert only.model.repo == ""


def test_a_file_that_names_no_records_root_names_none(tmp_path):
    """No schema default stands in for the root either, so a run cannot publish to the shared one unasked"""
    assert load(tmp_path)[0].run.output == ""


def test_a_checkpoint_id_is_taken_as_written(tmp_path):
    """A name holding a slash is a hub id or a path, and no asset is looked for"""
    only = load(tmp_path, MINIMAL.replace("model:\n", "model:\n  name: Qwen/Qwen3.5-4B-Base\n"))[0]
    assert only.model.asset is None
    assert only.model.repo == "Qwen/Qwen3.5-4B-Base"
    assert only.model.checkpoint == "Qwen/Qwen3.5-4B-Base"
    assert only.model.rotary_end == "auto"


def test_a_bare_path_has_no_hub_id(tmp_path):
    only = load(tmp_path, MINIMAL.replace("model:\n", "model:\n  name: /srv/models/some-checkpoint\n"))[0]
    assert only.model.repo == ""


def test_the_indexer_block_is_validated_by_the_design_settings(tmp_path):
    # top_p buckets of bucket_cap keys must be able to fill top_k - the design's own rule, not a second copy of it
    with pytest.raises(ValueError, match="fill top_k"):
        load(tmp_path, MINIMAL.replace("top_k: 2048", "top_k: 40000"))


def test_an_lsa_experiment_without_a_projection_file_is_refused(tmp_path):
    (only,) = load(tmp_path, MINIMAL.replace("  serving:\n    projections: /maps/glm53_flash.pt\n", ""))
    with pytest.raises(ValueError, match="projections"):
        _ = only.model.settings


def test_the_settings_are_the_design_flattened(tmp_path):
    body = load(tmp_path)[0].model.settings
    assert (body["window_size"], body["top_k"], body["index_dim"]) == (512, 2048, 64)
    assert (body["decode"], body["projections"]) == ("flat", "/maps/glm53_flash.pt")


def test_a_block_of_another_kind_is_rejected(tmp_path):
    """A kernel config's data block is not a suite, so read as accuracy it fails rather than being ignored"""
    with pytest.raises(ValueError, match="AccuracyDataConfig has no field \\['phase'\\]"):
        load(tmp_path, MINIMAL.replace("  suite: full\n", "  phase: prefill\n"))


def test_an_unknown_attention_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="model.attention must be one of"):
        load(tmp_path, MINIMAL.replace("  attention: lsa", "  attention: sglang"))


def test_an_unknown_rotary_end_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="rotary_end must be one of"):
        load(tmp_path, MINIMAL.replace("model:\n", "model:\n  rotary_end: middle\n"))


def test_the_tree_a_file_sits_in_is_its_kind(tmp_path):
    for kind in Evaluation.KINDS:
        assert experiment.kind_of(tmp_path / kind / "glm53_flash" / "native.yaml") == kind
    with pytest.raises(ValueError, match="sits in no"):
        experiment.kind_of(tmp_path / "elsewhere" / "native.yaml")


def test_a_rotary_dim_with_no_end_to_sit_at_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="needs an end to sit at"):
        load(tmp_path, MINIMAL.replace("model:\n", "model:\n  rotary_dim: 64\n"))


def test_a_pinned_layout_is_kept_for_the_measurement_to_check(tmp_path):
    only = load(tmp_path, MINIMAL.replace("model:\n", "model:\n  rotary_end: suffix\n  rotary_dim: 64\n"))[0]
    assert (only.model.rotary_end, only.model.rotary_dim) == ("suffix", 64)


def test_a_negative_rotary_dim_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonnegative integer"):
        load(tmp_path, MINIMAL.replace("model:\n", "model:\n  rotary_end: prefix\n  rotary_dim: -8\n"))


def test_an_adapter_can_be_named(tmp_path):
    only = load(tmp_path, MINIMAL.replace("model:\n", "model:\n  adapter: qwen3_5\n"))[0]
    assert only.model.adapter == "qwen3_5"


def test_the_server_is_the_run_s_launch(tmp_path):
    server = load(tmp_path, MINIMAL + "  tp: 4\n")[0].server()
    assert (server.image, server.tp) == ("/images/sglang.img", 4)


def test_the_run_s_environment_reaches_the_server(tmp_path):
    server = load(
        tmp_path, MINIMAL + "  environment:\n    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True\n"
    )[0].server()
    assert server.environment == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert load(tmp_path, MINIMAL)[0].server().environment == {}


SERVED = """  serve:
    context-length: 10500000
    mem-fraction-static: "0.75"
    disable-radix-cache: true
    enable-metrics: true
    watchdog-timeout: null
    reasoning-parser: false
"""


def test_a_scalar_is_a_flag_s_argument_and_true_the_bare_flag_and_null_or_false_no_flag(tmp_path):
    server = load(tmp_path, MINIMAL + SERVED)[0].server()
    assert server.serve_args == [
        "--context-length",
        "10500000",
        "--mem-fraction-static",
        "0.75",
        "--disable-radix-cache",
        "--enable-metrics",
    ]


def test_a_list_is_a_flag_with_several_arguments(tmp_path):
    server = load(tmp_path, MINIMAL + "  serve:\n    cuda-graph-bs: [1, 2, 4]\n")[0].server()
    assert server.serve_args == ["--cuda-graph-bs", "1", "2", "4"]


OVERRIDE = """  serve:
    json-model-override-args: '{"lsa": {"debug": {"check_index": true}}, "text_config": {"rope_parameters": {"factor": 40.0}}}'
"""


def test_a_run_s_model_override_string_merges_with_the_rotation_and_the_settings_block_into_one_flag(
    tmp_path,
):
    from lsa.sglang.launch import with_rotation, with_settings

    server = load(tmp_path, MINIMAL + OVERRIDE)[0].server()
    argv = with_settings(with_rotation(server.serve_args, {"factor": 40.0}), {"top_k": 2048})
    assert argv.count("--json-model-override-args") == 1
    assert json.loads(argv[argv.index("--json-model-override-args") + 1]) == {
        "lsa": {"debug": {"check_index": True}, "top_k": 2048},
        "text_config": {"rope_parameters": {"factor": 40.0}},
    }


def test_a_run_s_model_override_that_disagrees_with_the_launch_is_refused(tmp_path):
    from lsa.sglang.launch import with_rotation

    server = load(tmp_path, MINIMAL + OVERRIDE)[0].server()
    with pytest.raises(ValueError, match="rope_parameters.factor"):
        with_rotation(server.serve_args, {"factor": 8.0})


def test_a_run_s_model_override_must_be_a_json_string_not_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="json-model-override-args"):
        load(tmp_path, MINIMAL + "  serve:\n    json-model-override-args:\n      lsa: {}\n")[0]


def test_a_child_config_sets_overrides_and_removes_one_flag_of_its_parent(tmp_path):
    write(
        tmp_path,
        MINIMAL + SERVED.replace("    watchdog-timeout: null\n", "    watchdog-timeout: 1800\n"),
        "parent.yaml",
    )
    child = write(
        tmp_path,
        "defaults:\n  - parent\n  - _self_\n\nrun:\n  serve:\n"
        "    context-length: 1008000\n    disable-radix-cache: null\n    max-running-requests: 1\n",
        "child.yaml",
    )
    [run] = experiment.load(child, kind=Evaluation.ACCURACY)
    assert run.server().serve_args == [
        "--context-length",
        "1008000",
        "--mem-fraction-static",
        "0.75",
        "--enable-metrics",
        "--watchdog-timeout",
        "1800",
        "--max-running-requests",
        "1",
    ]


def test_a_command_line_override_sets_and_removes_one_flag(tmp_path):
    server = load(
        tmp_path,
        MINIMAL + SERVED,
        ["run.serve.mem-fraction-static=0.7", "run.serve.disable-radix-cache=null", "run.serve.page-size=1"],
    )[0].server()
    assert server.serve_args == [
        "--context-length",
        "10500000",
        "--mem-fraction-static",
        "0.7",
        "--enable-metrics",
        "--page-size",
        "1",
    ]


def test_a_flag_value_that_is_a_mapping_is_refused_saying_what_arrived(tmp_path):
    with pytest.raises(ValueError, match=r"run\.serve\.page-size must be a scalar.*got \{'size': 1\}"):
        load(tmp_path, MINIMAL + "  serve:\n    page-size:\n      size: 1\n")


def test_a_flag_written_with_its_dashes_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"without the leading dashes - got '--page-size'"):
        load(tmp_path, MINIMAL + '  serve:\n    "--page-size": 1\n')


def test_a_launch_the_server_would_refuse_is_rejected_at_load(tmp_path):
    with pytest.raises(ValueError, match="runner must be srun or docker"):
        load(tmp_path, MINIMAL + "  runner: ssh\n")


def test_an_indexer_the_kernels_cannot_run_is_rejected_at_load(tmp_path):
    with pytest.raises(ValueError, match="bucket_cap must be at least twice bucket_size"):
        load(tmp_path, overrides=["model.indexer.bucket_cap=8"])


def test_a_serving_choice_the_design_cannot_honour_is_rejected_at_load(tmp_path):
    with pytest.raises(ValueError, match="decode descend needs a hierarchy"):
        load(tmp_path, overrides=["model.serving.decode=descend"])


def test_a_hierarchy_design_reads_its_per_level_lists_from_yaml(tmp_path):
    text = MINIMAL.replace(
        "    index_dim: 64\n",
        "    index_dim: 64\n    top_level: 2\n    node_bucket_size: [4, 4]\n"
        "    node_cluster_block_size: 64\n    beam_width: [512, 256]\n    start_nodes: 32\n",
    ).replace(
        "    projections: /maps/glm53_flash.pt\n",
        "    projections: /maps/glm53_flash.pt\n    decode: descend\n",
    )
    (only,) = load(tmp_path, text)
    body = only.model.settings
    assert (body["top_level"], body["decode"]) == (2, "descend")
    assert only.model.indexer.at("beam_width", 2) == 256


def test_a_native_served_experiment_never_reads_the_indexer(tmp_path):
    only = load(
        tmp_path, MINIMAL.replace("  attention: lsa", "  attention: native"), ["model.indexer.top_k=128"]
    )[0]
    assert only.model.indexer.top_k == 128
    assert only.model.settings is None


PINNED = "glm-5.3-flash__native__full__2026-09-22__08-15-00"


def pinned_run(tmp_path, suite="full", overrides=()):
    """Load the minimal experiment pinned to one run directory under tmp_path, holding a run of `suite`."""
    run = load(tmp_path, overrides=[f"run.output={tmp_path}", f"run.id={PINNED}", *overrides])[0]
    run_dir = tmp_path / "accuracy" / PINNED
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"run": {"data": {"suite": suite}}}))
    return run, run_dir


def test_an_experiment_directory_holding_another_suite_says_so(tmp_path):
    from lsa.evaluation.run import run_accuracy

    run, _ = pinned_run(tmp_path, suite="ruler")
    with pytest.raises(ValueError, match="already holds a ruler run"):
        run_accuracy(run, force=False)


def write_scored_record(run_dir, lengths=None):
    """Write a record that scored every task of `full`, or only its RULER tasks at the given lengths."""
    from lsa.evaluation.accuracy.data import suites

    plans = suites.SUITES["full"]()
    if lengths:
        plans = [plan for plan in plans if plan.length in lengths or not plan.length]
    tasks = {plan.task: {"n": 4, "scored": 4} for plan in plans}
    (run_dir / "record.json").write_text(json.dumps({"tasks": tasks}))


def test_a_pinned_run_id_is_the_directory_a_finished_run_is_found_in(tmp_path, capsys):
    # The derived id closes on the second a run started, so the pin is the only way back to an old one
    from lsa.evaluation.run import run_accuracy

    run, run_dir = pinned_run(tmp_path)
    write_scored_record(run_dir)
    assert run_accuracy(run, force=False) == run_dir
    assert "nothing to run" in capsys.readouterr().out


def test_a_finished_run_missing_the_lengths_asked_for_is_continued(tmp_path):
    # A one-node run scored up to 1M, and the two-node half asks for what it could not hold
    from lsa.evaluation.run import run_accuracy

    run, run_dir = pinned_run(tmp_path, overrides=["data.lengths=[2000000,5000000,10000000]"])
    write_scored_record(run_dir, lengths=[4096, 32768, 131072, 524288, 1000000])
    with pytest.raises(ValueError, match="generation leaves"):
        run_accuracy(run, force=False)


def test_a_run_with_no_records_root_is_refused_before_a_server_starts(tmp_path):
    from lsa.evaluation.run import run_accuracy

    with pytest.raises(ValueError, match="names no records root"):
        run_accuracy(load(tmp_path)[0], force=False)


GENERATION = """
  generation:
    gpqa_diamond:
      temperature: 1.0
      top_p: 0.95
      top_k: 20
      max_tokens: 163840
    ruler:
      temperature: 0.0
      top_p: 1.0
      max_tokens: 2048
      chat_template_kwargs:
        enable_thinking: false
"""


def with_generation(extra=""):
    return MINIMAL.replace("data:\n", GENERATION.lstrip("\n") + extra + "data:\n")


def test_the_generation_block_is_read_per_dataset(tmp_path):
    request = load(tmp_path, with_generation())[0].model.generation.request()
    assert request["ruler"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["gpqa_diamond"]["chat_template_kwargs"] is None
    assert request["gpqa_diamond"]["top_k"] == 20
    # A dataset left empty is not handed to the runner
    assert "math500" not in request


def test_only_the_datasets_a_run_draws_from_have_to_be_filled_in(tmp_path):
    generation = load(tmp_path, with_generation())[0].model.generation
    assert generation.missing(["gpqa_diamond", "ruler_vt_4096", "ruler_cwe_131072"]) == []
    assert generation.missing(["graphwalks_128k", "graphwalks_1m"]) == [
        "graphwalks.temperature",
        "graphwalks.top_p",
        "graphwalks.max_tokens",
    ]
    assert generation.missing(["math500"]) == [
        "math500.temperature",
        "math500.top_p",
        "math500.max_tokens",
    ]
    # No task list is the whole suite, which draws from every dataset
    assert "aime25.temperature" in generation.missing([])


def test_a_dataset_the_suite_does_not_know_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, with_generation("    reasoning:\n      temperature: 1.0\n"))


def test_only_a_scored_model_says_how_it_is_sampled(tmp_path):
    """Sampling is the accuracy suite's alone, so a timing schema has no field for it"""
    with pytest.raises(ValueError, match="EvaluatedModelConfig has no field \\['generation'\\]"):
        experiment.load(
            write(tmp_path, "model:\n  generation:\n    ruler:\n      temperature: 0.0\n"),
            kind=Evaluation.PRODUCTION,
        )


def test_a_served_run_with_no_generation_block_is_refused_before_a_server_starts(tmp_path):
    from lsa.evaluation.run import run_accuracy

    only = load(tmp_path, overrides=[f"run.output={tmp_path}"])[0]
    only.run.gpu = "NVIDIA H200"
    assert "ruler.temperature" in only.model.generation.missing(only.data.tasks)
    with pytest.raises(ValueError, match="generation leaves"):
        run_accuracy(only, force=False)


REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "evaluation"
UNNAMED = experiment.RunConfig.name


def on_b300(path):
    """A config for the B300 - its name closes with the card, or opens with it in the files not yet renamed"""
    return path.stem.endswith("_b300") or path.stem.startswith("b300")


def repo_runs(kind=None):
    for path in sorted(REPO_CONFIGS.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        if kind is None or experiment.kind_of(path) == kind:
            for run in experiment.load(path):
                yield path, run


def test_every_named_config_in_the_repo_names_its_model_and_its_records_root():
    for path, run in repo_runs():
        # A file with no name is a base or an overlay, checked through every named configuration using it
        if run.name != UNNAMED:
            assert run.model.name, path
            assert run.run.output, path


def test_every_task_list_in_the_repo_is_one_its_suite_has():
    """A task a suite does not hold used to drop out silently, and a smoke run lost 35 of its 42"""
    from lsa.evaluation.accuracy.data import suites

    for path, run in repo_runs(Evaluation.ACCURACY):
        assert suites.select(run.data.suite, run.data.tasks or None, run.data.lengths or None), path


def test_every_served_config_in_the_repo_states_how_its_model_is_sampled():
    for path, run in repo_runs(Evaluation.ACCURACY):
        # A file with no name is a base or an overlay, checked through every named configuration using it
        if run.name != UNNAMED:
            assert run.model.generation.missing(run.data.tasks) == [], path


def test_every_config_serving_the_moe_on_deep_gemm_exports_its_warmup_and_layout():
    """sglang reads these two variables at import, and the launcher no longer reads the serve flags to guess them"""
    for path, run in repo_runs():
        serve = run.run.serve
        if serve.get("moe-runner-backend") == "deep_gemm":
            assert run.run.environment.get("SGLANG_JIT_DEEPGEMM_FAST_WARMUP") == "1", path
            assert run.run.environment.get("SGLANG_DEEPGEMM_STANDARD_LAYOUT") == "masked", path


def real_sites():
    return sorted(path.stem for path in REAL_SITES.glob("*.yaml"))


@pytest.mark.parametrize("name", real_sites())
def test_every_config_in_the_repo_runs_the_default_image_or_one_of_the_first_battery(name):
    """Models and layouts once ran five builds between them, so comparing two runs could also compare two builds

    The first battery's files name their builds because a cell's number belongs to the build that served it.
    A site that leaves its images empty serves nothing, and is held to the next test instead.
    """
    images = site.load(name)["images"]
    if not all(images.values()):
        # Nothing to compare against => the next test holds such a site to the names it asks for
        return
    first_battery = {images["early"], images["papertrail5m"], images["papertrail10m"]}
    runs = [(path, run) for path, run in repo_runs_at(name)]
    found = {run.run.image for path, run in runs if not on_b300(path)}
    assert found == {images["h200"]} | first_battery, found
    early = {path.stem for path, run in runs if run.run.image in first_battery}
    assert all(stem.startswith(("lsa_tree_cp8_", "lsa_flat_cp8_", "native_cp8_")) for stem in early), early
    # A B300 cannot run the Hopper image, so its configurations, context-parallel ones included, run the one B300 image
    b300 = {run.run.image for path, run in runs if on_b300(path)}
    assert b300 == {images["b300"]}, b300


def repo_runs_at(name):
    for path in sorted(REPO_CONFIGS.rglob("*.yaml")):
        if not path.name.startswith("_"):
            for run in experiment.load(path, [f"site={name}"]):
                yield path, run


def test_every_serving_config_in_the_repo_names_its_image_only_through_the_site():
    """A site may leave its images empty, so no file can spell one, and every name it asks for is one of the site's"""
    named = set()
    for path in sorted(REPO_CONFIGS.rglob("*.yaml")):
        for line in path.read_text().splitlines():
            found = re.match(r"\s*image:\s*(\S.*?)\s*(#.*)?$", line)
            if found:
                value = found.group(1).strip("'\"")
                match = re.fullmatch(r"\$\{site\.images\.(\w+)\}", value)
                assert match, f"{path}: {line.strip()}"
                named.add(match.group(1))
    assert named
    for name in real_sites():
        assert named <= set(site.load(name)["images"]), name


def test_every_glm_config_reasons_at_one_budget_and_leaves_longbench_room_for_it():
    from lsa.evaluation.accuracy.data import longbench_card

    for path, run in repo_runs(Evaluation.ACCURACY):
        if run.name == UNNAMED or run.model.name != "glm53_flash":
            continue
        # A file that reruns a first-battery cell keeps the budgets that battery sampled at
        if path.stem.endswith("_first"):
            continue
        generation = run.model.generation
        budgets = {
            generation.gpqa_diamond.max_tokens,
            generation.longbench2.max_tokens,
            generation.graphwalks.max_tokens,
        }
        assert budgets == {163840}, path
        # A RULER-only run, or one whose task list leaves LongBench out, draws no LongBench prompt
        if run.data.suite == "ruler" or (run.data.tasks and "longbench2" not in run.data.tasks):
            continue
        serve_args = run.server().serve_args
        context = int(serve_args[serve_args.index("--context-length") + 1])
        # The prepared prompts and the budget have to fit the shortest server a run is served on
        assert longbench_card().context + generation.longbench2.max_tokens <= context, path


PISA = """
model:
  attention: pisa
  pisa:
    keep_count: 40
data:
  suite: full
run:
  name: smoke
  image: /images/sglang.img
"""


def test_a_pisa_experiment_needs_no_projection_file_and_arms_the_plugin_with_its_own_settings(tmp_path):
    (only,) = load(tmp_path, PISA)
    body = only.model.settings
    assert body["selector"] == "pisa"
    assert (body["block_size"], body["keep_count"], body["branching"]) == (64, 40, 2)
    assert "projections" not in body


def test_an_lsa_experiment_serves_lsa_settings_whatever_its_pisa_block_holds(tmp_path):
    body = load(tmp_path)[0].model.settings
    assert "selector" not in body


def test_a_pisa_block_that_is_not_a_pyramid_is_refused_by_the_design_settings(tmp_path):
    with pytest.raises(ValueError, match="keep_count"):
        load(tmp_path, PISA.replace("keep_count: 40", "keep_count: 2"))


def test_a_pisa_run_is_served_and_timed():
    model = experiment.EvaluatedModelConfig(name="", attention="pisa")
    run = experiment.RunConfig(image="/images/sglang.img")
    assert experiment.ProductionConfig(model=model, run=run).model.attention == "pisa"
    assert experiment.KernelsConfig(model=model, run=run).model.settings["selector"] == "pisa"


@pytest.mark.parametrize(
    ("config", "recorded"),
    [
        ("glm53_flash/native_tp8.yaml", "block_sparse"),
        ("glm53_flash/lsa_flat_tp8.yaml", "lsa"),
        ("glm53_flash/pisa_tp8.yaml", "pisa"),
        ("qwen38_27b/full_tp1.yaml", "full"),
        ("qwen38_27b/lsa_flat_tp1.yaml", "lsa"),
        ("qwen38_27b/pisa_tp1.yaml", "pisa"),
    ],
)
def test_a_record_files_pisa_as_its_own_method_never_as_the_checkpoints_kind(config, recorded):
    [run] = experiment.load(REPO_CONFIGS / "accuracy" / config)
    assert run.model.recorded_attention == recorded


def mesh_of(run):
    """The mesh label a production config serves on, from its tensor, pipeline and context-parallel settings."""
    args = run.server().serve_args

    def size(flag):
        return int(args[args.index(flag) + 1]) if flag in args else 1

    cp = size("--attn-cp-size")
    return f"tp{run.run.tp // cp}-cp{cp}-dcp{size('--dcp-size')}-pp{run.run.pp}-ep1"


def mesh_token(label):
    """Name a mesh as a run's name does - `cp8`, `dcp8` or `tp8` by the widest context axis, `pp2` closing it."""
    tp, cp, dcp, pp = (
        int(size) for size in re.fullmatch(r"tp(\d+)-cp(\d+)-dcp(\d+)-pp(\d+)-ep\d+", label).groups()
    )
    head = f"cp{cp}" if cp > 1 else f"dcp{dcp}" if dcp > 1 else f"tp{tp}"
    return head + (f"pp{pp}" if pp > 1 else "")


def test_a_production_config_is_named_by_its_model_method_and_mesh():
    """The launch appends the card, so a served run's name is `<model>_<method>_<mesh>_<gpu>`."""
    named = [(path, run) for path, run in repo_runs("production") if run.model.name and run.name != UNNAMED]
    assert named
    for path, run in named:
        assert run.name.startswith(f"{path.parent.name}_"), path
        assert re.search(rf"_{mesh_token(mesh_of(run))}(_[a-z0-9]+)?$", run.name), path
        assert "production" not in run.name, path


def test_a_production_config_is_named_for_its_file():
    """A record id reads back to its file - the stem is the run's name less the model, and the launch appends the card."""
    named = [(path, run) for path, run in repo_runs("production") if run.model.name and run.name != UNNAMED]
    assert named
    for path, run in named:
        assert run.name == f"{path.parent.name}_{path.stem.removesuffix('_b300')}", path


def test_an_accuracy_config_is_named_for_its_file():
    """A record id reads back to its file - the stem is the run's name less the model, and the launch appends the card."""
    named = [(path, run) for path, run in repo_runs("accuracy") if run.model.name and run.name != UNNAMED]
    assert named
    for path, run in named:
        assert run.name == f"{path.parent.name}_{path.stem.removesuffix('_b300')}", path


def test_a_task_the_suite_does_not_offer_is_refused_at_load():
    """A pinned task list overridden onto another suite used to fail only after the server booted"""
    with pytest.raises(ValueError, match=r"data\.suite 'ruler'.*no task \['gpqa_diamond'\].*offers"):
        experiment.AccuracyDataConfig(suite="ruler", tasks=["gpqa_diamond", "ruler_cwe_4096"])


def test_a_length_the_suite_does_not_offer_is_refused_at_load():
    with pytest.raises(ValueError, match=r"no length \[12345\].*offers"):
        experiment.AccuracyDataConfig(suite="ruler", lengths=[4096, 12345])


def test_a_subset_of_the_suite_is_accepted():
    data = experiment.AccuracyDataConfig(suite="ruler", tasks=["ruler_cwe_4096"], lengths=[4096])
    assert data.tasks == ["ruler_cwe_4096"]


PAPER_RULER = ["data.suite=ruler", "data.tasks=[ruler]", "data.lengths=[4096,32768]"]


def test_the_ruler_group_names_every_ruler_task_at_the_lengths_asked(tmp_path):
    from lsa.evaluation.accuracy.data import suites

    run = load(tmp_path, overrides=PAPER_RULER)[0]
    selected = suites.select(run.data.suite, run.data.tasks, run.data.lengths)
    assert len(selected) == 26
    assert {plan.length for plan in selected} == {4096, 32768}
    assert all(task.startswith("ruler_") for task in run.data.tasks)


def test_the_ruler_group_replaces_the_tasks_a_config_pins():
    path = Path("configs/evaluation/accuracy/qwen38_27b/lsa_flat_tp1.yaml")
    run = experiment.load(path, PAPER_RULER, kind=Evaluation.ACCURACY)[0]
    assert "gpqa_diamond" not in run.data.tasks
    assert len(run.data.tasks) > 26


def test_a_task_beside_the_ruler_group_is_kept_once(tmp_path):
    run = load(tmp_path, overrides=["data.tasks=[ruler_cwe_4096,ruler]"])[0]
    assert run.data.tasks.count("ruler_cwe_4096") == 1
    assert run.data.tasks[0] == "ruler_cwe_4096"


def test_a_dry_run_prints_the_experiment_it_resolved(tmp_path, monkeypatch, capsys):
    from lsa import cli

    def refuse(*args, **kwargs):
        raise AssertionError("a dry run built an executor")

    monkeypatch.setattr("lsa.slurm.executor", refuse)
    path = write(tmp_path, MINIMAL)
    named = ["model.name=Qwen/Qwen3.8-27b", "run.output=results"]
    cli.main(["evaluate", "accuracy", str(path), *PAPER_RULER, *named, "--dry-run", "--cpus", "12"])
    printed = capsys.readouterr().out
    assert "smoke: attention lsa, suite ruler" in printed
    assert "tasks (26): ruler_niah_single_1_4096" in printed
    assert "and 0 more" not in printed
    assert "lengths: [4096, 32768]" in printed
    assert "server: srun --overlap" in printed
    assert "--container-image=/images/sglang.img" in printed
    assert "sglang serve --model-path" in printed
    assert "resources: " in printed
    assert "gpus_per_node 8" in printed
    assert "cpus_per_task 12" in printed


def test_a_dry_run_without_an_image_prints_the_experiment_and_says_so(tmp_path, monkeypatch, capsys):
    from lsa import cli

    monkeypatch.setattr(
        "lsa.slurm.executor", lambda *args, **kwargs: pytest.fail("a dry run built an executor")
    )
    path = write(tmp_path, MINIMAL.replace("  image: /images/sglang.img\n", ""))
    named = ["model.name=Qwen/Qwen3.8-27b", "run.output=results"]
    cli.main(["evaluate", "accuracy", str(path), *PAPER_RULER, *named, "--dry-run"])
    printed = capsys.readouterr().out
    assert "tasks (26): ruler_niah_single_1_4096" in printed
    assert f"server: (not resolved in site {site.current()} - run.image is not named" in printed
    assert "resources: " in printed
    with pytest.raises(SystemExit, match="run.image is not named"):
        cli.main(["evaluate", "accuracy", str(path), *PAPER_RULER, *named, "--cpus", "12"])


@pytest.mark.parametrize(
    ("kind", "config", "summary"),
    [
        ("kernels", "configs/evaluation/kernels/glm53_flash/lsa_flat.yaml", ", kernels "),
        (
            "production",
            "configs/evaluation/production/glm53_flash/lsa_flat_cp8.yaml",
            ", production, lengths ",
        ),
    ],
)
def test_a_dry_run_of_a_kind_that_is_never_pooled_prints_it(kind, config, summary, monkeypatch, capsys):
    from lsa import cli

    monkeypatch.setattr(
        "lsa.slurm.executor", lambda *args, **kwargs: pytest.fail("a dry run built an executor")
    )
    cli.main(["evaluate", kind, config, "--dry-run"])
    printed = capsys.readouterr().out
    assert summary in printed
    assert "resources: " in printed


def test_a_long_task_list_is_counted_after_its_first_forty(tmp_path):
    from lsa.evaluation import run

    experiment_ = load(tmp_path, overrides=["data.suite=ruler"])[0]
    tasks_line = run.describe(experiment_, {"gpus_per_node": 1}).splitlines()[1]
    assert tasks_line.startswith("  tasks (")
    assert tasks_line.count("ruler_") == 40
    assert ", and " in tasks_line
    assert tasks_line.endswith(" more")
