"""The kernel record from raw harness files: a prefill window, decode probes, the rebuild, the table.

The raw files are the shape `lsa.evaluation.performance.kernels.measure.layer` writes. They are built here rather than
captured from a run, because a captured one is 65 MiB of trials and what is being checked is the
arithmetic over them.
"""

import contextlib
import io
import json
import sys

import pytest

from lsa.evaluation import config as experiment_config
from lsa.evaluation import run
from lsa.evaluation.config import (
    EvaluatedModelConfig,
    KernelsConfig,
    KernelsDataConfig,
    RunConfig,
)
from lsa.evaluation.constants import Evaluation
from lsa.evaluation.performance import telemetry
from lsa.evaluation.performance.kernels import harness, record, show
from lsa.evaluation.performance.kernels.rebuild import main as rebuild_main
from lsa.sglang.settings.design import IndexerConfig, ServingChoices
from tests.helpers.evaluation.fakes import raw, sample, write


@pytest.fixture
def meta():
    return {
        "note": "n",
        "tag": "",
        "started": "2026-09-17T10:00:00",
        "date": "2026-09-17T11:00:00",
        "seconds": 3600.0,
        "model": "zai-org/GLM-5.3-Flash",
        "name": "native",
        "image": "lmsysorg/sglang:v0.5.19",
    }


# The one maintenance case a native layer has, and the design's heaviest
CASE = "pool_close"
CLUSTER = "cluster_block_close"


def test_a_prefill_window_becomes_one_point(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__native__prefill__2026-09-17"
    write(run_dir, "262144-253952.json", raw(253952, module_ms=10.0))
    write(run_dir, "262144-262144.json", raw(262144, module_ms=30.0, growth=True))
    write(run_dir, "131072.json", raw(131072, module_ms=8.0))

    built = record.build(run_dir.name, meta, run_dir, [131072, 262144, 524288], failed=[524288])

    assert built["schema"] == "lsa-kernels/1"
    assert built["run"]["phase"] == "prefill"
    assert built["run"]["commits"] == {"lsa": "e21842e"}
    assert built["run"]["name"] == "native"
    assert built["setup"]["window_chunks"] == 32
    assert built["setup"]["image"] == "lmsysorg/sglang:v0.5.19"
    # The checkpoint directory says which copy was loaded, and only the path keeps it
    assert built["model"]["name"] == "zai-org/GLM-5.3-Flash"
    assert built["model"]["label"] == "glm-5.3-flash"
    assert built["model"]["path"] == "/models/glm53flash-real5"
    assert sorted(built["points"]) == ["131072", "262144"]

    point = built["points"]["262144"]
    assert [row["end"] for row in point["chunks"]] == [253952, 262144]
    assert [row["growth"] for row in point["chunks"]] == [False, True]
    assert point["growth_chunks"] == 1
    assert point["ordinary_ms"] == 10.5
    assert point["growth_ms"] == 30.5
    assert point["chunks"][0]["module_ms"] == 10.5
    assert point["chunks"][1]["instrumented_ms"] == 30.5
    assert point["chunks"][1]["components_ms"]["topk attention"] == 1.0
    assert point["window"] == [253952, 262144]
    # Plain trials: 10, 11, 30, 31
    assert point["module_ms"]["mean"] == 20.5
    assert point["module_ms"]["n"] == 4
    assert point["components_ms"]["index q scan"] == 1.0
    assert point["memory"]["index_bytes"] == 100
    assert point["memory"]["working_bytes"] == 10 * 2**30
    assert point["checks"]["topk_exact"] == 0
    assert point["checks"]["output_exact"]
    assert point["gpu_uuid"] == "GPU-abc"
    assert point["samples"] == ["samples/262144-253952.json", "samples/262144-262144.json"]
    assert built["failed"] == [524288]


def test_decode_probes_group_into_cases(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__native__decode__2026-09-17"
    write(run_dir, "131072-131071.json", raw(131071, "native", "decode", module_ms=2.0))
    write(
        run_dir,
        "131072-131072.json",
        raw(131072, "native", "decode", module_ms=4.0, case=CASE),
    )
    write(run_dir, "131072-131073.json", raw(131073, "native", "decode", module_ms=2.0))
    meta["name"] = "native"

    built = record.build("x", meta, run_dir, [131072])

    assert built["run"]["phase"] == "decode"
    assert built["run"]["name"] == "native"
    assert built["setup"]["cuda_graph"]
    assert built["setup"]["offsets"] == [-1, 0, 1]

    point = built["points"]["131072"]
    assert [row["offset"] for row in point["tokens"]] == [-1, 0, 1]
    assert [row["case"] for row in point["tokens"]] == ["ordinary", CASE, "ordinary"]
    assert point["tokens"][1]["graph_ms"] == 0.4
    assert sorted(point["cases"]) == sorted([CASE, "ordinary"])
    assert point["cases"]["ordinary"]["tokens"] == 2
    assert point["cases"]["ordinary"]["graph_ms"] == 0.2
    assert point["cases"][CASE]["graph_ms"] == 0.4
    assert point["cases"]["ordinary"]["metadata_ms"] == 0.7
    # The pool-closing token costs 0.2 ms more, once every four tokens
    assert point["maintenance_ms"] == {CASE: round(0.2 / 4, 5)}
    assert point["amortized_ms"] == round(0.2 + 0.2 / 4, 4)
    assert point["graph_ms"]["n"] == 3
    assert point["checks"]["graph_validated"]
    assert point["checks"]["topk_exact"] == 0
    assert point["memory"]["index_bytes"] == 100
    assert show.primary(point, "decode") == 0.2


def test_a_free_indexer_replay_gives_the_indexer_s_graph_time_per_repeat(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__native__decode__2026-09-17"
    token = raw(131072, "native", "decode", module_ms=2.0)
    # Two repeats whose full replays drift together - a difference of means and of medians of each would mix them
    token["graph_comparison"]["samples"] = [
        dict(sample("graph", full), repeat=repeat) for repeat, full in enumerate((0.30, 0.50))
    ] + [dict(sample("free_indexer", free), repeat=repeat) for repeat, free in enumerate((0.10, 0.40))]
    token["graph_comparison"]["samples"] += [
        dict(sample("indexer_only", alone), repeat=repeat) for repeat, alone in enumerate((0.25, 0.15))
    ]
    write(run_dir, "131072-131072.json", token)

    point = record.build("x", meta, run_dir, [131072])["points"]["131072"]

    assert point["graph_ms"]["mean"] == pytest.approx(0.4)
    assert point["free_indexer_graph_ms"]["mean"] == pytest.approx(0.25)
    assert point["indexer_only_graph_ms"]["mean"] == pytest.approx(0.2)
    assert point["graph_indexer_ms"] == pytest.approx(0.15)
    assert point["cases"]["ordinary"]["graph_indexer_ms"] == pytest.approx(0.15)
    assert point["tokens"][0]["graph_indexer_ms"] == pytest.approx(0.15)


def test_a_decode_timed_without_a_free_indexer_has_no_graph_indexer_time(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__native__decode__2026-09-17"
    write(run_dir, "131072-131072.json", raw(131072, "native", "decode", module_ms=2.0))

    point = record.build("x", meta, run_dir, [131072])["points"]["131072"]

    assert point["free_indexer_graph_ms"] is None
    assert point["indexer_only_graph_ms"] is None
    assert point["graph_indexer_ms"] is None


def test_design_decode_probes_group_into_the_design_s_cases(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__lsa-flat__decode__2026-09-17"
    write(run_dir, "131072-131071.json", raw(131071, "lsa", "decode", module_ms=2.0))
    write(run_dir, "131072-131072.json", raw(131072, "lsa", "decode", module_ms=4.0, case=CLUSTER))
    write(run_dir, "131072-131073.json", raw(131073, "lsa", "decode", module_ms=2.0))
    meta["name"] = "lsa-flat"

    built = record.build("x", meta, run_dir, [131072])

    assert built["setup"]["config"] == {"top_level": 0, "top_k": 4096}
    point = built["points"]["131072"]
    assert [row["case"] for row in point["tokens"]] == ["ordinary", CLUSTER, "ordinary"]
    assert point["tokens"][1]["metadata_components_ms"]["index k cluster"] == 1.0
    # The clustering token costs 0.2 ms more, once every window_size keys the harness wrote
    assert point["maintenance_ms"] == {CLUSTER: round(0.2 / 512, 5)}
    assert point["amortized_ms"] == round(0.2 + 0.2 / 512, 4)
    assert point["memory"]["index_bytes"] == 50
    assert point["checks"]["selection_exact"]


def test_a_decode_point_surfaces_the_most_slots_a_graph_replay_selected_that_eager_did_not(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__lsa-tree-descend__decode__2026-09-17"
    write(run_dir, "131072-131071.json", raw(131071, "lsa", "decode", graph_differing=3))
    write(run_dir, "131072-131072.json", raw(131072, "lsa", "decode", case=CLUSTER, graph_differing=7))
    write(run_dir, "131072-131073.json", raw(131073, "lsa", "decode", graph_differing=0))

    built = record.build("x", meta, run_dir, [131072])

    assert built["points"]["131072"]["checks"]["graph_selection_differing_slots"] == 7
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        show.show(built)
    assert "graph slots differing 7" in out.getvalue()


def test_a_tree_decode_with_no_graph_is_read_by_its_eager_time(tmp_path, meta):
    """A descending decode of shaped node buckets runs eagerly in serving, so its amortized token is eager and its record has no graph."""
    run_dir = tmp_path / "glm-5.3-flash__lsa-tree-descend__decode__2026-09-17"
    write(run_dir, "131072-131071.json", raw(131071, "lsa", "decode", module_ms=2.0, graph=False))
    write(
        run_dir, "131072-131072.json", raw(131072, "lsa", "decode", module_ms=4.0, case=CLUSTER, graph=False)
    )

    point = record.build("x", meta, run_dir, [131072])["points"]["131072"]

    assert point["cases"]["ordinary"]["graph_ms"] is None
    assert point["cases"]["ordinary"]["eager_ms"] == 2.5
    assert point["maintenance_ms"] == {CLUSTER: round(2.0 / 512, 5)}
    assert point["amortized_ms"] == round(2.5 + 2.0 / 512, 4)
    assert show.primary(point, "decode") == 2.5
    # Nothing was replayed, so there is no graph verdict - not a failed one
    assert point["checks"]["graph_validated"] is None
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        show.show(record.build("x", meta, run_dir, [131072]))
    assert "graph none (eager)" in out.getvalue()


def test_a_rerun_with_a_differing_row_is_not_exact(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__lsa-flat__prefill__2026-09-17"
    differing = raw(131072, "lsa", differing=3)
    write(run_dir, "131072.json", differing)

    built = record.build(run_dir.name, meta, run_dir, [131072])
    checks = built["points"]["131072"]["checks"]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        show.show(built)

    assert checks["selection_exact"] is False
    assert checks["rows_differing"] == 3
    assert "selection DIFFERS (3 rows differ)" in out.getvalue()


def test_a_run_timed_without_its_checks_has_no_verdict(tmp_path, meta):
    """`--time-only` validates nothing, so its point is timed and says nothing about exactness."""
    run_dir = tmp_path / "glm-5.3-flash__lsa-flat__prefill__2026-09-30"
    timed_only = raw(131072, "lsa", module_ms=40.0)
    timed_only["validation"] = {"time_only": True}
    write(run_dir, "131072.json", timed_only)

    point = record.build(run_dir.name, meta, run_dir, [131072])["points"]["131072"]

    assert point["module_ms"]["mean"] > 0
    names = ("output_exact", "state_exact", "scores_exact", "topk_exact", "selection_exact")
    assert [point["checks"][name] for name in names] == [None] * len(names)


@pytest.mark.parametrize("exact", [True, False])
def test_a_decode_timed_without_its_selection_check_still_reports_its_graph_verdict(tmp_path, meta, exact):
    """`--time-only` skips the selection check, not the bitwise replay check - so the graph has a verdict."""
    run_dir = tmp_path / "glm-5.3-flash__pisa__decode__2026-10-21"
    timed_only = raw(131072, "lsa", "decode", module_ms=2.0)
    timed_only["validation"] = {"time_only": True}
    timed_only["graph_comparison"]["reference_order_output_exact"] = exact
    write(run_dir, "131072-131072.json", timed_only)

    checks = record.build(run_dir.name, meta, run_dir, [131072])["points"]["131072"]["checks"]

    assert checks["topk_exact"] is None
    assert checks["graph_validated"] is exact


def test_a_run_whose_checks_did_not_apply_reports_none_rather_than_a_pass(tmp_path, meta):
    """A layer with no index has no scores and no selection - calling that a pass hides a check that never ran."""
    run_dir = tmp_path / "glm-5.3-flash__none__prefill__2026-09-18"
    write(run_dir, "131072.json", raw(131072, "none", module_ms=40.0))

    built = record.build(run_dir.name, meta, run_dir, [131072])
    point = built["points"]["131072"]

    assert point["checks"]["topk_exact"] is None
    assert point["checks"]["scores_exact"] is None
    assert point["checks"]["output_exact"]
    # The model's index buffer is allocated but a layer without an indexer never reads it
    assert point["memory"]["index_bytes"] == 0
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        show.show(built)
    assert "topk n/a" in out.getvalue()


def test_a_ladder_with_no_samples_says_so(tmp_path, meta):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        record.build("empty", meta, run_dir, [131072])


def test_a_raw_file_of_another_schema_is_refused(tmp_path):
    run_dir = tmp_path / "wrong"
    body = raw(131072)
    body["schema"] = "something-else/1"
    write(run_dir, "131072.json", body)
    with pytest.raises(ValueError, match="expected"):
        record.read_samples(run_dir, 131072)


def test_rebuild_keeps_what_only_the_launcher_knew(tmp_path, meta, monkeypatch):
    run_dir = tmp_path / "glm-5.3-flash__native__prefill__2026-09-16"
    write(run_dir, "131072.json", raw(131072))
    harness.write(run_dir, run_dir.name, meta, [131072, 262144], [262144])

    monkeypatch.setattr(sys, "argv", ["lsa records rebuild", str(run_dir)])
    rebuild_main()
    built = json.loads((run_dir / "record.json").read_text())

    assert built["run"]["id"] == run_dir.name
    assert built["run"]["note"] == "n"
    assert built["run"]["name"] == "native"
    assert built["setup"]["image"] == "lmsysorg/sglang:v0.5.19"
    assert list(built["points"]) == ["131072"]
    assert built["failed"] == [262144]
    assert record.contexts_of(run_dir) == [131072]


def timed(
    attention="native", runner="srun", image="/images/sglang.img", tp=1, serving=None, indexer=None, **data
):
    """A kernel run of GLM-5.3-Flash, its data block as given and its launch at one card."""
    return KernelsConfig(
        model=EvaluatedModelConfig(
            name="zai-org/GLM-5.3-Flash",
            attention=attention,
            serving=ServingChoices(**(serving or {})),
            indexer=IndexerConfig(**(indexer or {})),
        ),
        data=KernelsDataConfig(**{"checkpoint": "/models/cut", "prompt": "/prompts/p.txt", **data}),
        run=RunConfig(image=image, runner=runner, tp=tp),
    )


def test_a_variant_with_no_indexer_is_asked_for_by_name(tmp_path):
    """`kda` is a different layer entirely."""
    argv = harness.harness_args(timed("full"), tmp_path / "indexer.json", 262144, tmp_path / "o.json")
    assert argv[argv.index("--indexer") + 1] == "full"
    # Settings may have been written for the run, and a variant with no indexer must not be handed them
    assert "--indexer-config" not in argv

    argv = harness.harness_args(timed("kda", layer=4), None, 262144, tmp_path / "o.json")
    assert argv[argv.index("--indexer") + 1] == "kda"
    assert argv[argv.index("--layer") + 1] == "4"


def test_the_sparse_attention_kernels_are_the_run_s_to_name(tmp_path):
    """The served meshes read a selection on flashmla_sparse and fa3, and tilelang stays the default."""
    argv = harness.harness_args(timed("native"), None, 262144, tmp_path / "o.json")
    assert argv[argv.index("--dsa-prefill-backend") + 1] == "tilelang"
    assert argv[argv.index("--dsa-decode-backend") + 1] == "tilelang"

    served = timed("native", prefill_backend="flashmla_sparse", decode_backend="fa3")
    argv = harness.harness_args(served, None, 262144, tmp_path / "o.json")
    assert argv[argv.index("--dsa-prefill-backend") + 1] == "flashmla_sparse"
    assert argv[argv.index("--dsa-decode-backend") + 1] == "fa3"


@pytest.mark.parametrize("attention", ["full", "kda", "gdn", "native", "lsa", "pisa"])
def test_a_kernel_record_is_labelled_by_its_run_name_whatever_path_it_timed(attention, tmp_path, monkeypatch):
    """The viewer lists a record by `run.name`, so `native` and `full` runs carry their own name like the rest."""
    labelled = []
    monkeypatch.setattr(
        harness,
        "run_kernels",
        lambda _experiment, _settings, given, run_dir: labelled.append(given) or run_dir,
    )
    monkeypatch.setattr(run, "written_settings", lambda _launch: None)
    monkeypatch.setattr(run, "launch_meta", lambda *_: {})
    kernels = timed(attention, serving={"projections": "/maps/glm.pt"})
    kernels.run.name, kernels.run.output, kernels.run.gpu = "a-name", str(tmp_path), "h200"
    run.run_kernels(kernels, force=False)
    assert labelled == ["a-name_h200"]


def test_the_design_is_timed_with_its_written_settings(tmp_path):
    settings = tmp_path / "indexer.json"
    argv = harness.harness_args(timed("lsa", phase="prefill"), settings, 262144, tmp_path / "o.json")

    assert argv[argv.index("--indexer") + 1] == "lsa"
    assert argv[argv.index("--indexer-config") + 1] == str(settings)
    with pytest.raises(ValueError, match="written settings"):
        harness.harness_args(timed("lsa"), None, 262144, tmp_path / "o.json")


def test_pisa_is_timed_with_its_written_settings(tmp_path):
    settings = tmp_path / "indexer.json"
    argv = harness.harness_args(timed("pisa", phase="decode"), settings, 262144, tmp_path / "o.json")

    assert argv[argv.index("--indexer") + 1] == "pisa"
    assert argv[argv.index("--indexer-config") + 1] == str(settings)
    # Its decode step is captured in a graph, so the harness compares the replay as it does LSA's
    assert "--graph-compare" in argv
    with pytest.raises(ValueError, match="pisa kernel run"):
        harness.harness_args(timed("pisa"), None, 262144, tmp_path / "o.json")


def test_a_card_with_a_rotation_hands_it_to_the_harness_as_the_server_reads_it(tmp_path):
    """Qwen3.8's card carries YaRN, and a checkpoint cut from the model has only the plain rotation."""
    card = KernelsConfig(
        model=EvaluatedModelConfig(name="qwen38_27b", attention="full"),
        data=KernelsDataConfig(checkpoint="/models/cut", prompt="/prompts/p.txt"),
        run=RunConfig(image="/images/sglang.img"),
    )
    argv = harness.harness_args(card, None, 262144, tmp_path / "o.json")
    override = json.loads(argv[argv.index("--model-override") + 1])
    assert override["text_config"]["rope_parameters"]["rope_type"] == "yarn"
    assert override["text_config"]["rope_parameters"]["factor"] == 40.0

    argv = harness.harness_args(timed("full"), None, 262144, tmp_path / "o.json")
    assert "--model-override" not in argv


@pytest.mark.parametrize(
    ("indexer", "serving", "graph"),
    [
        ({}, {}, True),
        (
            {"top_level": 1, "node_bucket_size": 4, "node_cluster_block_size": 1024, "beam_width": 2048},
            {},
            True,
        ),
        (
            {"top_level": 16, "node_bucket_size": 4, "node_cluster_block_size": 4096, "beam_width": 2048},
            {"decode": "descend"},
            True,
        ),
        (
            {
                "top_level": 16,
                "node_bucket_size": 4,
                "node_cluster_block_size": 4096,
                "beam_width": 2048,
                "node_bucket_shape": {"rank": 2, "deviations": 2.0},
            },
            {"decode": "descend", "shape_basis": "sign"},
            False,
        ),
    ],
    ids=["flat", "fixed", "tree descend", "shaped tree descend"],
)
def test_a_decode_is_compared_with_a_graph_unless_its_cache_builds_none(tmp_path, indexer, serving, graph):
    """A descending decode reads a mirror at fixed addresses, so it is captured - unless its node buckets are shaped."""
    decode = timed("lsa", indexer=indexer, serving=serving, phase="decode")
    argv = harness.harness_args(decode, tmp_path / "indexer.json", 131072, tmp_path / "o.json")
    assert ("--graph-compare" in argv) is graph


def test_a_served_run_refuses_a_variant_only_a_kernel_run_can_time(tmp_path):
    """Nothing scores `full` or `kda` - neither is a selection whose accuracy could be compared."""
    body = """
model:
  attention: kda
run:
  name: x
  image: /i.img
  tp: 1
"""
    path = tmp_path / "experiment.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match="serve a model"):
        experiment_config.load(path, kind=Evaluation.ACCURACY)
    with pytest.raises(ValueError, match="serve a model"):
        experiment_config.load(path, kind=Evaluation.PRODUCTION)
    assert experiment_config.load(path, kind=Evaluation.KERNELS)[0].model.attention == "kda"


def test_a_resumed_ladder_keeps_its_samples_and_its_wall_clock(tmp_path, monkeypatch):
    """A requeued job counting only its second attempt reports a three-hour run as twenty minutes."""
    run_dir = tmp_path / "glm-5.3-flash__native__prefill__2026-09-22"
    kernels = timed(phase="prefill", contexts=[131072, 262144])
    measured = []

    def fake_measure(_experiment, _settings, context, directory, log):
        if record.sample_paths(directory, context):
            return True
        measured.append(context)
        write(directory, f"{context}.json", raw(context))
        return True

    monkeypatch.setattr(harness, "measure", fake_measure)
    harness.run_kernels(kernels, None, "native", run_dir)
    first = json.loads((run_dir / "record.json").read_text())["run"]
    assert measured == [131072, 262144]

    harness.run_kernels(kernels, None, "native", run_dir)
    second = json.loads((run_dir / "record.json").read_text())["run"]

    # Nothing was measured again, and the run still says when it started and how long it has cost
    assert measured == [131072, 262144]
    assert second["started"] == first["started"]
    assert second["seconds"] >= first["seconds"]


def test_a_ladder_whose_every_context_failed_raises_rather_than_writing_nothing(tmp_path, monkeypatch):
    run_dir = tmp_path / "glm-5.3-flash__native__prefill__2026-09-22-b"
    monkeypatch.setattr(harness, "measure", lambda *args: False)
    with pytest.raises(RuntimeError, match="every context failed"):
        harness.run_kernels(timed(phase="prefill", contexts=[131072]), None, "native", run_dir)


def test_a_kernels_data_block_rejects_a_context_the_harness_would_refuse():
    with pytest.raises(ValueError, match="multiple"):
        KernelsDataConfig(contexts=[100000])
    with pytest.raises(ValueError, match="phase"):
        KernelsDataConfig(phase="both")
    with pytest.raises(ValueError, match="at least one"):
        KernelsDataConfig(contexts=[])


def test_a_point_keeps_its_host_and_what_its_card_did(tmp_path, meta):
    run_dir = tmp_path / "glm-5.3-flash__native__prefill__2026-09-17"
    write(run_dir, "262144-253952.json", raw(253952, module_ms=10.0))
    write(run_dir, "262144-262144.json", raw(262144, module_ms=30.0, growth=True))
    write(run_dir, "131072.json", raw(131072, module_ms=8.0))
    rows = "2026/10/20 10:00:00.000, 0, GPU-abc, {sm}, 2619, 650.0, 700.0, 61, {mask}\n"
    for stem, sm, mask in (("262144-253952", 1980, "0x0"), ("262144-262144", 1200, "0x4")):
        telemetry.sample_path(run_dir / "samples" / stem, "node-1").write_text(rows.format(sm=sm, mask=mask))

    built = record.build(run_dir.name, meta, run_dir, [131072, 262144])

    window = built["points"]["262144"]["hardware"]
    assert window["host"] == "node-1"
    assert window["driver"] == "570.211.01"
    gpu = window["telemetry"]["gpus"]["GPU-abc"]
    assert gpu["samples"] == 2
    assert gpu["sm_mhz_min"] == 1200.0
    assert gpu["power_capped_fraction"] == 0.5
    assert gpu["slowdown_fraction"] == 0.0
    assert gpu["throttle_reasons"] == {"sw_power_cap": 0.5}
    # A point measured before the sampler has the host and no telemetry
    assert built["points"]["131072"]["hardware"]["telemetry"] is None
    # The host is a fact of the point, so it stays out of what a join compares
    assert "hardware" not in str(built["setup"])
    assert built["run"]["host"] == "node-1"
