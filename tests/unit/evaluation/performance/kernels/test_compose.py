"""The whole-model composition from harness records and module timings.

The records and module files are built here in the shapes the harness and the module timer write, with
costs chosen so the composed totals can be worked out by hand.
"""

import contextlib
import io
import json
import sys

import pytest

from lsa.evaluation.constants import Kernels
from lsa.evaluation.performance.kernels import compose

CHUNK = 8192

GLM_TEXT = {
    "num_hidden_layers": 45,
    "first_k_dense_replace": 3,
    "layer_types": (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 11 + ["linear_attention"],
}


def prefill_record(label, indexer, cost, contexts, window_chunks=32, gpu="GPU-a"):
    # A point's mean chunk is the cost at its window's centre, which is what a linear cost gives
    points = {}
    for context in contexts:
        first_end = max(CHUNK, context - (window_chunks - 1) * CHUNK)
        centre = (first_end + context) / 2
        points[str(context)] = {
            "window": [first_end, context],
            "module_ms": {"mean": cost(centre)},
            "gpu_uuid": gpu,
        }
    return {
        "schema": Kernels.SCHEMA,
        "run": {"id": f"{label}-prefill", "name": label, "phase": "prefill", "commits": {"lsa": "abc"}},
        "setup": {"indexer": indexer, "chunk": CHUNK},
        "points": points,
    }


def decode_record(label, indexer, amortized, gpu="GPU-a"):
    return {
        "schema": Kernels.SCHEMA,
        "run": {"id": f"{label}-decode", "name": label, "phase": "decode"},
        "setup": {"indexer": indexer, "chunk": CHUNK},
        "points": {
            str(context): {"amortized_ms": value, "gpu_uuid": gpu} for context, value in amortized.items()
        },
    }


def modules_file(phase, cost=1.0, gpu="GPU-a"):
    key = "event_ms" if phase == "prefill" else "graph_ms"
    other = "graph_ms" if phase == "prefill" else "event_ms"
    return {
        "schema": Kernels.MODULES_SCHEMA,
        "phase": phase,
        "gpu_uuid": gpu,
        "modules": {name: {key: [cost, cost, 9 * cost], other: [100.0, 100.0]} for name in compose.MODULES},
    }


def test_an_unknown_layer_type_is_refused():
    text = {**GLM_TEXT, "layer_types": ["sliding_attention"] * 45}

    with pytest.raises(ValueError, match="sliding_attention"):
        compose.layer_counts(text)


def test_a_constant_chunk_cost_prefills_in_chunks_times_cost():
    record = prefill_record("native", "native", lambda centre: 7.0, [131072, 1048576])

    assert compose.cumulative_prefill_ms(record, 1048576, CHUNK) == pytest.approx(7.0 * 128)


def test_a_linear_chunk_cost_is_summed_exactly():
    # The first window is one chunk long, so there is no stretch before the first centre to hold flat
    def cost(prefix):
        return 2.0 + prefix / 65536

    record = prefill_record("native", "native", cost, [8192], window_chunks=1)
    record["points"].update(prefill_record("native", "native", cost, [262144, 1048576])["points"])
    exact = sum(cost(end) for end in range(CHUNK, 1048576 + 1, CHUNK))

    assert compose.cumulative_prefill_ms(record, 1048576, CHUNK) == pytest.approx(exact)


def test_a_prefill_reads_no_point_past_its_context():
    short = prefill_record("native", "native", lambda centre: 5.0, [131072])
    longer = prefill_record(
        "native", "native", lambda centre: 5.0 if centre < 100000 else 500.0, [131072, 262144]
    )

    assert compose.cumulative_prefill_ms(longer, 131072, CHUNK) == compose.cumulative_prefill_ms(
        short, 131072, CHUNK
    )


def test_module_costs_take_the_eager_median_at_prefill_and_the_replay_at_decode():
    assert compose.module_costs(modules_file("prefill", cost=2.0))["moe"] == 2.0
    assert compose.module_costs(modules_file("decode", cost=0.5))["head"] == 0.5


def test_a_module_file_of_another_schema_is_refused():
    with pytest.raises(ValueError, match="schema"):
        compose.module_costs({**modules_file("decode"), "schema": "lsa-kernels/1"})


def records_for_two_configurations():
    return {
        "native-prefill": prefill_record("native", "native", lambda centre: 10.0, [131072]),
        "lsa-prefill": prefill_record("lsa-top_level0", "lsa", lambda centre: 5.0, [131072]),
        "kda-prefill": prefill_record("kda", "kda", lambda centre: 1.0, [131072]),
        "native-decode": decode_record("native", "native", {131072: 0.2}),
        "lsa-decode": decode_record("lsa-top_level0", "lsa", {131072: 0.1}),
        "kda-decode": decode_record("kda", "kda", {131072: 0.1}),
    }


def test_the_composed_totals_add_every_part_by_its_count():
    phases = compose.compose(
        {"text_config": GLM_TEXT},
        {"prefill": modules_file("prefill"), "decode": modules_file("decode")},
        records_for_two_configurations(),
        "native",
    )["phases"]

    # 16 chunks - 11 sparse layers, 34 linear, then 42 + 3 + 90 + 1 + 1 modules a chunk and one head
    chunks = 131072 // CHUNK
    native_prefill = 11 * 10.0 * chunks + 34 * 1.0 * chunks + (42 + 3 + 90 + 1 + 1) * chunks + 1
    assert phases["prefill"]["total_ms"]["native"]["131072"] == pytest.approx(native_prefill)
    native_decode = 11 * 0.2 + 34 * 0.1 + 42 + 3 + 90 + 1 + 1 + 1
    lsa_decode = 11 * 0.1 + 34 * 0.1 + 42 + 3 + 90 + 1 + 1 + 1
    assert phases["decode"]["speedup"]["lsa-top_level0"]["131072"] == pytest.approx(
        native_decode / lsa_decode, abs=1e-4
    )
    assert phases["decode"]["speedup"]["native"]["131072"] == 1.0
    assert phases["prefill"]["same_card"]


def test_a_native_record_is_the_baseline_under_its_own_run_name():
    records = {
        path: {**record, "run": {**record["run"], "name": f"glm53_flash_kernels_{path}_h200"}}
        for path, record in records_for_two_configurations().items()
    }

    phases = compose.compose(
        {"text_config": GLM_TEXT},
        {"prefill": modules_file("prefill"), "decode": modules_file("decode")},
        records,
        "native",
    )["phases"]

    assert phases["decode"]["speedup"]["native"]["131072"] == 1.0
    assert "glm53_flash_kernels_lsa-decode_h200" in phases["decode"]["speedup"]


def test_records_from_two_cards_are_flagged():
    records = records_for_two_configurations()
    records["lsa-decode"] = decode_record("lsa-top_level0", "lsa", {131072: 0.1}, gpu="GPU-b")

    phases = compose.compose(
        GLM_TEXT, {"prefill": modules_file("prefill"), "decode": modules_file("decode")}, records, "native"
    )["phases"]

    assert not phases["decode"]["same_card"]
    assert phases["decode"]["cards"] == ["GPU-a", "GPU-b"]


def test_a_phase_without_a_linear_attention_record_is_refused():
    records = records_for_two_configurations()
    del records["kda-decode"]

    with pytest.raises(ValueError, match="no linear attention record"):
        compose.compose(
            GLM_TEXT,
            {"prefill": modules_file("prefill"), "decode": modules_file("decode")},
            records,
            "native",
        )


def test_two_records_for_one_configuration_are_refused():
    records = records_for_two_configurations()
    records["native-prefill-again"] = records["native-prefill"]

    with pytest.raises(ValueError, match="second native prefill"):
        compose.split_records(records)


def test_build_writes_a_composition_the_show_prints(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": GLM_TEXT}))
    module_paths = []
    for phase in ("prefill", "decode"):
        path = tmp_path / f"{phase}.json"
        path.write_text(json.dumps(modules_file(phase)))
        module_paths.append(str(path))
    record_paths = []
    for name, body in records_for_two_configurations().items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(body))
        record_paths.append(str(path))
    out = tmp_path / "composition.json"
    argv = ["compose", "--config", str(config), "--modules", *module_paths, "--records", *record_paths]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])
    printed = io.StringIO()

    with contextlib.redirect_stdout(printed):
        compose.main()

    written = json.loads(out.read_text())
    assert written["schema"] == Kernels.COMPOSITION_SCHEMA
    assert len(written["sources"]["records"]) == 6
    assert "lsa-top_level0" in printed.getvalue()
    assert "128K" in printed.getvalue()
