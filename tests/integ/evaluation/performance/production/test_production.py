"""A production run from the client's result files: the record, the table, the ladder and its resume.

The result files are the shape sglang's serving benchmark writes with `--output-details`. They are built
here rather than captured, because what is checked is the arithmetic over them and the ladder around them.
"""

import contextlib
import itertools
import json
import subprocess
from pathlib import Path

import pytest

from lsa.evaluation import run as runner
from lsa.evaluation.config import EvaluatedModelConfig, ProductionConfig, ProductionDataConfig, RunConfig
from lsa.evaluation.constants import Production
from lsa.evaluation.performance import telemetry
from lsa.evaluation.performance.production import measure, record
from lsa.evaluation.run import hour_limit
from lsa.sglang import launch
from tests.helpers.evaluation.sampler import fake_subprocess

INFO = {
    "model_path": "zai-org/GLM-5.3-Flash",
    "tp_size": 8,
    "pp_size": 1,
    "nnodes": 1,
    "context_length": 1052672,
}


def result(ttfts=(4.0, 4.2, 4.1), gap=0.0064, output_tokens=4, info=INFO, errors=None):
    return {
        "completed": len(ttfts),
        "max_concurrency": 1,
        "random_output_len": output_tokens,
        "duration": sum(ttfts),
        "output_throughput": 3 * output_tokens / sum(ttfts),
        "max_output_tokens_per_s": 160.0,
        "max_concurrent_requests": 1,
        "server_info": {
            **info,
            "cuda_graph_config": {
                "decode": {"backend": "full", "bs": [1, 2, 4]},
                "prefill": {"backend": "disabled"},
            },
            "internal_states": [{"memory_usage": {"token_capacity": 6671040}}],
        },
        "input_lens": [131072] * len(ttfts),
        "output_lens": [output_tokens] * len(ttfts),
        "ttfts": list(ttfts),
        "itls": [[gap] * (output_tokens - 1) for _ in ttfts],
        "errors": errors or [""] * len(ttfts),
    }


def write_result(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(body) + "\n")


CP_INFO = {
    **INFO,
    "attn_cp_size": 8,
    "dcp_size": 8,
    "enable_prefill_cp": True,
    "cp_strategy": "contiguous",
    "ep_size": 8,
    "moe_runner_backend": "deep_gemm",
}
TP = "tp8-cp1-dcp1-pp1-ep1"
TP_PP = "tp8-cp1-dcp1-pp2-ep1"
CP = "tp1-cp8-dcp8-pp1-ep8"


def test_a_deployment_names_every_axis_of_its_mesh():
    assert record.deployment(INFO) == TP
    assert record.deployment({**INFO, "pp_size": 2}) == TP_PP
    assert record.deployment({**INFO, "ep_size": 8}) == "tp8-cp1-dcp1-pp1-ep8"
    assert record.deployment(CP_INFO) == CP


def test_a_context_parallel_and_a_tensor_parallel_server_are_two_deployments(tmp_path):
    write_result(record.sample_path(tmp_path, TP, 131072), result())
    write_result(record.sample_path(tmp_path, CP, 131072), result(info=CP_INFO))
    body = record.build("run", {"image": "fork.img"}, tmp_path)
    assert set(body["points"]["131072"]) == {TP, CP}
    mesh = body["servers"][CP]["mesh"]
    assert (mesh["attn_tp_size"], mesh["attn_cp_size"], mesh["dcp_size"], mesh["ep_size"]) == (1, 8, 8, 8)
    assert (mesh["enable_prefill_cp"], mesh["cp_strategy"], mesh["moe_runner_backend"]) == (
        True,
        "contiguous",
        "deep_gemm",
    )
    assert mesh["image"] == "fork.img"
    assert body["servers"][TP]["mesh"]["attn_cp_size"] == 1


def test_tpot_is_the_mean_gap_after_the_first_token():
    row = record.request_rows(result(ttfts=(4.0,), gap=0.01, output_tokens=5))[0]
    assert row["tpot_ms"] == pytest.approx(10.0)
    assert row["e2e_s"] == pytest.approx(4.04)


def test_a_point_keeps_the_median_the_extremes_and_every_request():
    entry = record.point(result())
    assert entry["ttft_s"] == {"median": 4.1, "min": 4.0, "max": 4.2}
    assert entry["tpot_ms"]["median"] == pytest.approx(6.4)
    assert len(entry["requests"]) == 3


def test_a_point_keeps_what_the_node_served_and_never_the_clients_bucketed_peak():
    entry = record.point(result())
    assert entry["output_tokens_per_s"] == pytest.approx(12 / 12.3)
    assert entry["peak_output_tokens_per_s"] == 160.0
    assert "peak_running_requests" not in entry
    assert entry["wave_in_flight"] is None
    assert entry["wave_complete"] is None
    old = {key: value for key, value in result().items() if key not in ("output_throughput",)}
    assert record.point(old)["output_tokens_per_s"] is None


def test_a_rebuilt_record_counts_its_waves_from_the_server_log(tmp_path):
    write_result(
        record.sample_path(tmp_path, "tp8", 131072), result(ttfts=(1.0, 1.0)) | {"max_concurrency": 2}
    )
    log = [
        "[2026-10-18 00:00:00 TP0] Decode batch, #running-req: 1, #queue-req: 0",
        "[2026-10-18 00:00:01 TP0] Reset HybridReqToTokenPool",
        "[2026-10-18 00:00:02 TP0] Decode batch, #running-req: 1, #queue-req: 0",
    ]
    (tmp_path / "server.log").write_text("\n".join(log) + "\n")
    entry = record.build("run", {}, tmp_path)["points"]["131072"]["tp8-cp1-dcp1-pp1-ep1"]
    assert entry["wave_in_flight"] == 1
    assert entry["wave_complete"] is False


def test_a_rebuilt_record_without_a_log_has_no_wave_counts(tmp_path):
    write_result(record.sample_path(tmp_path, "tp8", 131072), result())
    entry = record.build("run", {}, tmp_path)["points"]["131072"]["tp8-cp1-dcp1-pp1-ep1"]
    assert (entry["wave_in_flight"], entry["wave_complete"]) == (None, None)


def test_wave_complete_is_equality():
    assert record.wave_complete(3, 3) is True
    assert record.wave_complete(2, 3) is False
    assert record.wave_complete(4, 3) is False
    assert record.wave_complete(None, 3) is None


def test_setup_keeps_the_concurrency_asked_and_the_request_limit_served():
    body = result(info=INFO | {"max_running_requests": 4}) | {"max_concurrency": 3}
    assert record.setup(body, 1)["concurrency"] == 3
    assert record.setup(body, 1)["max_running_requests"] == 4
    assert record.setup(result(), 1)["max_running_requests"] is None


def test_the_decode_window_opens_at_the_last_first_token_and_closes_at_the_first_finish():
    wave = {**result(ttfts=(1.0, 2.0), gap=0.5, output_tokens=5), "max_concurrency": 2}
    window = record.decode_window(wave)
    # First request's tokens at 1.5, 2.0, 2.5, 3.0; second at 2.5, 3.0, 3.5, 4.0
    assert window == {"start_s": 2.0, "seconds": 1.0, "output_tokens_per_s": 4.0, "tpot_ms": 500.0}
    assert record.decode_window(result()) is None
    apart = {**result(ttfts=(1.0, 9.0), gap=0.5, output_tokens=5), "max_concurrency": 2}
    assert record.decode_window(apart) is None


def test_the_client_caps_requests_in_flight_at_the_run_s_concurrency():
    output = Path("/runs/native/samples/tp8/131072.jsonl")
    joined = " ".join(
        measure.benchmark_args(timed(requests=3, concurrency=2), "http://127.0.0.1:61248", 131072, output)
    )
    assert "--num-prompts 3 --max-concurrency 2" in joined
    prefill = " ".join(
        measure.prefill_args(
            timed(requests=3, concurrency=3, profile_prefill_chunks=2),
            "http://127.0.0.1:61248",
            131072,
            output,
        )
    )
    assert "--num-prompts 1 --max-concurrency 1" in prefill


def test_a_production_data_block_refuses_a_cap_that_never_binds_or_more_requests_than_prompts():
    with pytest.raises(ValueError, match="concurrency"):
        ProductionDataConfig(concurrency=0)
    with pytest.raises(ValueError, match="2 in flight of 1 requests"):
        ProductionDataConfig(requests=1, concurrency=2)
    with pytest.raises(ValueError, match="3 entries of workload production at 131072 - got 4"):
        ProductionDataConfig(requests=4)


def test_the_last_line_of_a_retried_length_is_the_one_read(tmp_path):
    path = tmp_path / "131072.jsonl"
    write_result(path, result(ttfts=(9.0, 9.0, 9.0)))
    write_result(path, result(ttfts=(4.0, 4.0, 4.0)))
    assert record.read_result(path)["ttfts"] == [4.0, 4.0, 4.0]


def test_a_length_is_finished_only_with_every_request_and_no_error(tmp_path):
    path = tmp_path / "131072.jsonl"
    assert not record.finished(path, 3)
    write_result(path, result(ttfts=(4.0, 4.1)))
    assert not record.finished(path, 3)
    write_result(path, result(errors=["", "timeout", ""]))
    assert not record.finished(path, 3)
    write_result(path, result())
    assert record.finished(path, 3)


def test_the_record_keys_points_by_length_then_deployment(tmp_path):
    write_result(record.sample_path(tmp_path, TP, 1048576), result())
    write_result(record.sample_path(tmp_path, TP, 131072), result())
    write_result(record.sample_path(tmp_path, TP_PP, 1048576), result(info={**INFO, "pp_size": 2}))
    body = record.build("run", {"name": "native", "model": "zai-org/GLM-5.3-Flash"}, tmp_path)
    assert body["schema"] == Production.SCHEMA
    assert list(body["points"]) == ["131072", "1048576"]
    assert set(body["points"]["1048576"]) == {TP, TP_PP}
    assert body["servers"][TP]["token_capacity"] == 6671040
    assert body["servers"][TP]["cuda_graph"] == {"decode": "full", "prefill": "disabled"}
    assert body["model"]["label"] == "glm-5.3-flash"
    assert body["indexer"] is None


def test_the_record_quotes_the_settings_it_was_served_with(tmp_path):
    (tmp_path / "indexer.json").write_text(json.dumps({"bucket_size": 16, "top_k": 2048}))
    write_result(record.sample_path(tmp_path, TP, 131072), result())
    assert record.build("run", {}, tmp_path)["indexer"] == {"bucket_size": 16, "top_k": 2048}


def test_two_records_side_by_side_carry_the_speedup():
    native = {
        "run": {"name": "native"},
        "points": {"131072": {"tp8": record.point(result(ttfts=(8.0, 8.0, 8.0)))}},
    }
    ours = {
        "run": {"name": "lsa"},
        "points": {"131072": {"tp8": record.point(result(ttfts=(4.0, 4.0, 4.0)))}},
    }
    last = record.table([native, ours]).splitlines()[-1]
    assert last.split()[-2:] == ["2.00", "1.00"]


def timed(attention="native", **data):
    """A production run of GLM-5.3-Flash, its data block as given."""
    return ProductionConfig(
        model=EvaluatedModelConfig(name="zai-org/GLM-5.3-Flash", attention=attention),
        data=ProductionDataConfig(**data),
        run=RunConfig(name=attention, image="/images/sglang.img"),
    )


def test_the_client_profiles_decode_steps_only_when_asked():
    output = Path("/runs/native/samples/tp8/131072.jsonl")
    assert "--profile" not in measure.benchmark_args(timed(), "http://127.0.0.1:61248", 131072, output)
    argv = measure.benchmark_args(timed(profile_steps=20), "http://127.0.0.1:61248", 131072, output)
    joined = " ".join(argv)
    assert "--profile --profile-by-stage --profile-stages decode --profile-num-steps 20" in joined
    assert "--profile-output-dir /runs/native/profiles/131072" in joined
    with pytest.raises(ValueError, match="profile_steps"):
        ProductionDataConfig(profile_steps=-1)


def test_the_prefill_profile_is_one_cold_request_with_no_profile_flags():
    run = timed(profile_prefill_chunks=2, profile_steps=20, requests=3)
    joined = " ".join(
        measure.prefill_args(run, "http://127.0.0.1:61248", 1048576, Path("/runs/p/1048576.jsonl"))
    )
    assert "--num-prompts 1 --max-concurrency 1 --warmup-requests 0" in joined
    assert "--random-output-len 2" in joined
    assert "--profile" not in joined
    with pytest.raises(ValueError, match="profile_prefill_at"):
        ProductionDataConfig(profile_prefill_at=1.0)


def test_the_watcher_starts_the_profiler_after_its_share_of_chunks(tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    log.write_text("")
    posted = []

    class Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        measure.urllib.request, "urlopen", lambda request, timeout: posted.append(request) or Answer()
    )
    stop = measure.threading.Event()
    production = ProductionDataConfig(profile_prefill_chunks=2, profile_prefill_at=0.5)
    thread = measure.threading.Thread(
        target=measure.watch_prefill,
        args=("http://x", log, 32768, 4, production, tmp_path / "traces", stop, 0),
    )
    thread.start()
    line = "[2026 ATTN_CP0 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 32768, #cached-token: 0\n"
    other = "[2026 ATTN_CP1 TP1 EP1] Prefill batch, #new-seq: 1, #new-token: 32768, #cached-token: 0\n"
    with log.open("a") as handle:
        handle.write(line + other + line)
    thread.join(timeout=10)
    stop.set()
    assert len(posted) == 1
    assert json.loads(posted[0].data)["num_steps"] == 2


def test_a_production_data_block_refuses_a_measurement_that_cannot_say_what_a_token_costs():
    with pytest.raises(ValueError, match="output_tokens"):
        ProductionDataConfig(output_tokens=1)
    with pytest.raises(ValueError, match="lengths"):
        ProductionDataConfig(lengths=[])


def test_a_production_run_serves_so_it_takes_only_a_served_path():
    with pytest.raises(ValueError, match="kernel measurement"):
        timed("full")


def test_a_production_job_takes_its_own_ceiling():
    run = timed()
    assert hour_limit([run]) == Production.HOURS
    run.run.hours = 9
    assert hour_limit([run]) == 9


@pytest.fixture
def ladder(tmp_path, monkeypatch):
    """A run whose server is a stub and whose client writes a result, or fails, per length"""
    failing = set()
    sent = []
    real_run = subprocess.run

    @contextlib.contextmanager
    def serve(backend, model, settings, log):
        yield "http://127.0.0.1:61248"

    def client(argv, **kwargs):
        line = argv[-1]
        if "--random-input-len" not in line:
            return real_run(argv, **kwargs)
        length = int(line.split("--random-input-len ")[1].split()[0])
        sent.append(length)
        if length in failing:
            return subprocess.CompletedProcess(argv, 1)
        output = Path(line.split("--output-file ")[1].split()[0])
        write_result(output, result())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launch, "serve", serve)
    monkeypatch.setattr(measure, "server_info", lambda url: INFO)
    monkeypatch.setattr(measure.subprocess, "run", client)
    run = timed(lengths=[131072, 262144, 524288])
    run.run.output = str(tmp_path)
    run.run.id = "glm__native__production"
    run.run.gpu = "NVIDIA H200"
    return run, failing, sent


def test_the_ladder_writes_one_record_with_every_length(ladder, tmp_path):
    run, _, _ = ladder
    out = runner.run_production(run, force=False)
    body = json.loads((out / "record.json").read_text())
    assert list(body["points"]) == ["131072", "262144", "524288"]
    assert json.loads((out / "meta.json").read_text())["name"] == "native_h200"
    # The card the launch read is a fact of the record, beside the name that ends in it
    assert (body["run"]["name"], body["run"]["gpu"]) == ("native_h200", "NVIDIA H200")


def test_every_length_keeps_what_the_gpus_of_every_node_did_meanwhile(ladder, tmp_path, monkeypatch):
    run, _, _ = ladder
    started = []

    class Sampler:
        def __init__(self, argv, stdout, stderr):
            started.append(argv)
            host = "node-2" if "--nodelist=node-2" in argv else "node-1"
            uuid = "GPU-b" if host == "node-2" else "GPU-a"
            stdout.write(f"2026/10/20 10:00:00.000, 0, {uuid}, 1900, 2619, 600.0, 700.0, 60, 0x4\n".encode())

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(telemetry, "subprocess", fake_subprocess(Sampler))
    monkeypatch.setattr(measure, "job_hosts", lambda: ["node-1", "node-2"])
    monkeypatch.setattr(measure.socket, "gethostname", lambda: "node-1")

    out = runner.run_production(run, force=False)

    body = json.loads((out / "record.json").read_text())
    hardware = body["points"]["131072"][record.deployment(INFO)]["hardware"]
    assert hardware["hosts"] == ["node-1", "node-2"]
    assert hardware["gpus"]["GPU-b"]["host"] == "node-2"
    assert hardware["gpus"]["GPU-a"]["throttle_reasons"] == {"sw_power_cap": 1.0}
    assert len(started) == 6
    samples = out / "samples" / record.deployment(INFO)
    assert (samples / "131072.gpu-node-2.csv").is_file()
    # Facts of the point, so a join of parts from other hosts still agrees on the rest of the record
    assert "hardware" not in json.dumps({key: body[key] for key in ("setup", "servers", "run")})


def test_a_rerun_keeps_the_lengths_already_measured(ladder):
    run, _, sent = ladder
    runner.run_production(run, force=False)
    sent.clear()
    runner.run_production(run, force=False)
    assert sent == []


def test_the_ladder_stops_at_the_first_failure_and_says_so(ladder, tmp_path):
    run, failing, sent = ladder
    failing.add(262144)
    with pytest.raises(RuntimeError, match="stopped at 262144"):
        runner.run_production(run, force=False)
    assert sent == [131072, 262144]
    body = json.loads(
        (tmp_path / "performance" / "production" / "glm__native__production" / "record.json").read_text()
    )
    assert list(body["points"]) == ["131072"]


def test_a_length_past_the_server_s_context_is_skipped_not_failed(ladder, monkeypatch):
    run, _, sent = ladder
    monkeypatch.setattr(measure, "server_info", lambda url: {**INFO, "context_length": 300000})
    runner.run_production(run, force=False)
    assert sent == [131072, 262144]


def test_a_failed_attempt_is_set_aside_and_measured_again(ladder, tmp_path):
    run, _, sent = ladder
    output = record.sample_path(
        tmp_path / "performance" / "production" / "glm__native__production", record.deployment(INFO), 262144
    )
    write_result(output, result(ttfts=(4.0,)))
    runner.run_production(run, force=False)
    assert 262144 in sent
    assert output.with_suffix(".failed").exists()
    assert record.finished(output, run.data.requests)


def test_a_continued_run_must_serve_the_same_config(ladder, tmp_path):
    run, _, _ = ladder
    out = runner.run_production(run, force=False)
    with pytest.raises(ValueError, match="is native_h200 and this server runs lsa-top_level0"):
        measure.run_ladder(run, "http://127.0.0.1:61248", "lsa-top_level0", out)


def test_a_workload_counted_per_length_gives_each_length_its_own_disjoint_spans(monkeypatch):
    from lsa import assets
    from lsa.evaluation.performance.production.data import prompts

    card = assets.WorkloadCard(
        name="many",
        path="/w",
        tokenized="glm53_flash",
        source="gutenberg",
        file="g_{length}.json",
        lengths=[131072, 1048576],
        entries={131072: 8, 1048576: 2},
    )
    monkeypatch.setattr(prompts, "row_tokens", lambda card: 65536)
    assert not card.nested and card.entries_at(131072) == 8
    spans = [prompts.entry_rows(card, entry, 131072) for entry in range(8)]
    assert spans[:2] == [(0, 3), (3, 6)]
    assert all(end <= next_start for (_, end), (next_start, _) in itertools.pairwise(spans))
    assert prompts.entry_rows(card, 1, 1048576) == (17, 34)
    with pytest.raises(ValueError, match="entries counts lengths"):
        assets.WorkloadCard(
            name="bad",
            path="/w",
            tokenized="t",
            source="s",
            file="f_{length}",
            lengths=[1, 2],
            entries={1: 3},
        )
