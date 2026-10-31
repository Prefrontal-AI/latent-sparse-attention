"""The telemetry sampler's rows and what a record keeps of them."""

import sys

from lsa.evaluation.performance import telemetry
from tests.helpers.evaluation.sampler import fake_subprocess

HEADER = "timestamp, index, uuid, clocks.sm [MHz], clocks.mem [MHz], power.draw [W], power.limit [W], temperature.gpu, clocks_throttle_reasons.active"


def row(uuid="GPU-a", sm="1980", power="650.5", temperature="61", mask="0x0000000000000000", index="0"):
    return f"2026/10/20 10:00:00.000, {index}, {uuid}, {sm}, 2619, {power}, 700.00, {temperature}, {mask}"


def test_parse_skips_header_repeats_and_a_row_cut_short():
    text = "\n".join(
        [HEADER, row(), HEADER, row(sm="1500", mask="0x4"), "2026/10/20 10:00:00.200, 0, GPU-a, 19"]
    )

    readings = telemetry.parse(text)

    assert [reading["sm_mhz"] for reading in readings] == [1980.0, 1500.0]
    assert readings[0]["reasons"] == []
    assert readings[1]["reasons"] == ["sw_power_cap"]


def test_parse_reads_a_missing_cell_as_none():
    readings = telemetry.parse(row(power="[N/A]"))

    assert readings[0]["power_w"] is None
    assert readings[0]["temperature_c"] == 61.0


def test_a_mask_names_every_bit_set():
    assert telemetry.reason_names(0x1 | 0x4 | 0x40) == ["gpu_idle", "sw_power_cap", "hw_thermal_slowdown"]


def test_idle_and_the_application_clocks_setting_are_not_throttling():
    rows = [
        row(mask="0x1"),
        row(mask="0x2"),
        row(mask="0x3"),
        row(mask="0x4", sm="1200"),
        row(mask="0x24", sm="1000"),
        row(sm="1980"),
    ]

    card = telemetry.summarize(telemetry.parse("\n".join(rows)))["GPU-a"]

    assert card["samples"] == 6
    assert card["power_capped_fraction"] == round(2 / 6, 4)
    assert card["slowdown_fraction"] == round(1 / 6, 4)
    assert card["slowdown_reasons"] == {"sw_thermal_slowdown": round(1 / 6, 4)}
    assert card["throttle_reasons"] == {
        "sw_power_cap": round(2 / 6, 4),
        "sw_thermal_slowdown": round(1 / 6, 4),
    }
    assert card["sm_mhz_min"] == 1000.0
    assert card["sm_mhz_median"] == 1980.0
    assert card["power_w_max"] == 650.5
    assert card["power_limit_w"] == 700.0
    assert card["temperature_c_max"] == 61.0


def test_a_summary_is_kept_per_card():
    rows = [row("GPU-a", index="0"), row("GPU-b", sm="1000", index="1", temperature="80")]

    summary = telemetry.summarize(telemetry.parse("\n".join(rows)))

    assert sorted(summary) == ["GPU-a", "GPU-b"]
    assert summary["GPU-b"]["index"] == 1
    assert summary["GPU-b"]["temperature_c_max"] == 80.0


def test_a_point_is_summarized_over_every_file_and_node_it_wrote(tmp_path):
    first, second = tmp_path / "1024-512", tmp_path / "1024-1024"
    telemetry.sample_path(first, "node-1").write_text(row() + "\n" + row(sm="1000", mask="0x20") + "\n")
    telemetry.sample_path(second, "node-1").write_text(row(sm="1980") + "\n" + row(sm="1980") + "\n")
    telemetry.sample_path(second, "node-2").write_text(row("GPU-b") + "\n")

    point = telemetry.read_point([first, second])

    assert point is not None
    assert point["hosts"] == ["node-1", "node-2"]
    assert point["gpus"]["GPU-a"]["host"] == "node-1"
    assert point["gpus"]["GPU-a"]["samples"] == 4
    assert point["gpus"]["GPU-a"]["slowdown_fraction"] == 0.25
    assert point["gpus"]["GPU-a"]["power_capped_fraction"] == 0.0
    assert point["gpus"]["GPU-b"]["host"] == "node-2"
    assert point["files"] == [
        "1024-1024.gpu-node-1.csv",
        "1024-1024.gpu-node-2.csv",
        "1024-512.gpu-node-1.csv",
    ]


def test_a_point_without_rows_has_no_telemetry(tmp_path):
    assert telemetry.read_point([tmp_path / "1024"]) is None
    telemetry.sample_path(tmp_path / "1024", "node-1").write_text(HEADER + "\n")
    assert telemetry.read_point([tmp_path / "1024"]) is None


def test_the_sampler_command_names_the_cards_and_the_interval():
    argv = telemetry.command(["GPU-a"])

    assert argv[0] == "nvidia-smi"
    assert "--id=GPU-a" in argv
    assert "-lms=100" in argv
    assert "clocks_throttle_reasons.active" in argv[1]
    assert not any(part.startswith("--id") for part in telemetry.command())


def test_a_sampler_that_cannot_start_leaves_an_empty_file_and_the_measurement_runs(tmp_path):
    path = tmp_path / "a.gpu-x.csv"

    with telemetry.Sampler(path, prefix=["/nonexistent-binary"]):
        pass

    assert path.read_text() == ""


def test_a_sampler_writes_while_it_runs_and_stops_with_the_window(tmp_path):
    path = tmp_path / "a.gpu-x.csv"
    loop = [sys.executable, "-c", "import time\nprint('a,b', flush=True)\ntime.sleep(60)"]

    sampler = telemetry.Sampler(path)
    sampler.argv = loop
    with sampler:
        assert sampler.process is not None
        while not path.read_text():
            pass
    assert sampler.process.poll() is not None
    assert path.read_text() == "a,b\n"


def test_nodes_other_than_this_one_are_sampled_through_a_step(tmp_path, monkeypatch):
    started = []

    class Fake:
        def __init__(self, argv, stdout, stderr):
            started.append(argv)
            stdout.write(row().encode() + b"\n")

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(telemetry, "subprocess", fake_subprocess(Fake))

    with telemetry.sample_nodes(tmp_path / "1024", ["node-1", "node-2"], "node-1", 8):
        pass

    assert started[0][0] == "nvidia-smi"
    assert started[1][:2] == ["srun", "--overlap"]
    assert "--nodelist=node-2" in started[1]
    assert "--gres=gpu:8" in started[1]
    assert started[1][-len(started[0]) :] == started[0]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "1024.gpu-node-1.csv",
        "1024.gpu-node-2.csv",
    ]
