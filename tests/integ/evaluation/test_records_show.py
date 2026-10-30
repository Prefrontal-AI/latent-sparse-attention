"""`lsa records show` reads the kind from each record and prints the table that kind reads as."""

import json
import sys

import pytest

from lsa.evaluation.accuracy.records import show
from lsa.evaluation.constants import Kernels, Production, Records


def accuracy(name, score):
    return {
        "schema": Records.SCHEMA,
        "run": {"id": name, "name": name, "suite": "s", "date": "d", "seconds": 1, "note": ""},
        "model": {"name": "m"},
        "server": None,
        "tasks": {"mmlu_pro": {"score": score}},
    }


def kernels(name, milliseconds):
    point = {
        "module_ms": {"mean": milliseconds, "std": 0.1},
        "chunks": [1, 2],
        "ordinary_ms": 1.0,
        "growth_ms": None,
        "components_ms": {"index q scan": 0.1, "index k select": 0.2, "topk attention": 0.3},
        "memory": {"kv_bytes": 2**30, "index_bytes": 0, "working_bytes": 0},
        "checks": {"topk_exact": 2, "selection_exact": True, "rows_differing": 0},
    }
    return {
        "schema": Kernels.SCHEMA,
        "run": {"id": name, "name": name, "phase": "prefill", "date": "d", "seconds": 1, "gpu": "H200"},
        "model": {"name": "m"},
        "points": {"131072": point},
    }


def production(name, ttft):
    cell = {
        "ttft_s": {"median": ttft, "min": ttft, "max": ttft},
        "tpot_ms": {"median": 5, "min": 5, "max": 5},
    }
    return {
        "schema": Production.SCHEMA,
        "run": {"id": name, "name": name},
        "points": {"131072": {"tp8": cell}},
    }


def write(root, record):
    run = root / record["run"]["id"]
    run.mkdir()
    (run / Records.RECORD).write_text(json.dumps(record))
    return run


def shown(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["lsa records show", *map(str, arguments)])
    show.main()
    return capsys.readouterr().out


def test_an_accuracy_record_is_a_line_per_task(tmp_path, monkeypatch, capsys):
    run = write(tmp_path, accuracy("a", 0.5))
    assert "mmlu_pro" in shown(monkeypatch, capsys, run / Records.RECORD)


def test_two_accuracy_records_align_their_tasks(tmp_path, monkeypatch, capsys):
    a, b = write(tmp_path, accuracy("a", 0.5)), write(tmp_path, accuracy("b", 0.25))
    out = shown(monkeypatch, capsys, a, b)
    assert "0.500" in out
    assert "0.250" in out


def test_a_kernel_record_is_a_line_per_context(tmp_path, monkeypatch, capsys):
    run = write(tmp_path, kernels("k", 3.0))
    out = shown(monkeypatch, capsys, run / Records.RECORD)
    assert "128K" in out
    assert "3.000 ms" in out


def test_two_kernel_records_carry_the_speedup_over_the_first(tmp_path, monkeypatch, capsys):
    slow, fast = write(tmp_path, kernels("slow", 4.0)), write(tmp_path, kernels("fast", 2.0))
    assert "x 2.00" in shown(monkeypatch, capsys, slow, fast)


def test_two_production_records_carry_the_speedup_over_the_first(tmp_path, monkeypatch, capsys):
    native, ours = write(tmp_path, production("native", 8.0)), write(tmp_path, production("lsa", 4.0))
    last = shown(monkeypatch, capsys, native, ours).splitlines()[-1]
    assert last.split()[-2:] == ["2.00", "1.00"]


def test_a_call_that_mixes_kinds_is_refused(tmp_path, monkeypatch, capsys):
    a, k = write(tmp_path, accuracy("a", 0.5)), write(tmp_path, kernels("k", 3.0))
    with pytest.raises(SystemExit):
        shown(monkeypatch, capsys, a, k)
    assert "one kind at a time" in capsys.readouterr().err


def test_a_run_with_no_record_is_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(FileNotFoundError):
        shown(monkeypatch, capsys, tmp_path)
