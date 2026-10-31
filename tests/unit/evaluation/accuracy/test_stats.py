"""A score's error with each problem counted once, and the paired gap between two runs on the same problems."""

import json
import math

import pytest

from lsa.evaluation.accuracy import records
from lsa.evaluation.accuracy.records import compare
from lsa.evaluation.constants import Records


def rows_of(scores_by_doc, error_at=()):
    """One row per (doc, repeat) from a list of repeat scores per doc, a row at an `error_at` pair failed."""
    return [
        {
            "doc_id": doc_id,
            "repeat": repeat,
            "metrics": {} if (doc_id, repeat) in error_at else {"m": score},
            "error": "boom" if (doc_id, repeat) in error_at else None,
            "answer": "x",
        }
        for doc_id, scores in enumerate(scores_by_doc)
        for repeat, score in enumerate(scores)
    ]


def test_doc_means_average_the_repeats_of_a_doc_and_skip_failed_rows():
    rows = rows_of([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], error_at={(1, 3)})
    assert records.doc_means(rows, "m") == {0: 0.75, 1: 0.0}


def test_clustered_error_is_the_error_over_doc_means_not_over_samples():
    # Two problems asked four times each - one always solved, one never
    rows = rows_of([[1.0] * 4, [0.0] * 4])
    stats = records.doc_statistics(rows, "m")
    assert stats["stderr_doc"] == pytest.approx(0.5)
    # The binomial treats the eight answers as independent draws, and is under half of that
    assert records.summarize(rows, "m")["stderr"] == pytest.approx(math.sqrt(0.25 / 8), abs=1e-4)
    assert stats["stderr_doc"] > 2.5 * records.summarize(rows, "m")["stderr"]


def test_clustered_error_equals_the_binomial_for_one_repeat_per_doc():
    rows = rows_of([[1.0], [0.0], [1.0], [1.0], [0.0], [1.0]])
    out = records.summarize(rows, "m")
    # stdev uses n - 1, the binomial n
    assert out["stderr_doc"] == pytest.approx(out["stderr"] * math.sqrt(6 / 5), abs=1e-3)


def test_one_doc_has_no_spread():
    stats = records.doc_statistics(rows_of([[1.0, 0.0]]), "m")
    assert stats == {"stderr_doc": None, "ci95": None}


def test_the_bootstrap_is_deterministic_and_brackets_the_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    first = records.bootstrap_interval(values)
    assert records.bootstrap_interval(values) == first
    assert first == records.bootstrap_interval(list(values))
    low, high = first
    assert 0.0 < low < 0.7 < high <= 1.0


def test_the_bootstrap_interval_collapses_where_every_doc_agrees():
    assert records.bootstrap_interval([1.0] * 5) == [1.0, 1.0]


def test_the_bootstrap_interval_barely_moves_with_the_block_size(monkeypatch):
    values = [float(i % 2) for i in range(40)]
    whole = records.bootstrap_interval(values)
    monkeypatch.setattr(records, "BOOTSTRAP_BLOCK_CELLS", 40)
    one_draw_at_a_time = records.bootstrap_interval(values)
    # Blocks only change how the indices are cut up, so the interval stays within sampling noise of itself
    assert one_draw_at_a_time == pytest.approx(whole, abs=0.03)


def test_summarize_carries_both_errors_and_the_interval():
    out = records.summarize(rows_of([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]), "m")
    assert out["score"] == 0.625
    assert out["stderr_doc"] == pytest.approx(0.2394, abs=1e-4)
    assert len(out["ci95"]) == 2
    assert out["ci95"][0] < out["score"] < out["ci95"][1]


def test_a_paired_gap_is_the_mean_of_the_docs_gaps_with_wins_losses_and_ties():
    base = rows_of([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    other = rows_of([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    out = records.paired_comparison(base, other, "m")
    assert out["docs"] == 4
    assert out["gap"] == pytest.approx(0.125)
    assert (out["wins"], out["losses"], out["ties"]) == (1, 1, 2)
    assert out["stderr"] == pytest.approx(0.3146, abs=1e-4)
    assert out["score_a"] == 0.625
    assert out["score_b"] == 0.75


def test_a_paired_comparison_matches_by_doc_id_and_counts_the_rest():
    base = rows_of([[1.0], [1.0], [0.0]])
    other = [row for row in rows_of([[0.0], [1.0], [1.0], [1.0]]) if row["doc_id"] != 1]
    out = records.paired_comparison(base, other, "m")
    assert out["docs"] == 2
    assert (out["only_a"], out["only_b"]) == (1, 1)
    assert out["gap"] == pytest.approx(0.0)


def test_pairing_removes_the_difficulty_the_runs_share():
    # Every doc is solved by b and half of them missed by a - the gap is steady in every doc
    docs = 40
    base = rows_of([[0.0] if i % 2 else [1.0] for i in range(docs)])
    other = rows_of([[1.0] for _ in range(docs)])
    out = records.paired_comparison(base, other, "m")
    assert out["gap"] == pytest.approx(0.5)
    assert out["ci95"][0] > 0
    unpaired = math.sqrt(2 * 0.25 / docs)
    assert out["stderr"] < unpaired


def test_a_gap_of_nothing_has_an_interval_at_zero():
    rows = rows_of([[1.0], [0.0], [1.0]])
    out = records.paired_comparison(rows, rows, "m")
    assert out["gap"] == 0.0
    assert out["ci95"] == [0.0, 0.0]
    assert out["ties"] == 3


def test_identical_answers_count_the_samples_both_runs_finished_with_the_same_text():
    rows = rows_of([[1.0], [1.0], [1.0], [1.0]], error_at={(3, 0)})
    other = [{**row, "answer": "y" if row["doc_id"] == 1 else "x"} for row in rows_of([[1.0]] * 5)]
    assert records.identical_answers(rows, other) == (2, 3)
    assert records.paired_comparison(rows, other, "m")["identical"] == pytest.approx(2 / 3, abs=1e-4)


def test_no_sample_finished_by_both_has_no_share_of_identical_answers():
    out = records.paired_comparison(rows_of([[1.0]]), [{**rows_of([[1.0]])[0], "doc_id": 9}], "m")
    assert out["identical"] is None


def test_no_shared_doc_is_no_gap():
    out = records.paired_comparison(rows_of([[1.0]]), [{**rows_of([[1.0]])[0], "doc_id": 9}], "m")
    assert out["docs"] == 0
    assert out["gap"] is None
    assert out["ci95"] is None


def write_run(root, name, tasks):
    """A run directory whose record names each task's samples file, as a joined run's does."""
    entries = {}
    for task, rows in tasks.items():
        path = root / name / "samples" / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        entries[task] = {"metric": "m", "samples": f"samples/{task}.jsonl"}
    (root / name / Records.RECORD).write_text(json.dumps({"schema": Records.SCHEMA, "tasks": entries}))
    return root / name


def test_runs_compare_over_the_tasks_both_hold(tmp_path):
    first = write_run(tmp_path, "a", {"t1": rows_of([[1.0], [0.0]]), "t2": rows_of([[1.0], [1.0]])})
    second = write_run(tmp_path, "b", {"t1": rows_of([[1.0], [1.0]]), "t3": rows_of([[0.0]])})
    out = records.compare_runs(first, second)
    assert list(out) == ["t1"]
    assert out["t1"]["gap"] == 0.5
    with pytest.raises(ValueError, match="not in both runs: t2"):
        records.compare_runs(first, second, ["t1", "t2"])


def test_the_command_prints_a_table_and_marks_a_gap_whose_interval_excludes_zero(
    tmp_path, capsys, monkeypatch
):
    first = write_run(tmp_path, "a", {"cwe_512": rows_of([[1.0]] * 30), "cwe_4096": rows_of([[1.0]] * 30)})
    other_rows = rows_of([[0.0]] * 12 + [[1.0]] * 18)
    second = write_run(tmp_path, "b", {"cwe_512": other_rows, "cwe_4096": rows_of([[1.0]] * 30)})
    monkeypatch.setattr("sys.argv", ["lsa records compare", str(first), str(second / Records.RECORD)])
    compare.main()
    lines = capsys.readouterr().out.splitlines()
    marked = [line for line in lines if line.startswith("cwe_512")]
    level = [line for line in lines if line.startswith("cwe_4096")]
    assert marked[0].count("*") == 1
    assert "-0.400" in marked[0]
    assert "*" not in level[0]
    # Each length rolls up under the tasks, with the share of answers that read the same
    assert [line.split()[-1] for line in lines if line.startswith("RULER")] == ["1.000", "1.000"]
    # A task's lengths read in order
    assert lines.index(marked[0]) < lines.index(level[0])


def test_a_pattern_that_matches_no_task_is_refused():
    with pytest.raises(ValueError, match="no task matches"):
        compare.chosen(["nope*"], ["t1"])
    assert compare.chosen(["t*"], ["t1", "u"]) == ["t1"]


def test_the_rollup_weights_each_length_by_the_docs_its_tasks_matched():
    def comparison(docs, base, run, identical):
        return {"docs": docs, "score_a": base, "score_b": run, "identical": identical}

    comparisons = {
        "ruler_cwe_4096": comparison(30, 1.0, 0.9, 1.0),
        "ruler_vt_4096": comparison(10, 0.6, 0.6, 0.5),
        "ruler_cwe_32768": comparison(0, None, None, None),
        "mmlu_pro": comparison(5, 0.5, 0.5, 1.0),
    }
    rolled = compare.rollup(comparisons)
    assert list(rolled) == [4096]
    assert rolled[4096]["docs"] == 40
    assert rolled[4096]["base"] == pytest.approx(0.9)
    assert rolled[4096]["run"] == pytest.approx(0.825)
    assert rolled[4096]["gap"] == pytest.approx(-0.075)
    assert rolled[4096]["same"] == pytest.approx(0.875)
