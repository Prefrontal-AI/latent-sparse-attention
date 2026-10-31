"""What a read holds of a probe's targets, how the rows are described, and how the tables take their error over prompts."""

import math

import pytest
import torch

from lsa.training.distillation import probe_report
from lsa.training.distillation.metrics import (
    agreement,
    probe_metrics,
    probe_rows,
    window_and_sinks,
)

SINK, WINDOW = 4, 8
POSITIONS = torch.tensor([100, 100, 100])
READ = torch.tensor(
    [
        [20, 21, 40, -1, -1],
        [60, 61, -1, -1, -1],
        [20, 90, -1, -1, -1],
    ]
)


def test_a_read_holds_the_share_of_the_long_range_targets_it_has_and_whether_it_has_any():
    targets = torch.tensor([[20, 21, 22, -1], [3, 95, 50, -1], [-1, -1, -1, -1]])
    line = torch.tensor([[19, 20, 21, 22, 23], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    metrics = probe_metrics(READ, POSITIONS, targets, line, SINK, WINDOW)
    # Row 0 holds 20 and 21 of three, row 1 has only key 50 past the sinks and the window and misses it, row 2 has none
    assert metrics["target_held"][:2].tolist() == pytest.approx([2 / 3, 0.0])
    assert metrics["target_any"][:2].tolist() == [1.0, 0.0]
    assert math.isnan(float(metrics["target_held"][2])) and math.isnan(float(metrics["target_any"][2]))
    assert metrics["line_held"][0] == pytest.approx(2 / 5)
    assert math.isnan(float(metrics["line_held"][1]))


def test_a_target_in_the_window_or_the_sinks_is_not_counted_whatever_the_read():
    targets = torch.tensor([[2, 95, -1], [2, 95, -1], [2, 95, -1]])
    metrics = probe_metrics(READ, POSITIONS, targets, None, SINK, WINDOW)
    assert torch.isnan(metrics["target_held"]).all()
    assert "line_held" not in metrics


def test_a_probe_row_says_its_kind_its_long_range_target_count_and_its_nearest_distance():
    targets = torch.tensor([[20, 21, 22, -1], [3, 95, 50, -1], [-1, -1, -1, -1]])
    described = probe_rows(POSITIONS, torch.tensor([1, 0, 2], dtype=torch.int8), targets, SINK, WINDOW)
    assert described["query_kind"].tolist() == [1, 0, 2]
    assert described["target_count"].tolist() == [3, 1, 0]
    assert described["target_distance"].tolist() == [78, 50, -1]


def test_agreement_is_the_share_of_the_served_designs_long_range_keys_the_read_holds():
    served = torch.tensor([[20, 30, 99, -1, -1], [60, 61, -1, -1, -1], [-1, -1, -1, -1, -1]])
    shared = agreement(READ, served, POSITIONS, SINK, WINDOW)
    # Row 0 holds 20 and not 30, and 99 is in the window; row 1 holds both; row 2 has nothing to agree on
    assert shared[:2].tolist() == [0.5, 1.0]
    assert math.isnan(float(shared[2]))


def test_the_window_and_the_sinks_are_the_last_keys_and_the_first_ones_the_window_leaves():
    kept = window_and_sinks(torch.tensor([100, 5]), SINK, WINDOW)
    assert sorted(kept[0][kept[0] >= 0].tolist()) == [0, 1, 2, 3, *range(93, 101)]
    assert sorted(kept[1][kept[1] >= 0].tolist()) == list(range(0, 6))


def test_a_bucket_is_the_last_lower_edge_a_value_reaches():
    assert [probe_report.bucket_of(value, (1, 2, 4, 8)) for value in (0, 1, 3, 4, 9)] == [-1, 0, 1, 2, 3]
    assert probe_report.bucket_labels((1, 2, 4, 8), True) == ["1", "2-3", "4-7", "8+"]
    assert probe_report.bucket_labels((0, 4096, 16384), False) == ["0-4k", "4k-16k", "16k+"]


def probe_record(document, layers, values, task="cwe", length=16384):
    """A probe record of two rows whose `target_held` is `values` for read `a`"""
    return {
        "document": document,
        "task": task,
        "length": length,
        "layer": layers,
        "probe": {
            "query_kinds": ["question", "answer", "control"],
            "rows": {"query_kind": [1, 1], "target_count": [1, 3], "target_distance": [100, 9000]},
            "reads": {"a": {"target_held": values}},
        },
    }


def test_a_cell_averages_the_rows_then_the_layers_of_a_prompt_and_takes_the_error_over_prompts():
    records = [
        probe_record("p0", 3, [1.0, 0.0]),
        probe_record("p0", 7, [1.0, 1.0]),
        probe_record("p1", 3, [0.0, 0.0]),
        probe_record("p1", 7, [math.nan, 0.5]),
    ]
    mean, error, prompts = probe_report.cell(records, "a", "target_held", lambda record: [0, 1])
    # p0 is the mean of 0.5 and 1.0, p1 the mean of 0 and 0.5 - a row is not weighed against another prompt's
    assert mean == pytest.approx((0.75 + 0.25) / 2)
    assert error == pytest.approx(0.25)
    assert prompts == 2


def test_the_list_tasks_rows_are_split_by_target_count_and_by_distance_back():
    records = [probe_record("p0", 3, [1.0, 0.0]), probe_record("p1", 3, [0.0, 0.0])]
    lines, summary = probe_report.probe_tables(records)
    text = "\n".join(lines)
    assert "target_held by target count - cwe 16k, answer queries" in text
    assert summary["splits"]["target_count"]["cwe 16k"]["answer"]["a"]["1"]["mean"] == pytest.approx(0.5)
    # Target counts 1 and 3 fall in the buckets 1 and 2-3, distances 100 and 9000 in 0-4k and 4k-16k
    assert summary["splits"]["target_count"]["cwe 16k"]["answer"]["a"]["2-3"]["mean"] == 0.0
    assert summary["splits"]["target_distance"]["cwe 16k"]["answer"]["a"]["4k-16k"]["mean"] == 0.0


def test_a_table_prints_the_mean_and_its_error_and_leaves_out_a_read_with_nothing():
    records = [probe_record("p0", 3, [1.0, 0.0]), probe_record("p1", 3, [0.0, 0.0])]
    records[0]["probe"]["reads"]["empty"] = {"target_held": [math.nan, math.nan]}
    lines, _ = probe_report.probe_tables(records)
    text = "\n".join(lines)
    assert "| a | 0.250 ± 0.250 |" in text
    assert "| empty |" not in text
    assert "Prompts a column: cwe 16k: 2." in text


def test_prompts_whose_lengths_print_alike_share_a_column():
    records = [
        probe_record("p0", 3, [1.0, 0.0], length=16400),
        probe_record("p1", 3, [0.0, 0.0], length=16500),
    ]
    lines, summary = probe_report.probe_tables(records)
    assert "Prompts a column: cwe 16k: 2." in "\n".join(lines)
    assert summary["answer"]["target_held"]["a"]["cwe 16k"]["prompts"] == 2


def test_records_with_no_probe_make_no_probe_tables():
    assert probe_report.probe_tables([{"document": "a", "means": {}}]) == ([], {})
