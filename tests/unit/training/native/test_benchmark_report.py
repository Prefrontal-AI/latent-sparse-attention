import numpy as np
import pytest

from lsa import io
from lsa.training.native.benchmark import records, report


def logged(doc_id, **metrics):
    return {
        "doc_id": doc_id,
        "doc": {"input": "a very long prompt"},
        "doc_hash": f"d{doc_id}",
        "prompt_hash": f"p{doc_id}",
        "target": "t",
        "arguments": [["prompt", {}]],
        "resps": [["reply"]],
        "filtered_resps": ["reply"],
        "metrics": list(metrics),
        **metrics,
    }


def results_of(task, samples):
    return {
        "results": {task: {"acc,none": 0.5}},
        "versions": {task: 1},
        "n-samples": {task: len(samples)},
        "samples": {task: samples},
    }


def test_a_sample_keeps_what_a_comparison_reads_and_drops_the_prompt():
    row = records.slim_sample(logged(3, acc=1.0))
    assert row == {
        "doc_id": 3,
        "doc_hash": "d3",
        "prompt_hash": "p3",
        "target": "t",
        "resps": ["reply"],
        "metrics": {"acc": 1.0},
    }


def test_samples_gathered_from_several_ranks_are_written_in_id_order_and_read_back_by_id(tmp_path):
    results = results_of("hellaswag", [logged(2, acc=0.0), logged(0, acc=1.0), logged(1, acc=1.0)])
    records.write_records(tmp_path, results, {"checkpoint": "/x/step-8000", "step": 8000})
    assert list(records.read_samples(tmp_path, "hellaswag")) == [0, 1, 2]
    assert records.read_config(tmp_path)["step"] == 8000
    assert io.read_json(tmp_path / records.RESULTS)["results"]["hellaswag"]["acc,none"] == 0.5


def test_a_mean_and_a_pooled_perplexity_are_the_two_statistics():
    assert report.statistic(np.array([1.0, 0.0, 1.0, 1.0])) == 0.75
    pairs = np.array([[-2.0, 2.0], [-4.0, 2.0]])
    assert report.statistic(pairs) == pytest.approx(np.exp(1.5))


def test_identical_arms_differ_by_nothing_and_a_shifted_arm_is_separated_from_zero():
    base = np.array([1.0, 0.0] * 50)
    delta, low, high = report.paired_bootstrap(base, base.copy(), iterations=200)
    assert (delta, low, high) == (0.0, 0.0, 0.0)
    better = np.ones(100)
    delta, low, high = report.paired_bootstrap(base, better, iterations=200)
    assert delta == pytest.approx(0.5)
    assert low > 0.3


def test_pairing_removes_the_noise_two_independent_means_would_carry():
    generator = np.random.default_rng(0)
    easy = generator.integers(0, 2, 400).astype(float)
    other = easy.copy()
    other[:4] = 1 - other[:4]
    _, low, high = report.paired_bootstrap(easy, other, iterations=500)
    assert high - low < 0.03


def test_the_same_seed_gives_the_same_interval():
    base, other = np.arange(50.0) % 2, np.arange(50.0) % 3 / 2
    assert report.paired_bootstrap(base, other, iterations=300) == report.paired_bootstrap(
        base, other, iterations=300
    )


def test_only_examples_with_the_same_prompt_in_every_arm_are_paired():
    first = {0: {"prompt_hash": "a"}, 1: {"prompt_hash": "b"}, 2: {"prompt_hash": "c"}}
    second = {0: {"prompt_hash": "a"}, 1: {"prompt_hash": "other"}, 3: {"prompt_hash": "d"}}
    assert report.shared_ids(first, second) == [0]


def test_a_report_tabulates_each_arm_and_its_difference_from_the_baseline(tmp_path):
    for arm, accuracies in (("dense", [1.0, 1.0, 0.0, 1.0]), ("window", [1.0, 0.0, 0.0, 0.0])):
        results = results_of("sciq", [logged(i, acc=value) for i, value in enumerate(accuracies)])
        records.write_records(tmp_path / arm / "core", results, {})
    text = report.report(tmp_path, "dense", ["core"])
    assert "| sciq (acc) | 0.750 | 0.250 |" in text
    assert "-0.500" in text


def test_a_ruler_report_gives_scores_by_length_and_the_window_arms_cells(tmp_path):
    for arm, score in (("dense", 1.0), ("window", 0.0)):
        for name in ("ruler_niah_single_1_4096", "ruler_niah_single_1_8192"):
            results = results_of(name, [logged(i, score=score if i else 1.0) for i in range(4)])
            records.write_records(tmp_path / arm / "ruler", results, {})
    text = report.report(tmp_path, "dense", ["ruler"], window="window")
    assert "| 4096 | 1.000 | 0.250 |" in text
    assert "scores above zero on 2 of 2 cells" in text
