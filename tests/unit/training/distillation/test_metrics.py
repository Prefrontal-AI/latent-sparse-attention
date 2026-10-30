"""What a read captures - the padding left out, the shares, the exact read, and the budget it is measured at."""

import math
from types import SimpleNamespace

import pytest
import torch

from lsa.training.distillation.constants import Report
from lsa.training.distillation.data.cache import Document, layer_path
from lsa.training.distillation.metrics import (
    depth_metrics,
    exact_read,
    held,
    key_read,
    mean_reports,
    native_overlap,
    reach_metrics,
    row_metrics,
)
from lsa.training.distillation.model import IndexMap, init_rows
from tests.helpers.training.distillation.data import fake_cache

HEADS = 4
METRICS = {
    "mass",
    "mass_top",
    "recall16",
    "recall64",
    "recall",
    "peak",
    "teacher_ids_read",
    "ids_read",
}
SHARES = METRICS - {"teacher_ids_read", "ids_read"}


def scored(tmp_path, **overrides):
    fake_cache.collection(tmp_path)
    config = fake_cache.config(**overrides)
    document = Document(layer_path(tmp_path, "tokens", "train", "source0", 0), config)
    return document, config


def index_space(document, rows=None):
    """The document's index queries at some rows and its index keys, in the space of a PCA start"""
    rows = document.eval_rows() if rows is None else rows
    index_map = IndexMap(document.model_dim, fake_cache.INDEX_DIM, init=init_rows(document.x))
    return index_map.index_queries(document.query_inputs(rows)), index_map.index_keys(document.x)


def test_a_teacher_entry_of_no_mass_is_padding_even_where_the_read_holds_its_id():
    # The row sees two keys - the cache pads its list with a sink and a window key, which every read holds
    selected = torch.tensor([[7, 0, 1, 2, 3, 30, 31, -1]])
    top_key_ids = torch.tensor([[7, 9, 0, 30]])
    top_key_mass = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    assert held(selected, top_key_ids, top_key_mass > 0).tolist() == [[True, False, False, False]]
    metrics = row_metrics(selected, top_key_ids, top_key_mass, HEADS)
    assert metrics["teacher_ids_read"].item() == 1
    assert metrics["recall"].item() == 0.5
    assert metrics["recall16"].item() == 0.5


def test_padding_is_no_miss_of_a_row_with_fewer_teacher_keys_than_the_depth():
    # Two visible keys, both read - the 62 padding entries past them do not count against recall64
    top_key_ids = torch.cat([torch.tensor([10, 11]), torch.zeros(62, dtype=torch.long)])[None]
    top_key_mass = torch.cat([torch.tensor([1.0, 0.5]), torch.zeros(62)])[None]
    metrics = row_metrics(torch.tensor([[10, 11, -1]]), top_key_ids, top_key_mass, HEADS)
    assert metrics["recall16"].item() == 1.0
    assert metrics["recall64"].item() == 1.0


def test_mass_is_the_share_per_head_and_mass_top_the_share_of_the_cached_mass():
    selected = torch.tensor([[5, 6, -1, -1], [6, 8, 5, -1]])
    top_key_ids = torch.tensor([[5, 8, 9], [8, 6, 7]])
    top_key_mass = torch.tensor([[1.0, 0.5, 0.5], [2.0, 1.0, 0.0]])
    metrics = row_metrics(selected, top_key_ids, top_key_mass, HEADS)
    assert metrics["mass"].tolist() == [0.25, 0.75]
    assert metrics["mass_top"].tolist() == [0.5, 1.0]
    assert metrics["recall"].tolist() == pytest.approx([1 / 3, 1.0])
    assert metrics["teacher_ids_read"].tolist() == [1, 2]
    assert metrics["ids_read"].tolist() == [2, 3]


def test_a_recall_counts_the_teachers_top_keys_at_its_depth_alone():
    top_key_ids = torch.arange(100, 400)[None]
    top_key_mass = torch.linspace(2, 1, 300)[None]
    # The first 16 and every key past 64
    selected = torch.cat([torch.arange(100, 116), torch.arange(164, 400)])[None]
    metrics = row_metrics(selected, top_key_ids, top_key_mass, HEADS)
    assert metrics["recall16"].item() == 1.0
    assert metrics["recall64"].item() == 0.25
    assert metrics["recall"].item() == pytest.approx((16 + 236) / 300)


def test_a_row_whose_teacher_has_no_key_has_no_recall():
    metrics = row_metrics(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), torch.zeros(1, 2), HEADS)
    assert math.isnan(metrics["recall"].item())
    assert metrics["mass"].item() == 0


def brute_force_read(queries, keys, position, top_k, sink_size, window_size):
    scores = queries @ keys.float().t()
    visible = [key for key in range(len(keys)) if key >= sink_size and position - key >= window_size]
    best = sorted(visible, key=lambda key: -scores[key].item())[:top_k]
    window_start = max(0, position + 1 - window_size)
    return set(best) | set(range(window_start, position + 1)) | set(range(min(sink_size, window_start)))


def test_the_exact_read_is_the_best_visible_keys_then_the_window_and_the_sinks():
    generator = torch.Generator().manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(60, 8, generator=generator), dim=-1).bfloat16()
    positions = torch.tensor([3, 9, 10, 25, 59], dtype=torch.int32)
    queries = torch.nn.functional.normalize(torch.randn(len(positions), 8, generator=generator), dim=-1)
    lists = exact_read(queries, keys, positions, top_k=5, sink_size=2, window_size=8, chunk_size=2)
    assert lists.shape == (5, 5 + 8 + 2)
    for row, position in enumerate(positions.tolist()):
        ids = lists[row][lists[row] >= 0].tolist()
        assert len(ids) == len(set(ids))
        assert set(ids) == brute_force_read(queries[row : row + 1][0], keys, position, 5, 2, 8)


def test_an_exact_read_past_its_budget_reads_top_k_long_range_keys():
    keys = torch.nn.functional.normalize(torch.randn(100, 8), dim=-1).bfloat16()
    lists = exact_read(torch.randn(1, 8), keys, torch.tensor([99]), top_k=10, sink_size=4, window_size=16)
    assert int((lists >= 0).sum()) == 10 + 16 + 4


def test_a_read_is_scored_at_the_budget_the_config_names(tmp_path):
    document, config = scored(tmp_path)
    queries, keys = index_space(document)
    report = key_read(document, queries, keys, config.top_k)
    assert METRICS <= set(report)
    for name in SHARES:
        assert 0.0 <= report[name] <= 1.0


def test_a_budget_that_takes_every_readable_key_captures_everything(tmp_path):
    document, _ = scored(tmp_path)
    queries, keys = index_space(document)
    report = key_read(document, queries, keys, document.key_count)
    assert report["recall16"] == 1.0
    assert report["recall64"] == 1.0
    assert report["peak"] == 1.0
    assert report["mass_top"] == pytest.approx(1.0)
    assert report["mass"] > 0.99


def test_a_wider_budget_never_captures_less(tmp_path):
    document, narrow = scored(tmp_path, top_k=fake_cache.BUDGET)
    queries, keys = index_space(document)
    assert (
        key_read(document, queries, keys, 4 * fake_cache.BUDGET)["mass"]
        >= key_read(document, queries, keys, narrow.top_k)["mass"]
    )


def test_the_rows_a_read_scores_are_the_ones_it_is_given(tmp_path):
    document, config = scored(tmp_path)
    rows = document.eval_rows()[:2]
    queries, keys = index_space(document, rows)
    part = key_read(document, queries, keys, config.top_k, rows)
    assert METRICS <= set(part)


def test_a_tree_of_reports_is_averaged_leaf_by_leaf():
    first = {"a": {"mass": 0.2}, "b": 1.0}
    second = {"a": {"mass": 0.6}, "b": 3.0}
    even = mean_reports([first, second])
    assert even["a"]["mass"] == pytest.approx(0.4)
    assert even["b"] == pytest.approx(2.0)
    weighted = mean_reports([first, second], [3, 1])
    assert weighted["a"]["mass"] == pytest.approx(0.3)
    assert weighted["b"] == pytest.approx(1.5)


def test_reach_counts_the_far_keys_read_and_the_far_mass_captured():
    far = Report.FAR_BEHIND
    position = 3 * far
    # A sink, two far keys, a near one and a window key, -1 padded
    read = torch.tensor([[0, position - far - 5, position - 2 * far, position - 600, position - 3, -1]])
    teacher_ids = torch.tensor([[position - far - 5, position - far - 6, position - 600, 7]])
    teacher_mass = torch.tensor([[1.0, 1.0, 5.0, 0.0]])
    reach = reach_metrics(
        read, torch.tensor([position]), teacher_ids, teacher_mass, sink_size=4, window_size=512
    )
    assert math.isclose(float(reach["far_share"][0]), 2 / 3, rel_tol=1e-6)
    assert math.isclose(float(reach["far_mass"][0]), 0.5, rel_tol=1e-6)


def test_native_overlap_counts_glms_long_range_picks_alone():
    position = 20000
    picked = torch.tensor([[0, position - 1, position - 600, position - 9000, -1]])
    read = torch.tensor([[position - 600, 5, 6]])
    overlap = native_overlap(read, picked, torch.tensor([position]), sink_size=4, window_size=512)
    assert math.isclose(float(overlap["native_overlap"][0]), 0.5, rel_tol=1e-6)
    assert float(overlap["native_far_overlap"][0]) == 0.0


def a_teacher(ids, mass, heads):
    return SimpleNamespace(
        device="cpu",
        head_count=heads,
        top_key_ids=torch.tensor([ids]),
        top_key_mass=torch.tensor([mass]),
    )


def test_a_depth_scores_the_reads_front_against_the_teachers_front():
    teacher = a_teacher([1, 2, 3], [3.0, 2.0, 1.0], heads=10)
    read = torch.tensor([[1, 9, 2, 8]])
    metrics = depth_metrics(teacher, torch.tensor([0]), read, depth=2)
    # The read of two is keys 1 and 9, the teacher's top two are 1 and 2
    assert metrics["recall"] == 0.5
    assert metrics["stored_held"] == pytest.approx(1 / 3)
    assert metrics["mass_lower"] == pytest.approx(0.3)


def test_the_mass_a_read_may_hold_off_the_list_is_capped_by_the_lists_weakest_key():
    teacher = a_teacher([1, 2, 3], [3.0, 2.0, 1.0], heads=10)
    read = torch.tensor([[1, 9, 8, 7]])
    # The list holds 6 of 10, so 4 is unseen - but three picks off the list weigh at most 1 each
    wide = depth_metrics(teacher, torch.tensor([0]), read, depth=4)
    assert wide["mass_lower"] == pytest.approx(0.3)
    assert wide["mass_upper"] == pytest.approx(0.6)
    assert wide["mass_width"] == pytest.approx(0.3)
    assert wide["outside_share"] == pytest.approx(0.75)
    # One pick off the list weighs at most 1
    narrow = depth_metrics(teacher, torch.tensor([0]), read, depth=2)
    assert narrow["mass_upper"] == pytest.approx(0.4)


def test_a_depth_past_the_list_has_no_recall_and_a_list_that_is_everything_has_no_unseen_mass():
    teacher = a_teacher([1, 2, 3], [3.0, 2.0, 1.0], heads=6)
    metrics = depth_metrics(teacher, torch.tensor([0]), torch.tensor([[1, 2, 3, 9]]), depth=4)
    assert "recall" not in metrics
    assert metrics["stored_held"] == 1.0
    assert metrics["mass_lower"] == pytest.approx(1.0)
    assert metrics["mass_upper"] == pytest.approx(1.0)


def test_a_depth_in_a_report_is_the_exact_read_at_that_budget(tmp_path):
    document, _ = scored(tmp_path)
    queries, keys = index_space(document)
    depth = 2 * fake_cache.BUDGET
    plain = key_read(document, queries, keys, depth)
    deep = key_read(document, queries, keys, fake_cache.BUDGET, depths=[depth])
    at = deep[Report.DEPTHS][str(depth)]
    # Every stored key the read holds, and the mass of them, are what a read at that budget alone reports
    assert at["stored_held"] == pytest.approx(plain["recall"])
    assert at["mass_lower"] == pytest.approx(plain["mass"])
    assert at["mass_lower"] <= at["mass_upper"] <= 1.0 + 1e-6


def test_the_depths_of_a_report_are_one_exact_read_each(tmp_path):
    document, _ = scored(tmp_path)
    queries, keys = index_space(document)
    report = key_read(document, queries, keys, fake_cache.BUDGET, depths=[16, 64])
    assert set(report[Report.DEPTHS]) == {"16", "64"}
    # Recall at K against a read of K, the k-vs-k the first sixteen of a list of sixty-four are scored by
    alone = key_read(document, queries, keys, 16)
    assert report[Report.DEPTHS]["16"]["recall"] == pytest.approx(alone["recall16"])
