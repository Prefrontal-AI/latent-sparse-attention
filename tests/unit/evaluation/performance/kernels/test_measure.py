"""The harness arithmetic a CPU can exercise: the accounting, the selection check and the admission.

Ported with the code, from the fork's `python/sglang/benchmark/test_attention_layer.py`. Every CUDA call
it needs is mocked and `Recorder` is built through `__new__`, so nothing here touches a device - which is
what lets `make check` cover the parts of the benchmark that decide whether a number is right.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.measure import layer
from lsa.evaluation.performance.kernels.measure import snapshot as snapshot_module
from lsa.evaluation.performance.kernels.measure.decode import maintenance_periods
from lsa.evaluation.performance.kernels.measure.instrument import wrap_pisa
from lsa.evaluation.performance.kernels.measure.recorder import REQUIRED, Recorder
from lsa.evaluation.performance.kernels.measure.selection import NoSelection, SelectionCheck
from lsa.evaluation.performance.kernels.measure.snapshot import (
    LinearSnapshot,
    PisaState,
    Snapshot,
    pisa_indexes,
)
from lsa.evaluation.performance.kernels.measure.variant import FAMILY_VARIANTS, KINDS, NoHandle, Variant
from lsa.evaluation.performance.kernels.measure.workload import memory_admission, score_rows


class Event:
    def __init__(self, ms):
        self.ms = ms

    def elapsed_time(self, other):
        return other.ms - self.ms


class AccountingTests(unittest.TestCase):
    def test_metadata_and_module_accounting_are_separate(self):
        recorder = Recorder.__new__(Recorder)
        recorder.stack = []
        recorder.ranges = [
            ("index k cluster", None),
            ("index k prep", 0),
            ("attention projections", None),
            ("index q scan", 2),
        ]
        recorder.events = [
            (Event(0), Event(7)),
            (Event(1), Event(3)),
            (Event(10), Event(14)),
            (Event(11), Event(12)),
        ]
        metadata = recorder.read(10, last=2)["exclusive_ms"]
        module = recorder.read(6, first=2)["exclusive_ms"]
        self.assertEqual(
            (metadata["index k cluster"], metadata["index k prep"], metadata["other"]),
            (5, 2, 3),
        )
        self.assertEqual(
            (module["attention projections"], module["index q scan"], module["other"]),
            (3, 1, 2),
        )
        self.assertEqual(sum(metadata.values()) + sum(module.values()), 16)
        with self.assertRaisesRegex(RuntimeError, "crosses"):
            recorder.read(6, first=3)

    def test_nested_ranges_and_gaps(self):
        recorder = Recorder.__new__(Recorder)
        recorder.stack = []
        recorder.ranges = [
            ("attention projections", None),
            ("index q scan", 0),
            ("topk attention", None),
        ]
        recorder.events = [
            (Event(0), Event(5)),
            (Event(1), Event(4)),
            (Event(6), Event(10)),
        ]
        times = recorder.read(11)["exclusive_ms"]
        self.assertEqual(times["attention projections"], 2)
        self.assertEqual(times["index q scan"], 3)
        self.assertEqual(times["topk attention"], 4)
        self.assertEqual(times["other"], 2)

    def test_invalid_total_fails(self):
        recorder = Recorder.__new__(Recorder)
        recorder.stack = []
        recorder.ranges = [("index q scan", None)]
        recorder.events = [(Event(0), Event(12))]
        with self.assertRaises(RuntimeError):
            recorder.read(11)

    def test_snapshot_restores_mutable_state(self):
        kv = torch.arange(16).reshape(4, 1, 4).clone()
        index = torch.arange(8).clone()
        tail = torch.arange(4).clone()
        pool = SimpleNamespace(
            get_key_buffer=lambda layer: kv,
            get_index_k_with_scale_buffer=lambda layer: index,
            _compress_tail_k=[tail],
        )
        module = SimpleNamespace(layer_id=3, indexer=SimpleNamespace(step=2))
        snapshot = Snapshot(module, pool, SimpleNamespace(out_cache_loc=torch.tensor([2, 3])))
        for _ in range(2):
            kv[2:] = -1
            index.fill_(-1)
            tail.fill_(-1)
            snapshot.restore()
            torch.testing.assert_close(kv, torch.arange(16).reshape(4, 1, 4))
            torch.testing.assert_close(index, torch.arange(8))
            torch.testing.assert_close(tail, torch.arange(4))

    def test_snapshot_restores_qsa_compressed_keys_and_its_open_block_ring(self):
        keys = torch.arange(16.0).reshape(4, 2, 2)
        compressed = torch.arange(6.0).reshape(3, 1, 2)
        ring = torch.arange(8.0).reshape(4, 1, 2)
        positions = torch.arange(12).reshape(4, 3)
        pool = SimpleNamespace(
            get_key_buffer=lambda layer: keys,
            get_value_buffer=lambda layer: keys.clone(),
            get_qsa_compressed_k_buffer=lambda layer: compressed,
            get_qsa_key_state_buffer=lambda layer: ring,
            qsa_rope_position_buffer=positions,
            qsa_compress_ratio=1,
        )
        snapshot = Snapshot(
            SimpleNamespace(layer_id=3, indexer=None), pool, SimpleNamespace(out_cache_loc=torch.tensor([1]))
        )
        assert snapshot.index.data_ptr() == compressed[1:].data_ptr()
        for tensor in (compressed, ring, positions):
            tensor.fill_(-1)
        with pytest.raises(AssertionError):
            snapshot.check()
        snapshot.restore()
        snapshot.check()
        torch.testing.assert_close(compressed[1:], torch.arange(2.0, 6.0).reshape(2, 1, 2))
        torch.testing.assert_close(ring[1:], torch.arange(2.0, 8.0).reshape(3, 1, 2))
        torch.testing.assert_close(positions[1:], torch.arange(3, 12).reshape(3, 3))
        # The dump rows no request owns, and the padding slot a graph step writes, are scratch
        ring[0] = 99
        compressed[0] = 99
        snapshot.check()

    def test_snapshot_restores_keys_and_values_of_a_pool_with_no_index(self):
        keys = torch.arange(16.0).reshape(4, 2, 2)
        values = -torch.arange(16.0).reshape(4, 2, 2)
        pool = SimpleNamespace(get_key_buffer=lambda layer: keys, get_value_buffer=lambda layer: values)
        module = SimpleNamespace(layer_id=3, indexer=None)
        snapshot = Snapshot(module, pool, SimpleNamespace(out_cache_loc=torch.tensor([1, 2])))
        assert snapshot.index is None
        assert snapshot.nbytes == 2 * 8 + 2 * (2 * 4 * 4)
        keys[1:3] = 7
        values[1:3] = 7
        with pytest.raises(AssertionError):
            snapshot.check()
        snapshot.restore()
        snapshot.check()
        assert values[2].flatten().tolist() == [-8.0, -9.0, -10.0, -11.0]

    def test_snapshot_does_not_ask_a_dense_pool_for_an_index(self):
        keys = torch.zeros(4, 2, 2)

        def refuse(layer):
            raise AssertionError("get_index_k_with_scale_buffer called when use_dsa is False")

        pool = SimpleNamespace(
            get_key_buffer=lambda layer: keys,
            get_value_buffer=lambda layer: keys.clone(),
            get_index_k_with_scale_buffer=refuse,
            use_dsa=False,
        )
        snapshot = Snapshot(
            SimpleNamespace(layer_id=3, indexer=None), pool, SimpleNamespace(out_cache_loc=torch.tensor([0]))
        )
        assert snapshot.index is None

    def test_snapshot_keeps_one_copy_of_a_latent_both_getters_return(self):
        latent = torch.arange(8.0).reshape(4, 1, 2)
        pool = SimpleNamespace(get_key_buffer=lambda layer: latent, get_value_buffer=lambda layer: latent)
        snapshot = Snapshot(
            SimpleNamespace(layer_id=3, indexer=None), pool, SimpleNamespace(out_cache_loc=torch.tensor([0]))
        )
        assert len(snapshot.buffers) == 1


class SelectionTests(unittest.TestCase):
    def setUp(self):
        indexer = SimpleNamespace(index_topk=8, index_kpool=4)
        self.check = SelectionCheck(SimpleNamespace(indexer=indexer), torch.arange(20), 20)
        self.check.reference = {
            "scores": torch.tensor([[10.0, 9.0, 9.0, 1.0, 99.0]]),
            "lengths": torch.tensor([4]),
        }
        self.check.prepare()

    def selection(self, *groups):
        return torch.tensor([[4 * group + offset for group in groups for offset in range(4)]])

    def test_boundary_ties_are_valid(self):
        for groups in ((0, 1), (2, 0)):
            self.assertTrue(self.check.check(self.selection(*groups)))

    def test_nonfinite_visible_scores_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.check.reference = {
                "scores": torch.tensor([[10.0, 9.0, value, 1.0, 99.0]]),
                "lengths": torch.tensor([4]),
            }
            with self.assertRaisesRegex(AssertionError, "Non-finite"):
                self.check.prepare()

    def test_invalid_topk(self):
        for groups in ((0, 3), (0, 4), (0, 0), (1, 2)):
            with self.subTest(groups=groups), self.assertRaises(AssertionError):
                self.check.check(self.selection(*groups))

    def test_incomplete_pool(self):
        selected = self.selection(0, 1)
        selected[0, 2] = 7
        with self.assertRaises(AssertionError):
            self.check.check(selected)

    def test_physical_mapping_and_causality(self):
        slots = torch.tensor([19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0])
        mapped = SelectionCheck(SimpleNamespace(indexer=self.check.indexer), slots, 20, torch.tensor([7]))
        mapped.reference = {
            "scores": torch.tensor([[10.0, 9.0, 9.0, 1.0, 99.0]]),
            "lengths": torch.tensor([4]),
        }
        mapped.prepare()
        self.assertTrue(mapped.check(slots[self.selection(0, 1)]))
        with self.assertRaisesRegex(AssertionError, "Non-causal"):
            mapped.check(slots[self.selection(0, 2)])

    def test_decode_ignores_unwritten_padding(self):
        indexer = SimpleNamespace(index_topk=8, index_kpool=4)
        check = SelectionCheck(SimpleNamespace(indexer=indexer), torch.arange(20), 20, decode=True)
        check.reference = {
            "scores": torch.tensor([[10.0, 9.0, 1.0, float("nan")]]),
            "lengths": torch.tensor([3]),
            "all_lengths": torch.tensor([3]),
        }
        actual = {key: tensor.clone() for key, tensor in check.reference.items()}
        actual["scores"][0, -1] = -77
        check.compare_scores(actual)
        actual["scores"][0, 1] = 8
        with self.assertRaisesRegex(AssertionError, "visible scores"):
            check.compare_scores(actual)
        check.prepare()
        self.assertTrue(check.check(self.selection(0, 1)))

    def test_sampled_scores_keep_full_causal_checks(self):
        self.check.rows = [0, 2]
        self.check.reference = {
            "scores": torch.tensor([[10.0, 9.0, 9.0, 1.0, 99.0]]).repeat(2, 1),
            "lengths": torch.tensor([4, 4]),
            "all_lengths": torch.tensor([4, 4, 4]),
        }
        self.check.prepare()
        chosen = self.selection(0, 1).repeat(3, 1)
        self.assertTrue(self.check.check(chosen))
        chosen[1] = self.selection(0, 4)[0]
        with self.assertRaisesRegex(AssertionError, "invisible"):
            self.check.check(chosen)

    def test_diagnostic_topk_is_explicitly_invalid(self):
        self.check.allow_invalid_topk = True
        self.assertFalse(self.check.check(self.selection(0, 3)))
        self.assertEqual(self.check.failures[0]["selected_below_cutoff"], 1)
        with self.assertRaises(AssertionError):
            self.check.check(self.selection(0, 4))

    def test_memory_admission_rejects_insufficient_space(self):
        with (
            patch("torch.cuda.mem_get_info", return_value=(100 * 2**30, 140 * 2**30)),
            patch("torch.cuda.memory_allocated", return_value=30 * 2**30),
            patch("torch.cuda.memory_reserved", return_value=31 * 2**30),
        ):
            native = memory_admission(2048, 10 * 2**20, "native", 128, 6)
            self.assertEqual(native["score_workspace_bound_bytes"], 20 * 2**30)
            with self.assertRaises(MemoryError):
                memory_admission(2048, 10 * 2**20, "native", 128, 90)

    def test_qsa_admits_a_32k_chunk_at_10m_under_its_fixed_score_tile(self):
        with (
            patch("torch.cuda.mem_get_info", return_value=(100 * 2**30, 140 * 2**30)),
            patch("torch.cuda.memory_allocated", return_value=30 * 2**30),
            patch("torch.cuda.memory_reserved", return_value=31 * 2**30),
        ):
            qsa = memory_admission(32768, 10 * 2**20, "qsa", 128, 6)
            self.assertEqual(qsa["score_workspace_bound_bytes"], 512 * 2**20)
            self.assertEqual(qsa["index_gather_bound_bytes"], 0)

    def test_dsa_s_logits_are_bounded_by_the_slice_the_indexer_reads_in(self):
        """A 32k chunk at 10M is a 320 GiB score matrix, which the indexer reads a slice of rows at a time"""
        with (
            patch("torch.cuda.mem_get_info", return_value=(100 * 2**30, 140 * 2**30)),
            patch("torch.cuda.memory_allocated", return_value=30 * 2**30),
            patch("torch.cuda.memory_reserved", return_value=31 * 2**30),
        ):
            native = memory_admission(32768, 10 * 2**20, "native", 128, 6)
            self.assertEqual(native["score_workspace_bound_bytes"], int(140 * 2**30 * 0.3))
            self.assertTrue(native["admitted"])

    def test_score_budget_includes_last_query(self):
        self.assertIsNone(score_rows(8192, 131072, "native", 1024))
        rows = score_rows(8192, 10485760, "native", 1024)
        self.assertEqual((rows[0], rows[-1]), (0, 8191))
        self.assertLessEqual(len(rows) * 10485760 // 4 * 4, 1024 * 2**20)


class VariantTests(unittest.TestCase):
    """The three kinds of layer, and what each of them does not have."""

    def test_layers_without_an_index_need_no_score_workspace(self):
        assert score_rows(8192, 10485760, "full", 128) is None
        assert score_rows(8192, 10485760, "kda", 128) is None
        with (
            patch("torch.cuda.mem_get_info", return_value=(10 * 2**30, 140 * 2**30)),
            patch("torch.cuda.memory_allocated", return_value=30 * 2**30),
            patch("torch.cuda.memory_reserved", return_value=31 * 2**30),
        ):
            for indexer in ("full", "kda"):
                admission = memory_admission(8192, 10 * 2**20, indexer, 128, 6)
                assert admission["score_workspace_bound_bytes"] == 0
                assert admission["index_gather_bound_bytes"] == 0

    def test_no_selection_is_vacuous(self):
        check = NoSelection()
        with check.capture(check.reference):
            pass
        check.compare_scores({})
        check.prepare()
        assert check.check(torch.empty(0, 0, dtype=torch.int32)) is None
        assert list(check.failures) == []
        # A layer with no index opens no index ranges, so requiring one would fail every trial
        assert "index q scan" not in REQUIRED["full"] + REQUIRED["linear"]

    def test_linear_snapshot_restores_only_the_request_rows(self):
        conv = torch.arange(24.0).reshape(3, 2, 4)
        state = torch.arange(12.0).reshape(3, 2, 2)
        cache = SimpleNamespace(conv=[conv], temporal=state)
        pool = SimpleNamespace(
            mamba2_layer_cache=lambda layer: cache,
            get_mamba_indices=lambda rows: torch.tensor([2]),
        )
        snapshot = LinearSnapshot(
            SimpleNamespace(layer_idx=4), pool, SimpleNamespace(req_pool_indices=torch.tensor([0]))
        )
        # One conv window and one recurrent state for the row - nothing that grows with the context
        assert snapshot.nbytes == (8 + 4) * 4
        conv[2] = -1
        state[2] = -1
        conv[0] = -2
        with pytest.raises(AssertionError):
            snapshot.check()
        snapshot.restore()
        snapshot.check()
        assert conv[2].flatten().tolist() == list(range(16, 24))
        # A row the request does not own is left exactly as it was found
        assert conv[0].flatten().tolist() == [-2.0] * 8

    def test_linear_snapshot_reads_a_gated_delta_net_layer_id(self):
        cache = SimpleNamespace(conv=[torch.zeros(2, 2)], temporal=torch.zeros(2, 2))
        layers = []

        def layer_cache(layer):
            layers.append(layer)
            return cache

        pool = SimpleNamespace(
            mamba2_layer_cache=layer_cache, get_mamba_indices=lambda rows: torch.tensor([1])
        )
        LinearSnapshot(SimpleNamespace(layer_id=4), pool, SimpleNamespace(req_pool_indices=torch.tensor([0])))
        assert layers == [4]


@pytest.mark.parametrize(
    ("family", "indexer", "nulled"),
    [
        ("glm5_next", "full", True),
        ("glm5_next", "native", False),
        ("glm5_next", "lsa", False),
        ("other", "full", False),
    ],
)
def test_only_glm_read_densely_is_built_without_its_indexer(monkeypatch, family, indexer, nulled):
    # Only GLM is built without an indexer, so any other family stands for the rest
    other = "another family"
    monkeypatch.setattr(
        layer, "checkpoint_family", lambda path: Family.GLM5_NEXT if family == "glm5_next" else other
    )
    rope = {"text_config": {"rope_parameters": {"factor": 40}}}
    args = SimpleNamespace(
        indexer=indexer, model_path="unused", model_override=json.dumps(rope), indexer_config=None
    )
    override = json.loads(layer.model_override(args))
    assert override["text_config"]["rope_parameters"] == {"factor": 40}
    assert ("index_topk" in override["text_config"]) is nulled
    assert override["text_config"].get("index_topk") is None
    # The native arm carries no settings block
    assert "lsa" not in override


def test_the_harness_hands_the_plugin_its_settings_as_the_model_config_block_beside_the_rotation(tmp_path):
    body = {"top_k": 2048, "projections": "/maps/x.pt"}
    path = tmp_path / "indexer.json"
    path.write_text(json.dumps(body))
    rope = {"text_config": {"rope_parameters": {"factor": 40}}}
    args = SimpleNamespace(
        indexer="lsa", model_path="unused", model_override=json.dumps(rope), indexer_config=str(path)
    )
    assert json.loads(layer.model_override(args)) == {**rope, "lsa": body}


class IndexerOutput(torch.nn.Module):
    """A layer's indexer, whose output is the selection the layer hands on"""

    def forward(self):
        return torch.tensor([[1, 2, -1]], dtype=torch.int32)


def hooked_variant(family, qsa, indexer, lsa=False):
    variant = Variant.__new__(Variant)
    variant.family, variant.qsa, variant.indexer, variant.lsa = family, qsa, indexer, lsa
    variant.name = "lsa" if lsa else "qsa" if qsa else "native"
    return variant


@pytest.mark.parametrize("family", [Family.GLM5_NEXT, Family.QWEN3_5])
def test_a_module_indexer_selection_is_captured_and_pinned(family):
    indexer = IndexerOutput()
    # GLM's native indexer and QSA both answer to a forward hook, and Qwen's QSA is the one Qwen design that does
    variant = hooked_variant(family, family == Family.QWEN3_5, indexer)
    captured = []
    handle = variant.selection_hook(lambda module, positional, output: captured.append(output.clone()))
    indexer()
    handle.remove()
    assert captured[0].tolist() == [[1, 2, -1]]
    pinned = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    with variant.pinned_selection(pinned):
        assert indexer().tolist() == [[7, 8, 9]]
    assert indexer().tolist() == [[1, 2, -1]]


def test_a_qwen_design_read_inside_the_backend_has_no_hook():
    indexer = IndexerOutput()
    variant = hooked_variant(Family.QWEN3_5, False, indexer)
    assert isinstance(variant.selection_hook(lambda *unused: None), NoHandle)
    with variant.pinned_selection(torch.zeros(1, 3, dtype=torch.int32)):
        assert indexer().tolist() == [[1, 2, -1]]


class CountedIndexer(IndexerOutput):
    """An indexer that counts the times it did its work"""

    def __init__(self):
        super().__init__()
        self.runs = 0

    def forward(self):
        self.runs += 1
        return super().forward()


@pytest.mark.parametrize("family", [Family.GLM5_NEXT, Family.QWEN3_5])
def test_a_free_indexer_does_no_work_and_hands_over_the_list(family):
    indexer = CountedIndexer()
    variant = hooked_variant(family, family == Family.QWEN3_5, indexer)
    seen = []
    handle = variant.selection_hook(lambda module, positional, output: seen.append(output))
    selected = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    with variant.free_indexer(selected) as served:
        assert indexer() is selected
        assert served == [selected]
    # The harness still reads the selection through its hook
    assert seen == [selected]
    assert indexer.runs == 0
    handle.remove()
    assert indexer().tolist() == [[1, 2, -1]]
    assert indexer.runs == 1


class BackendIndexer:
    """A Qwen design's indexer, whose `select` the backend calls and whose output no module hands on"""

    def __init__(self):
        self.runs = 0

    def select(self, q, k, forward_batch):
        self.runs += 1
        return torch.tensor([[1, 2, -1]], dtype=torch.int32)


def test_a_qwen_lsa_listing_is_seen_by_its_hook_and_leaves_the_checked_selection_alone():
    indexer = BackendIndexer()
    variant = hooked_variant(Family.QWEN3_5, False, indexer, lsa=True)
    seen = []
    handle = variant.listing_hook(lambda module, arguments, output: seen.append((module, output.tolist())))
    assert indexer.select(None, None, None).tolist() == [[1, 2, -1]]
    assert seen == [(indexer, [[1, 2, -1]])]
    assert not variant.hooked
    assert isinstance(variant.selection_hook(lambda *unused: None), NoHandle)
    handle.remove()
    indexer.select(None, None, None)
    assert len(seen) == 1
    assert "select" not in vars(indexer)


def test_a_module_indexer_listing_hook_is_its_selection_hook():
    indexer = IndexerOutput()
    variant = hooked_variant(Family.GLM5_NEXT, False, indexer)
    seen = []
    handle = variant.listing_hook(lambda module, positional, output: seen.append(output.tolist()))
    indexer()
    handle.remove()
    assert seen == [[[1, 2, -1]]]
    assert variant.hooked


@pytest.mark.parametrize(
    ("family", "qsa", "lsa", "frees"),
    [
        (Family.GLM5_NEXT, False, False, True),
        (Family.GLM5_NEXT, False, True, True),
        (Family.QWEN3_5, True, False, True),
        (Family.QWEN3_5, False, True, True),
        (Family.QWEN3_5, False, False, False),
    ],
)
def test_which_designs_free_their_indexer(family, qsa, lsa, frees):
    variant = hooked_variant(family, qsa, object(), lsa=lsa)
    assert variant.frees_indexer is frees


def test_a_layer_without_an_indexer_frees_none():
    assert not hooked_variant(Family.QWEN3_5, False, None, lsa=True).frees_indexer


def test_a_free_qwen_lsa_indexer_does_no_work_and_hands_over_the_pinned_list():
    indexer = BackendIndexer()
    variant = hooked_variant(Family.QWEN3_5, False, indexer, lsa=True)
    seen = []
    handle = variant.listing_hook(lambda module, arguments, output: seen.append(output))
    pinned = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    with variant.free_indexer(pinned) as served:
        assert indexer.select(None, None, None) is pinned
        pinned.copy_(torch.tensor([[4, 5, 6]], dtype=torch.int32))
        assert indexer.select(None, None, None).tolist() == [[4, 5, 6]]
        assert len(served) == 2
    assert indexer.runs == 0
    assert seen == []
    # The hook the free indexer stood in front of is back
    assert indexer.select(None, None, None).tolist() == [[1, 2, -1]]
    assert len(seen) == 1
    handle.remove()
    assert "select" not in vars(indexer)


def test_a_qwen_pisa_read_inside_the_backend_cannot_free_its_indexer():
    variant = hooked_variant(Family.QWEN3_5, False, BackendIndexer())
    variant.name = "pisa"
    assert not variant.frees_indexer
    with pytest.raises(ValueError, match="no free indexer"), variant.free_indexer(torch.zeros(1, 3)):
        pass


class PisaTests(unittest.TestCase):
    """PISA is timed beside LSA, with its own state to restore and no score matrix to hold."""

    @staticmethod
    def leaf_means():
        return SimpleNamespace(
            means=torch.arange(24.0).reshape(3, 4, 2),
            upper=torch.arange(18.0).reshape(3, 3, 2),
            closed=[0, 3, 0],
            rids=[None, "a", None],
        )

    def test_the_means_of_the_batch_rows_are_restored_and_the_others_left_alone(self):
        leaf_means = self.leaf_means()
        state = PisaState([SimpleNamespace(leaf_means=leaf_means)], [1])
        assert state.nbytes == (4 + 3) * 2 * 4
        leaf_means.means[1] = -1
        leaf_means.upper[1] = -1
        leaf_means.means[2] = -2
        leaf_means.closed[1] = 4
        assert state.changed(1) == {"pool_close": True}
        assert state.changed(0) == {"pool_close": False}
        with pytest.raises(AssertionError):
            state.check()
        state.restore()
        state.check()
        assert leaf_means.means[1].flatten().tolist() == list(range(8, 16))
        assert leaf_means.upper[1].flatten().tolist() == list(range(6, 12))
        assert leaf_means.means[2].flatten().tolist() == [-2.0] * 8
        assert leaf_means.closed == [0, 3, 0]

    def test_a_restored_host_field_is_checked(self):
        leaf_means = self.leaf_means()
        state = PisaState([SimpleNamespace(leaf_means=leaf_means)], [1])
        leaf_means.rids[1] = "b"
        with pytest.raises(AssertionError, match="host fields"):
            state.check()

    def test_the_snapshot_of_a_pisa_layer_holds_its_means_and_no_native_index(self):
        leaf_means = self.leaf_means()
        indexer = SimpleNamespace(is_pisa=True, ready=lambda: SimpleNamespace(leaf_means=leaf_means))
        pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: torch.zeros(8, 1, 2),
            get_value_buffer=lambda layer_id: torch.zeros(8, 1, 2),
        )
        batch = SimpleNamespace(out_cache_loc=torch.tensor([0, 1]), req_pool_indices=torch.tensor([1]))
        with patch.object(
            snapshot_module, "kv_buffers", lambda pool, layer_id: [pool.get_key_buffer(layer_id)]
        ):
            snapshot = Snapshot(SimpleNamespace(layer_id=3, indexer=indexer), pool, batch)
        assert isinstance(snapshot.state, PisaState)
        assert snapshot.index is None

    def test_it_has_its_own_required_components_and_no_query_scan(self):
        assert KINDS["pisa"] == "sparse"
        assert "index q scan" not in REQUIRED["pisa"]
        assert {"index k select", "topk attention"} <= set(REQUIRED["pisa"])

    def test_both_model_families_have_it(self):
        assert all("pisa" in variants for variants in FAMILY_VARIANTS.values())

    def test_a_layer_with_an_index_a_key_head_saves_and_restores_each(self):
        heads = [self.leaf_means(), self.leaf_means()]
        state = PisaState([SimpleNamespace(leaf_means=leaf) for leaf in heads], [1])
        assert state.nbytes == 2 * (4 + 3) * 2 * 4
        heads[1].means[1] = -1
        heads[1].closed[1] = 4
        assert state.changed(1) == {"pool_close": True}
        with pytest.raises(AssertionError):
            state.check()
        state.restore()
        state.check()
        assert heads[1].closed == [0, 3, 0]

    def test_the_indexes_of_one_head_or_many_are_a_list(self):
        one, many = SimpleNamespace(), [SimpleNamespace(), SimpleNamespace()]
        assert pisa_indexes(SimpleNamespace(ready=lambda: one)) == [one]
        assert pisa_indexes(SimpleNamespace(ready=lambda: many)) == many

    def test_every_key_heads_search_is_charged(self):
        charged = []
        indexes = [SimpleNamespace(), SimpleNamespace()]
        wrap_pisa(
            SimpleNamespace(ready=lambda: indexes), lambda obj, at, name, stack: charged.append(obj), None
        )
        assert [id(obj) for obj in charged] == [id(i) for i in indexes for _ in range(3)]

    def test_it_gathers_no_score_matrix(self):
        assert score_rows(8192, 10485760, "pisa", 128) is None
        with (
            patch("torch.cuda.mem_get_info", return_value=(10 * 2**30, 140 * 2**30)),
            patch("torch.cuda.memory_allocated", return_value=30 * 2**30),
            patch("torch.cuda.memory_reserved", return_value=31 * 2**30),
        ):
            admission = memory_admission(8192, 10 * 2**20, "pisa", 128, 6)
        assert admission["score_workspace_bound_bytes"] == 0

    def test_a_leaf_block_closing_is_its_one_maintenance_period(self):
        variant = SimpleNamespace(
            indexer=SimpleNamespace(config=SimpleNamespace(block_size=64)), lsa=False, pisa=True
        )
        assert maintenance_periods(variant) == {"pool_close": 64}

    def test_a_pisa_layer_checks_no_selection_and_holds_no_oracle(self):
        variant = Variant.__new__(Variant)
        variant.indexer, variant.pisa, variant.qsa = SimpleNamespace(), True, False
        assert isinstance(variant.selection(None), NoSelection)
        result = {}
        variant.oracle(result, decode=False)
        assert result["passed"] is None

    def test_the_search_and_the_key_write_are_charged_to_components(self):
        charged = []

        def wrap(obj, attribute, name, stack):
            charged.append((attribute, name))

        index = SimpleNamespace()
        wrap_pisa(SimpleNamespace(ready=lambda: index), wrap, None)
        assert charged == [
            ("write", "index k prep"),
            ("select", "index k select"),
            ("decode", "index k select"),
        ]


@pytest.mark.parametrize(("indexer", "backends"), [("native", {}), ("full", {"attention_backend": "fa3"})])
def test_qwen_qsa_names_no_backend_as_it_is_served(monkeypatch, indexer, backends):
    # A Qwen checkpoint's family
    monkeypatch.setattr(layer, "checkpoint_family", lambda path: Family.QWEN3_5)
    assert layer.engine_backends(SimpleNamespace(indexer=indexer, model_path="unused")) == backends


@pytest.mark.parametrize(
    ("indexer", "page"), [("pisa", {"page_size": 1}), ("lsa", {"page_size": 1}), ("full", {})]
)
def test_qwen_lists_are_read_at_a_page_of_one(monkeypatch, indexer, page):
    monkeypatch.setattr(layer, "checkpoint_family", lambda path: Family.QWEN3_5)
    backends = layer.engine_backends(SimpleNamespace(indexer=indexer, model_path="unused"))
    assert backends == {"attention_backend": "fa3", **page}


def served(monkeypatch, generation, family, indexer, **named):
    monkeypatch.setattr(layer, "checkpoint_family", lambda path: family)
    monkeypatch.setattr(layer, "device_generation", lambda: generation)
    args = SimpleNamespace(
        indexer=indexer,
        model_path="unused",
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_sparse",
        **named,
    )
    return layer.engine_backends(args)


@pytest.mark.parametrize(
    ("generation", "family", "indexer", "backends"),
    [
        (
            "hopper",
            Family.GLM5_NEXT,
            "full",
            {"prefill_attention_backend": "fa3", "decode_attention_backend": "flashmla"},
        ),
        (
            "blackwell",
            Family.GLM5_NEXT,
            "full",
            {"prefill_attention_backend": "trtllm_mla", "decode_attention_backend": "triton"},
        ),
        ("hopper", Family.QWEN3_5, "full", {"attention_backend": "fa3"}),
        ("blackwell", Family.QWEN3_5, "full", {"attention_backend": "trtllm_mha"}),
        (
            "hopper",
            Family.GLM5_NEXT,
            "native",
            {
                "attention_backend": "dsa",
                "dsa_prefill_backend": "flashmla_sparse",
                "dsa_decode_backend": "fa3",
            },
        ),
        (
            "blackwell",
            Family.GLM5_NEXT,
            "native",
            {
                "attention_backend": "dsa",
                "dsa_prefill_backend": "flashmla_sparse",
                "dsa_decode_backend": "trtllm",
            },
        ),
    ],
)
def test_a_baseline_arm_runs_upstream_sglangs_backend_for_the_gpu(
    monkeypatch, generation, family, indexer, backends
):
    assert served(monkeypatch, generation, family, indexer) == backends


@pytest.mark.parametrize("generation", ["hopper", "blackwell"])
def test_a_design_arm_reads_the_backends_the_config_names(monkeypatch, generation):
    # Ours, so upstream's default does not reach them
    backends = served(monkeypatch, generation, Family.GLM5_NEXT, "lsa")
    assert backends["dsa_prefill_backend"] == backends["dsa_decode_backend"] == "flashmla_sparse"


def test_a_named_backend_replaces_upstreams_for_one_phase(monkeypatch):
    named = {"prefill_attention_backend": None, "decode_attention_backend": "flashinfer"}
    assert served(monkeypatch, "blackwell", Family.GLM5_NEXT, "full", **named) == {
        "prefill_attention_backend": "trtllm_mla",
        "decode_attention_backend": "flashinfer",
    }
    named = {"prefill_attention_backend": None, "decode_attention_backend": "flashmla_sparse"}
    native = served(monkeypatch, "blackwell", Family.GLM5_NEXT, "native", **named)
    assert (native["dsa_prefill_backend"], native["dsa_decode_backend"]) == (
        "flashmla_sparse",
        "flashmla_sparse",
    )


def test_a_design_arm_refuses_a_named_backend(monkeypatch):
    with pytest.raises(ValueError, match="not a baseline"):
        served(monkeypatch, "hopper", Family.GLM5_NEXT, "pisa", decode_attention_backend="fa3")


@pytest.mark.parametrize(
    ("indexer", "family", "named", "source"),
    [
        ("full", Family.GLM5_NEXT, {}, "upstream"),
        ("native", Family.GLM5_NEXT, {"decode_attention_backend": "triton"}, "override"),
        ("pisa", Family.GLM5_NEXT, {}, "design"),
        ("native", Family.QWEN3_5, {}, "fork"),
    ],
)
def test_the_record_says_where_the_backends_came_from(monkeypatch, indexer, family, named, source):
    monkeypatch.setattr(layer, "checkpoint_family", lambda path: family)
    args = SimpleNamespace(indexer=indexer, model_path="unused", **named)
    assert layer.backend_source(args) == source


def test_a_hybrid_backend_is_named_by_the_class_of_each_phase():
    class Hybrid:
        prefill_backend, decode_backend = FlashInfer(), Fa3()

    assert layer.backend_classes(Hybrid()) == {"prefill": "FlashInfer", "decode": "Fa3"}
    assert layer.backend_classes(Fa3()) == {"prefill": "Fa3", "decode": "Fa3"}


class FlashInfer:
    pass


class Fa3:
    pass
