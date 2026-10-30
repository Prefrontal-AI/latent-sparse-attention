"""The decode graphs a context range each - which graph a batch replays and which range a capture is sized to.

On the host alone. The reads of a range graph are `test_selector`, the hook that hands the ranges to sglang's
runner is `test_plugin`.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lsa.sglang import context_buckets
from lsa.sglang.context_buckets import FULL, ContextBuckets, captured_limit, label_of, pages_of
from lsa.sglang.settings.design import DECODE_CONTEXT_LIMITS, PAGE_SIZE, ServingChoices

LIMITS = (131072, 1048576, 4194304)


def batch(*lengths):
    return SimpleNamespace(seq_lens_cpu=torch.tensor(lengths))


def capturing(monkeypatch, label):
    capture_mode = ModuleType(context_buckets.CAPTURE_MODE)
    capture_mode.get_capture_attention_variant = lambda: label
    monkeypatch.setitem(sys.modules, context_buckets.CAPTURE_MODE, capture_mode)


def test_the_batch_replays_the_narrowest_range_its_longest_request_fits():
    buckets = ContextBuckets(LIMITS)
    assert buckets.select(batch(100, 131072)) == label_of(131072)
    assert buckets.select(batch(100, 131073)) == label_of(1048576)
    assert buckets.select(batch(1048576)) == label_of(1048576)
    assert buckets.select(batch(1048577)) == label_of(4194304)
    assert buckets.select(batch(4194305)) == FULL


def test_a_batch_whose_lengths_are_not_on_the_host_replays_the_full_graph():
    buckets = ContextBuckets(LIMITS)
    assert buckets.select(SimpleNamespace(seq_lens_cpu=None)) == FULL
    assert buckets.select(SimpleNamespace()) == FULL
    assert buckets.select(batch()) == FULL


def test_every_range_and_the_full_graph_are_captured_the_widest_first():
    assert ContextBuckets(LIMITS).capture_labels == (FULL, *(label_of(limit) for limit in reversed(LIMITS)))


def test_a_range_is_kept_only_where_it_is_narrower_than_the_server():
    assert ContextBuckets.under(LIMITS, 10_500_000) == ContextBuckets(LIMITS)
    assert ContextBuckets.under(LIMITS, 1_048_576) == ContextBuckets((131072,))
    assert ContextBuckets.under(LIMITS, 1_052_672) == ContextBuckets((131072, 1048576))
    assert ContextBuckets.under(LIMITS, 131072) is None
    assert ContextBuckets.under((), 10_500_000) is None


def test_a_capture_is_sized_to_the_range_it_is_labelled_with(monkeypatch):
    capturing(monkeypatch, label_of(131072))
    assert captured_limit() == 131072


@pytest.mark.parametrize("label", [None, FULL, "dense"])
def test_the_full_graph_an_eager_step_and_another_variant_are_sized_to_the_server(monkeypatch, label):
    capturing(monkeypatch, label)
    assert captured_limit() is None


def test_without_sglang_loaded_nothing_is_being_captured(monkeypatch):
    monkeypatch.delitem(sys.modules, context_buckets.CAPTURE_MODE, raising=False)
    assert captured_limit() is None


def test_an_image_whose_runner_keys_no_variant_captures_the_one_graph(monkeypatch):
    monkeypatch.setitem(sys.modules, context_buckets.CAPTURE_MODE, ModuleType(context_buckets.CAPTURE_MODE))
    assert captured_limit() is None


def test_a_range_holds_the_pages_its_keys_fill():
    assert pages_of(PAGE_SIZE) == 1
    assert pages_of(PAGE_SIZE + 1) == 2
    assert pages_of(131072) == 16


def test_the_default_ranges_are_ascending_pages():
    assert ServingChoices().decode_context_limits == DECODE_CONTEXT_LIMITS
    assert list(DECODE_CONTEXT_LIMITS) == sorted(set(DECODE_CONTEXT_LIMITS))


def test_the_ranges_are_read_from_a_list_and_may_be_none():
    two = ServingChoices.from_mapping({"decode_context_limits": [PAGE_SIZE, 2 * PAGE_SIZE]})
    assert two.decode_context_limits == (PAGE_SIZE, 2 * PAGE_SIZE)
    assert ServingChoices.from_mapping({"decode_context_limits": []}).decode_context_limits == ()


@pytest.mark.parametrize(
    "limits", [[PAGE_SIZE, PAGE_SIZE], [2 * PAGE_SIZE, PAGE_SIZE], [PAGE_SIZE - 1], ["a"]]
)
def test_ranges_that_are_not_ascending_pages_are_refused(limits):
    with pytest.raises(ValueError, match="decode_context_limits"):
        ServingChoices(decode_context_limits=limits)
