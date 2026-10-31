"""The prefill selection's buffers are one set for every layer, and a top-k hands its listing back after a read."""

import weakref

import torch

from lsa.core.kernels.triton.top_k import TopK
from lsa.sglang.index import selector as selector_module
from lsa.sglang.index.selector import Selector, clear_scratches, scratch_for
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

SETTINGS = DesignSettings.from_blocks({"input": "attention_input"}, {})
TOP_P = SETTINGS.indexer.top_p


def make_selector():
    slots = 2 * PAGE_SIZE
    table = torch.zeros(1, slots, dtype=torch.int32)
    return Selector(SETTINGS, slots, 1, slots, 512, "cpu", lambda: table)


def test_layers_on_one_device_share_the_selection_buffers():
    clear_scratches()
    first, second = make_selector(), make_selector()
    assert first.scan_select is second.scan_select
    assert first.prefill_top_k is second.prefill_top_k
    assert first.tree_top_k is second.tree_top_k
    assert first.beam_top_k(2048) is second.beam_top_k(2048)
    assert first.beam_select(2048) is second.beam_select(2048)
    # A decode step's listing stays a layer's own, since a captured graph writes to it
    assert first.decode_top_k is not second.decode_top_k
    assert len(selector_module.SCRATCHES) == 1


def test_a_scratch_is_a_chunk_and_a_width_and_a_device_and_a_kernel_language():
    clear_scratches()
    assert scratch_for(TOP_P, 512, "cpu") is scratch_for(TOP_P, 512, "cpu")
    assert scratch_for(TOP_P, 512, "cpu") is not scratch_for(TOP_P, 1024, "cpu")
    assert scratch_for(TOP_P, 512, "cpu") is not scratch_for(TOP_P - 1, 512, "cpu")
    assert scratch_for(TOP_P, 512, "cpu") is not scratch_for(TOP_P, 512, "cpu", "cutedsl")


def test_a_design_names_the_kernel_language_its_selects_run():
    clear_scratches()
    assert make_selector().scan_select.kernels == "triton"
    settings = DesignSettings.from_blocks({"input": "attention_input"}, {"kernels": "cutedsl"})
    slots = 2 * PAGE_SIZE
    table = torch.zeros(1, slots, dtype=torch.int32)
    selector = Selector(settings, slots, 1, slots, 512, "cpu", lambda: table)
    assert selector.scan_select.kernels == "cutedsl"
    assert selector.beam_select(2048).kernels == "cutedsl"


def test_clearing_the_scratches_frees_what_they_held():
    clear_scratches()
    scan_select = weakref.ref(scratch_for(TOP_P, 512, "cpu").scan_select)
    clear_scratches()
    assert scan_select() is None


def test_releasing_a_top_k_lets_its_listing_go():
    pick = TopK(TOP_P)
    keys = torch.zeros(512, 512, dtype=torch.int64)
    pick.keys, pick.counts = keys, torch.zeros(512, dtype=torch.int32)
    listing = weakref.ref(keys)
    del keys
    assert listing() is not None
    pick.release()
    assert listing() is None
    # Twice does no harm
    pick.release()


def test_a_listing_in_buffers_sized_once_is_kept_through_a_release():
    pick = TopK(TOP_P, 4, "cpu", exact=True)
    pick.keys, pick.counts = pick.held[0][:2], pick.held[1][:2]
    pick.release()
    assert pick.keys.shape[0] == 2
