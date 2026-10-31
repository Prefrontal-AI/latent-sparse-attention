"""The width a TileLang read is handed when an early budget makes the lists shorter.

The kernel walks every 64-column block of its lists and skips only `-1`, so an early budget saves the read only when
the lists reach it narrower. A cut that dropped a real entry would be a silent wrong answer, so the width decision is
tested on its own, apart from the kernel.
"""

import pytest
import torch

from lsa.constants import Sentinel
from lsa.sglang import plugin
from lsa.sglang.index.read import trim_early
from lsa.sglang.settings.design import DesignSettings

# The GLM design's list - top-k 4096, window 512 and 4 sinks - and an early budget of 2048 keys
WIDTH = 4612
EARLY_WIDTH = 2624
DESIGN = {
    "top_k": 4096,
    "window_size": 512,
    "sink_size": 4,
    "top_p": 1024,
    "bucket_size": 8,
    "bucket_cap": 16,
}


def lists_of(counts: list[int], width: int = WIDTH) -> torch.Tensor:
    """Build lists [len(counts), width] int32 with the first `count` slots real and `-1` after."""
    lists = torch.full((len(counts), width), Sentinel.PAD, dtype=torch.int32)
    for i, count in enumerate(counts):
        lists[i, :count] = torch.arange(count, dtype=torch.int32)
    return lists


def original(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, return_lse=False):
    """Stand in for `_forward_tilelang`, answering the width it was handed."""
    return page_table_1


def hooked_width(lists: torch.Tensor) -> int:
    """Return the width the read is handed through the hook."""
    handed = plugin.tilelang(original, None, None, None, 512, lists, 1.0, early_width=EARLY_WIDTH)
    return handed.shape[1]


def test_lists_that_are_all_early_are_cut_to_the_early_width():
    assert hooked_width(lists_of([2564, 100, 1, 2564])) == EARLY_WIDTH


def test_a_cut_keeps_every_entry_of_every_row():
    lists = lists_of([2564, 100, 7])
    assert torch.equal(trim_early(lists, EARLY_WIDTH), lists[:, :EARLY_WIDTH])


def test_a_chunk_with_one_late_row_keeps_every_column():
    assert hooked_width(lists_of([2564, 100, WIDTH, 2564])) == WIDTH


def test_lists_that_are_all_late_keep_every_column():
    assert hooked_width(lists_of([WIDTH, 4000, 3000])) == WIDTH


def test_a_row_exactly_as_long_as_the_early_width_is_cut():
    assert hooked_width(lists_of([EARLY_WIDTH, 5])) == EARLY_WIDTH


def test_a_row_one_past_the_early_width_keeps_every_column():
    assert hooked_width(lists_of([EARLY_WIDTH + 1, 5])) == WIDTH


def test_a_real_entry_past_a_gap_is_not_cut():
    # Entries are meant to come first, but the cut does not depend on it
    lists = lists_of([10, 10])
    lists[1, 3000] = 7
    assert hooked_width(lists) == WIDTH


def test_lists_no_wider_than_the_early_width_are_passed_on_as_they_are():
    lists = lists_of([5], width=EARLY_WIDTH)
    assert trim_early(lists, EARLY_WIDTH) is lists


def test_the_early_width_is_the_early_keys_window_and_sinks_in_blocks_of_64():
    assert (
        DesignSettings.from_blocks(
            {**DESIGN, "early_budget": {"position": 1_048_576, "top_k": 2048, "top_p": 512}}, {}
        ).early_width
        == EARLY_WIDTH
    )
    assert (
        DesignSettings.from_blocks(
            {**DESIGN, "early_budget": {"position": 1, "top_k": 2043, "top_p": 512}}, {}
        ).early_width
        == 2560
    )


def test_a_design_without_an_early_budget_has_no_early_width():
    assert DesignSettings.from_blocks(DESIGN, {}).early_width is None


@pytest.mark.parametrize("counts", [[1], [2564, 2564], [WIDTH], [0]])
def test_the_read_is_handed_the_other_arguments_untouched(counts):
    lists = lists_of(counts)
    sentinel = object()
    seen = []

    def recording(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, return_lse=False):
        seen.append((q_all, kv_cache, v_head_dim, sm_scale, return_lse))
        return sentinel

    result = plugin.tilelang(
        recording, None, "q", "kv", 512, lists, 0.5, early_width=EARLY_WIDTH, return_lse=True
    )
    assert result is sentinel
    assert seen == [("q", "kv", 512, 0.5, True)]
