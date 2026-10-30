"""What a selection's lengths are on the way to a flash-attention read.

Padding arithmetic is the one part of a sparse read that fails without raising - a count one too high makes
every short row attend to slot 0 as well, which is a real key belonging to some other request, and the output
is wrong by a little on every token rather than wrong loudly on one.
"""

import torch

from lsa.constants import Sentinel
from lsa.sglang.index.read import list_lengths


def a_selection(rows):
    """Pad ragged rows of slots into the `[tokens, budget]` int32 a selection is."""
    budget = max(len(row) for row in rows)
    padded = [row + [Sentinel.PAD] * (budget - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.int32)


def test_a_row_counts_the_slots_it_selected():
    rows = [[5, 9, 12, 40], [7], [0, 1, 2], [31, 30, 29, 28]]
    assert list_lengths(a_selection(rows)).tolist() == [4, 1, 3, 4]


def test_slot_zero_is_selectable_and_not_confused_with_padding():
    assert list_lengths(a_selection([[0, 3], [3]])).tolist() == [2, 1]


def test_an_empty_row_reads_one_slot_rather_than_none():
    empty = torch.full((2, 3), Sentinel.PAD, dtype=torch.int32)
    assert list_lengths(empty).tolist() == [1, 1]
