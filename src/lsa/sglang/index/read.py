"""The count of real entries in a selection, for a flash-attention read over arbitrary key slots.

A selection is `[tokens, budget]` int32 KV slots with the valid entries first and `-1` padding, and the read
is `flash_attn_with_kvcache` at `page_size` 1, where a page table row is a list of slots rather than a run of
pages. sglang states the trick that makes a *per-query* list work in its own metadata rather than in a
kernel: `DsaMetadata.dsa_cu_seqlens_q` carries the comment `must be arange(0, len(dsa_cu_seqlens_k))` and
`dsa_max_seqlen_q` is typed `Literal[1]`, so every query token is its own sequence of length one with its own
page table row and its own valid count.

The kernel reads `cache_seqlens[i]` entries of row `i` and does not look at the padding, so the count has to
be exact: one too many and a query attends to whatever `-1` clamped to, which is slot 0 - a real key, in
another request's context, added to every row that was short. One too few and the read silently drops keys
the budget paid for. The checkpoint's own indexer never had this problem because its list is full once the
context passes `index_topk`; an LSA list is as long as the selection is.

`list_lengths` is the one count of a list's real entries - the GLM backend's hooks in [plugin](../plugin.py) and
the kernel study's page tables read it.

The TileLang read has no count to give. It walks a list in blocks of 64 columns and skips only the `-1` entries, so
it pays for the list's whole width however few entries are real. `trim_early` cuts a list down to the columns a
batch fills when they all fit under the width an early budget needs, which is what lets that budget save the read.
"""

import torch

from lsa.constants import Sentinel


def list_lengths(selection: torch.Tensor) -> torch.Tensor:
    """Count the real entries of each selection row.

    A row with nothing in it is reported as 1 rather than 0, because a zero-length read is a softmax over
    an empty set and comes back NaN. LSA always selects the sinks and the local window, so an empty row does
    not arise from a selection - only from a bug or an idle row, and one wrong key beats a NaN that
    propagates through the residual stream.

    Args:
        selection: Selected KV slots [tokens, budget] int32, `Sentinel.PAD` padded.

    Returns:
        How many entries of each row are real [tokens] int32, at least 1.
    """
    return (selection > Sentinel.PAD).sum(1, dtype=torch.int32).clamp_(min=1)


def trim_early(selection: torch.Tensor, early_width: int) -> torch.Tensor:
    """Cut a selection to `early_width` columns when no row has a real entry past them.

    The cut drops only `-1` columns, so the read of the cut list is the read of the whole one. It is decided on the
    columns that hold an entry rather than on a row's count, so it holds whether or not the entries come first.
    A chunk with one late row keeps every column, since that row fills them.
    Costs one host read, which a CUDA graph capture cannot make - the caller passes the list on untouched there.

    Args:
        selection: Selected KV slots [tokens, budget] int32, `Sentinel.PAD` padded.
        early_width: Columns an early row needs, a multiple of 64.

    Returns:
        The first `early_width` columns [tokens, early_width] int32, contiguous, or the selection itself when it is
        no wider or a row reaches past them.
    """
    if selection.shape[1] <= early_width:
        return selection
    columns = torch.arange(1, selection.shape[1] + 1, device=selection.device, dtype=torch.int32)
    filled = ((selection > Sentinel.PAD) * columns).max()
    if int(filled) > early_width:
        return selection
    return selection[:, :early_width].contiguous()
