"""Decode CUDA graphs keyed by the longest context in the batch, so the index kernels are sized to it.

A flat decode scores every bucket row a request can hold, and a captured graph has static shapes - so the score
matrix and the scan's grid followed the server's `--context-length`, however short the request. A graph per
context range sizes them to the range instead, and a replay picks the graph that covers the batch's longest
request, as sglang's batch-size graphs pick the size.

sglang's decode runner already keys a graph by an attention variant, a label the runner reads from an object
with `capture_labels` and `select`. [plugin](plugin.py) hands the runner this one, and the selector reads the
label being captured to size its buffers. What a graph computes is unchanged - a range only trims the columns
that no request of it can fill.
"""

import sys
from dataclasses import dataclass

from lsa.sglang.settings.design import PAGE_SIZE

# A label is the prefix and the range's longest context, or `FULL` for the server's own
LABEL_PREFIX: str = "context_"
FULL: str = f"{LABEL_PREFIX}full"

# Where sglang keeps the label of the graph being captured
CAPTURE_MODE: str = "sglang.srt.model_executor.runner_utils.capture_mode"


@dataclass(frozen=True)
class ContextBuckets:
    """The decode graphs of a server, one a context range, in sglang's attention-variant protocol."""

    # Longest context of each range below the server's, ascending
    limits: tuple[int, ...]

    @property
    def capture_labels(self) -> tuple[str, ...]:
        """Return the graphs to capture, the widest first so the narrower ones reuse its memory."""
        return (FULL, *(label_of(limit) for limit in reversed(self.limits)))

    def select(self, forward_batch: object) -> str:
        """Return the graph that covers the batch's longest request.

        Args:
            forward_batch: The batch, whose `seq_lens_cpu` is the host mirror decode keeps without a sync.

        Returns:
            The label of the narrowest range the longest request fits in, or `FULL` when none does or the
            lengths are not on the host.
        """
        lengths = getattr(forward_batch, "seq_lens_cpu", None)
        if lengths is None or lengths.numel() == 0:
            return FULL
        longest = int(lengths.max())
        for limit in self.limits:
            if longest <= limit:
                return label_of(limit)
        return FULL

    @classmethod
    def under(cls, limits: tuple[int, ...], context_length: int) -> "ContextBuckets | None":
        """Return the ranges that are narrower than the server's context.

        Args:
            limits: The design's ranges.
            context_length: The longest context the server holds.

        Returns:
            The buckets, or None when no range is narrower than the context and one graph is all there is.
        """
        narrower = tuple(limit for limit in limits if limit < context_length)
        return cls(narrower) if narrower else None


def label_of(limit: int) -> str:
    """Return the label of the graph for contexts up to `limit`."""
    return f"{LABEL_PREFIX}{limit}"


def captured_limit() -> int | None:
    """Return the longest context of the graph being captured, or None outside a range's capture.

    Returns:
        The limit, which is None in an eager step, a replay, the full graph's capture, and on an image whose
        runner keys no graph by attention variant (the nightly has the module but not the getter).
    """
    getter = getattr(sys.modules.get(CAPTURE_MODE), "get_capture_attention_variant", None)
    label = None if getter is None else getter()
    if label is None or label == FULL or not label.startswith(LABEL_PREFIX):
        return None
    return int(label.removeprefix(LABEL_PREFIX))


def pages_of(limit: int) -> int:
    """Return the pages of cache rows a request of `limit` keys can fill."""
    return -(-limit // PAGE_SIZE)
