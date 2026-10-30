"""Per-component GPU time inside one attention layer, as nested ranges over a single stream.

A range is opened by [lsa.sglang.index.probe.region](../../../../sglang/index/probe.py) from inside the indexer and the
store, and by the wrappers [instrument](instrument.py) installs around everything else. `read` turns the nested inclusive times into exclusive ones and charges
whatever is left to `other`.
"""

from collections import defaultdict
from contextlib import contextmanager

import torch

COMPONENTS = (
    "index qk projections",
    "index q prep",
    "index k prep",
    "index q scan",
    "index k cluster",
    "index k select",
    "attention projections",
    "topk attention",
)
# The ranges every instrumented trial of a variant must have seen - an unindexed layer opens no index range
REQUIRED = {
    "sparse": ("attention projections", "index q scan", "index k select", "topk attention"),
    # PISA scores its pyramid and keeps in one search, so it has no query scan range apart from its select
    "pisa": ("attention projections", "index k select", "topk attention"),
    "full": ("attention projections", "topk attention"),
    "linear": ("attention projections", "topk attention"),
}


class Recorder:
    """Record GPU time as nested ranges on one stream, then split it per component."""

    def __init__(self, capacity=8192) -> None:
        """Allocate and warm the event pairs.

        Args:
            capacity: Ranges one trial may open.
        """
        # QSA tiles a prefill's scores by rows, three ranges a tile
        # => about a thousand tiles at 10M for a 32k chunk
        self.events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(capacity)
        ]
        for pair in self.events:
            for event in pair:
                event.record()
        torch.cuda.synchronize()
        self.reset()

    def reset(self) -> None:
        """Forget the ranges recorded so far and bind the current stream."""
        self.ranges: list[tuple[str, int | None]] = []
        self.stack: list[int] = []
        self.stream = torch.cuda.current_stream().cuda_stream
        self.enabled = True
        self.backends: defaultdict[str, int] = defaultdict(int)

    @contextmanager
    def region(self, name):
        """Time the block as a range nested under the open one.

        Args:
            name: The component to charge.

        Raises:
            RuntimeError: The stream changed, the capacity ran out or ranges closed out of order.
        """
        if not self.enabled:
            yield
            return
        if torch.cuda.current_stream().cuda_stream != self.stream:
            raise RuntimeError("Component event accounting supports one stream only")
        index = len(self.ranges)
        if index >= len(self.events):
            raise RuntimeError("Component event capacity exceeded")
        parent = self.stack[-1] if self.stack else None
        self.ranges.append((name, parent))
        self.stack.append(index)
        start, end = self.events[index]
        torch.cuda.nvtx.range_push(name)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.nvtx.range_pop()
            if self.stack.pop() != index:
                raise RuntimeError("Component ranges closed out of order")

    def read(self, module_ms, first=0, last=None):
        """Turn the recorded ranges into exclusive time per component.

        Args:
            module_ms: The whole module's time, of which what no range took goes to `other`.
            first: First range of the trial.
            last: End of the trial's ranges, or None for all.

        Returns:
            `exclusive_ms` per component and the `ranges` with their parents and inclusive times.

        Raises:
            RuntimeError: A range is open, crosses the boundary or the times do not add up.
        """
        if self.stack:
            raise RuntimeError("Unclosed component range")
        last = len(self.ranges) if last is None else last
        ranges = self.ranges[first:last]
        inclusive = [a.elapsed_time(b) for a, b in self.events[first:last]]
        exclusive = list(inclusive)
        for index, (_, parent) in enumerate(ranges):
            if parent is not None:
                if parent < first:
                    raise RuntimeError("Component crosses the timing boundary")
                exclusive[parent - first] -= inclusive[index]
        times = defaultdict(float)
        for (name, _), elapsed in zip(ranges, exclusive, strict=True):
            if elapsed < -0.005:
                raise RuntimeError(f"Negative exclusive range: {name}: {elapsed}")
            times[name] += elapsed
        components = {name: times[name] for name in COMPONENTS}
        components["other"] = module_ms - sum(components.values())
        if components["other"] < -0.005:
            raise RuntimeError("Component total exceeds module time")
        return {
            "exclusive_ms": components,
            "ranges": [
                {"name": name, "parent": parent, "inclusive_ms": elapsed}
                for (name, parent), elapsed in zip(ranges, inclusive, strict=True)
            ],
        }
