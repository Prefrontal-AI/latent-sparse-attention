"""Opt-in timing ranges the indexer and the store hand to whoever is measuring them.

Nothing is recording by default, so `region` is a `nullcontext` and the serving path pays an attribute read.
[lsa.evaluation.performance.kernels.measure.recorder](../../evaluation/performance/kernels/measure/recorder.py) is what sets `recorder`, for the length of one benchmark.

The hook lives here rather than in `lsa.evaluation.performance.kernels.measure` because it belongs to the code being measured: the indexer
imports it, and the measurer plugs into it. That direction is what keeps `lsa.sglang` free of any
dependency on the harness.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

# Set by the attention-layer harness's `instrument` while a benchmark runs, and None every other moment
recorder: Any = None

_disabled: AbstractContextManager[None] = nullcontext()


def region(name: str) -> AbstractContextManager[None]:
    """Return the context manager that times `name`, or one that does nothing.

    Args:
        name: The component this range belongs to, as `lsa.evaluation.performance.kernels.measure.recorder.COMPONENTS` spells it.

    Returns:
        The recorder's range when one is recording, and a `nullcontext` otherwise.
    """
    if recorder is None:
        return _disabled
    return recorder.region(name)
