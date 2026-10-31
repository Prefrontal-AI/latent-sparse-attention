"""What runs inside the serving container to time one layer - the harness and the module timer.

[The orchestration one level up](../harness.py) builds the
container command, and every module here runs against sglang on a GPU. The truncated checkpoint both of them
load is [the cut](../data/cut.py), the kernels' data.

| module | what it does |
|---|---|
| [layer](layer.py) | time and validate one attention layer - `python -m lsa.evaluation.performance.kernels.measure.layer` |
| [modules](modules.py) | time the modules a layer runs besides its attention - `python -m ...measure.modules` |
| [variant](variant.py) | the layer under test - sparse, full or linear - and what belongs to it |
| [instrument](instrument.py) | wrap the layer so the recorder can charge time to a component |
| [graph](graph.py) | capture the layer as a CUDA graph and compare it against the eager run |
| [decode](decode.py) | the decode counterpart - one token against a filled cache |
| [score_check](score_check.py) | an exact reference for the index scores |
| [recorder](recorder.py) | per-component GPU time as nested ranges, opened through [lsa.sglang.index.probe](../../../../sglang/index/probe.py) |
| [selection](selection.py) | whether a selection is the true top-k |
| [snapshot](snapshot.py) | save what one forward pass mutates and put it back |
| [workload](workload.py) | whether a measurement fits in memory, and which query rows to score |

The last four are the half with logic a CPU can exercise, and the tests import them. The arithmetic stays as written - see the `lsa.evaluation.performance.kernels.measure.*` note in
`pyproject.toml` for why none of this is annotated.
"""
