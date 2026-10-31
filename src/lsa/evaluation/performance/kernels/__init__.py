"""Kernel performance of one attention layer - what the harness measured, as one record per run.

Everything here but `measure/` and `data/` runs outside the container: it builds the container command, runs it
and writes the run directory. `measure/` is what runs inside the serving container, and `data/` builds the
truncated checkpoint it loads.

| module | what it does |
|---|---|
| [harness](harness.py) | run the attention-layer harness in the container, one process per context, and rewrite the record after each |
| [modules](modules.py) | run the module timer in the container, as a one-GPU job - `lsa kernels modules` |
| [isolated](isolated.py) | time every arm of one model's layer, both phases, in one job on a node held whole - `lsa evaluate isolated` |
| [record](record.py) | build a run's record from the harness's raw files |
| [compose](compose.py) | sum one layer of each kind and the modules between them into a whole model - `lsa kernels compose` |
| [show](show.py) | a record as a table - what `lsa records show` prints |
| [rebuild](rebuild.py) | rebuild a run's record from its samples - `lsa records rebuild` |
| [data](data/__init__.py) | the truncated real-weight checkpoint the harness and the module timer load - `lsa kernels cut` |
| [measure](measure/__init__.py) | the harness and the module timer, run inside the container |
"""
