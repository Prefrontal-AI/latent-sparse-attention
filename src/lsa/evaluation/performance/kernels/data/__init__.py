"""What a kernel measurement loads - the truncated real-weight checkpoint, built once from a model's snapshot.

| module | what it does |
|---|---|
| [cut](cut.py) | build the truncated real-weight checkpoint the harness and the module timer load - `lsa kernels cut` |

The cut is a script ported with [measure](../measure/__init__.py) and shares its exemption from annotation -
see the `lsa.evaluation.performance.kernels.measure.*` note in `pyproject.toml`.
"""
