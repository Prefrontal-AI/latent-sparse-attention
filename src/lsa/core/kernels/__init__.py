"""The device kernels every backend draws from, one subpackage a kernel language.

[core](../__init__.py) is the specification and runs on a CPU. These are the same arithmetic written for a
GPU, and they live here rather than beside one caller because the serving path and the training path want
the same ones: a scan is a scan whether the cache is paged or a training batch.

[triton](triton/__init__.py) holds the kernels where a fused launch beats a chain of small ones, and the host code
that orders them. [cutedsl](cutedsl/__init__.py) holds the kernels that need what Triton does not expose, such as
shared memory a kernel stages and compacts itself.

The Triton modules are pure Triton. A step that has both has a front door beside this file - [ward](ward.py),
[expansion](expansion.py) and [scan_select](scan_select.py), named after the Triton module they front - and a front
door is the one place that step's language is chosen, by [dispatch](dispatch.py) and the CuTeDSL kernel's own `supports`.

Nothing is imported at package level, so `import lsa` never reaches a kernel language. A caller names the
module it wants.

[dispatch](dispatch.py) is the table of which language runs each step on which GPU.
"""
