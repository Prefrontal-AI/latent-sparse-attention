"""Where each kernel of a read sits on its GPU's roofline - an H200's or a B300's - from Nsight Compute counters.

One one-GPU job profiles every kernel the dispatch table can send to CuTeDSL, one call each, on inputs shaped as a
served chunk sees them, and writes the counters plus the roofline's machine peaks as one JSON a page plots.

| module | what it does |
|---|---|
| [workloads](workloads.py) | build each step's inputs at a context and the call that launches it once |
| [profile](profile.py) | time each step, then launch it once inside a profiler window - the process Nsight Compute wraps |
| [anchors](anchors.py) | measure a GPU's own copy bandwidth and cuBLAS GEMM rates, the roofline's anchors |
| [report](report.py) | read the profile's report into one record per kernel, with its FLOPs, bytes and intensities |
| [run](run.py) | submit the profile as a one-GPU job and write the roofline JSON - `lsa kernels roofline` |
"""
