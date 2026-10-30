"""The kernels the native sparse arms read and select with, built on first use.

Nothing here compiles at import: a Triton kernel compiles when it first launches and a CUDA source builds through
`torch.utils.cpp_extension.load_inline` when its module is first imported, which `cuda_sparse` and `fa3_index` do on the
first call, so a CPU-only environment imports the package and runs `reference`.
"""
