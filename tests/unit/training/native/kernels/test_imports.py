import importlib

import pytest

MODULES = [
    "cuda_sparse",
    "dsa_sparse",
    "fa4",
    "fa3_index",
    "gather_sparse",
    "lsa_rank",
    "lsa_select",
    "reference",
]


@pytest.mark.parametrize("name", MODULES)
def test_a_kernel_module_imports_without_a_gpu_or_a_build(name):
    # CI is CPU-only => nothing may compile or touch a device at import
    importlib.import_module(f"lsa.training.native.kernels.{name}")
