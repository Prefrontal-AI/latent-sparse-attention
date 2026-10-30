"""Shared fixtures for the native training tests."""

import gc
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist


# A trainer built in one test leaves its process group and cached device memory behind
# => tear both down after every test so the next one starts clean
@pytest.fixture(autouse=True)
def fresh_process_group() -> Iterator[None]:
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
