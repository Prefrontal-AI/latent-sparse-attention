import logging

import pytest

from fusion_lsa.budget import EarlyBudget, budget_at

logger = logging.getLogger(__name__)


def test_budget_at_no_early_budget():
    assert budget_at(0, 4096, 64, None) == (4096, 64)
    assert budget_at(100000, 4096, 64, None) == (4096, 64)


def test_budget_at_early_position():
    early = EarlyBudget(position=8192, top_k=1024, top_p=16)
    assert budget_at(0, 4096, 64, early) == (1024, 16)
    assert budget_at(8191, 4096, 64, early) == (1024, 16)
    assert budget_at(8192, 4096, 64, early) == (4096, 64)


def test_early_budget_rejects_non_positive():
    with pytest.raises(ValueError, match="at least 1"):
        EarlyBudget(position=0, top_k=1024, top_p=16)
    with pytest.raises(ValueError, match="at least 1"):
        EarlyBudget(position=8192, top_k=0, top_p=16)


def test_early_budget_rejects_non_integer():
    with pytest.raises(ValueError, match="must be an integer"):
        EarlyBudget(position=8192.0, top_k=1024, top_p=16)
    with pytest.raises(ValueError, match="must be an integer"):
        EarlyBudget(position=8192, top_k=True, top_p=16)


def test_early_budget_read_from_mapping():
    early = EarlyBudget.read({"position": 8192, "top_k": 1024, "top_p": 16})
    assert early == EarlyBudget(position=8192, top_k=1024, top_p=16)
    assert EarlyBudget.read(None) is None
    with pytest.raises(ValueError, match="keys"):
        EarlyBudget.read({"position": 8192, "top_k": 1024})
    with pytest.raises(ValueError, match="mapping"):
        EarlyBudget.read([1, 2, 3])


def test_budget_at_early_parity(upstream_src):
    from lsa.budget import EarlyBudget as TorchEarlyBudget
    from lsa.budget import budget_at as torch_budget_at

    early = EarlyBudget(position=2048, top_k=1024, top_p=256)
    torch_early = TorchEarlyBudget(position=2048, top_k=1024, top_p=256)
    for position in (0, 1, 511, 512, 2047, 2048, 2049, 4096, 100000):
        assert budget_at(position, 4096, 1024, early) == torch_budget_at(position, 4096, 1024, torch_early)
