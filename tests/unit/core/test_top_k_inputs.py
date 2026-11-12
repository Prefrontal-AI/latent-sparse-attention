"""The checks `top_keys` makes of its inputs, which need no GPU."""

import pytest
import torch

from lsa.core.kernels.triton.top_k import top_keys


def test_scores_that_are_not_fp32_are_refused():
    scores = torch.zeros(2, 8, dtype=torch.bfloat16)
    lengths = torch.full((2,), 8, dtype=torch.int32)
    with pytest.raises(TypeError, match=r"fp32 scores, not torch\.bfloat16"):
        top_keys(scores, lengths, 4)
