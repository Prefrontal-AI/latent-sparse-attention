from types import SimpleNamespace

import torch

from lsa.sglang.index import registry


def batch(extend: bool, seq_lens: list[int], extend_lens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: extend),
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens_cpu=extend_lens,
    )


def test_a_prefill_ends_before_its_new_tokens():
    assert registry.ends_of(batch(True, [10, 7], [4, 7]), 2) == [6, 0]


def test_a_decode_ends_one_token_back():
    assert registry.ends_of(batch(False, [10, 7], []), 2) == [9, 6]


def test_a_decode_padded_for_a_cuda_graph_ends_only_its_real_requests():
    assert registry.ends_of(batch(False, [10, 7, 1, 1], []), 2) == [9, 6]
