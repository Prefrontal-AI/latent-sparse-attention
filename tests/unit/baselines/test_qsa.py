"""QSA's indexer and read against naive loops - the pooling, the norm, the rotation, the score and the selection."""

import math

import pytest
import torch

from lsa.baselines import qsa
from lsa.sglang.settings.design import QSAConfig


def rotate_half(rows):
    first, second = rows.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def test_pooling_is_the_mean_of_each_closed_block_and_drops_the_open_one():
    x = torch.randn(10, 3)
    pooled = qsa.pool_blocks(x, 4)
    assert pooled.shape == (2, 3)
    assert torch.allclose(pooled[0], x[0:4].mean(0))
    assert torch.allclose(pooled[1], x[4:8].mean(0))


def test_the_gemma_norm_scales_unit_rms_rows_by_one_plus_the_weight():
    rows = torch.randn(5, 8)
    weight = torch.randn(8)
    expected = rows / rows.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * (1 + weight)
    assert torch.allclose(qsa.gemma_norm(rows, weight, 1e-6), expected, atol=1e-6)


def test_rotation_is_the_models_rotate_half_on_the_leading_dims_with_the_tables_scaled():
    torch.manual_seed(0)
    rows = torch.randn(6, 3, 16)
    positions = torch.tensor([0, 1, 5, 100, 4096, 1_000_000])
    inverse_frequencies = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    scaling = 1.37
    angles = positions.float()[:, None] * inverse_frequencies[None]
    # The HF tables - each frequency twice, scaled - over the first 8 of 16 dims
    cos = torch.cat((angles, angles), -1).cos()[:, None] * scaling
    sin = torch.cat((angles, angles), -1).sin()[:, None] * scaling
    turned = rows[..., :8] * cos + rotate_half(rows[..., :8]) * sin
    expected = torch.cat((turned, rows[..., 8:]), -1)
    assert torch.allclose(qsa.rotate(rows, positions, inverse_frequencies, scaling), expected, atol=1e-5)


def test_the_block_score_sums_each_heads_rectified_dot_over_root_head_dim():
    index_queries = torch.randn(3, 4, 8)
    index_keys = torch.randn(5, 8)
    scores = qsa.block_scores(index_queries, index_keys)
    for row in range(3):
        for block in range(5):
            expected = sum(
                max(float(index_queries[row, head] @ index_keys[block]), 0.0) for head in range(4)
            ) / math.sqrt(8)
            assert float(scores[row, block]) == pytest.approx(expected, abs=1e-5)


def naive_select(scores, position, block_size, top_blocks):
    closed = (position + 1) // block_size
    ranked = sorted(range(closed), key=lambda block: -float(scores[block]))[:top_blocks]
    keys = {block * block_size + offset for block in ranked for offset in range(block_size)}
    return keys | set(range(closed * block_size, position + 1))


@pytest.mark.parametrize("block_size,top_blocks", [(4, 3), (4, 64), (2, 5)])
def test_the_read_is_the_best_closed_blocks_as_keys_and_the_open_block(block_size, top_blocks):
    torch.manual_seed(1)
    length = 61
    scores = torch.randn(length, length // block_size)
    positions = torch.arange(length)
    read = qsa.select(scores, positions, block_size, top_blocks)
    assert read.shape == (length, top_blocks * block_size + block_size - 1)
    for row, position in enumerate(positions.tolist()):
        got = [key for key in read[row].tolist() if key >= 0]
        assert len(got) == len(set(got))
        assert set(got) == naive_select(scores[row], position, block_size, top_blocks)


def test_a_query_never_selects_a_block_it_has_not_closed():
    scores = torch.tensor([[1.0, 2.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0]])
    # Position 9 has closed blocks 0 and 1, and reads 8 and 9 of its open block 2
    read = qsa.select(scores, torch.tensor([9]), 4, 1)
    assert set(read[0].tolist()) - {-1} == {4, 5, 6, 7, 8, 9}


def test_the_published_shape_reads_2048_keys_in_512_blocks():
    config = QSAConfig()
    assert (config.query_head_count, config.head_dim, config.rotary_dim) == (4, 128, 64)
    assert (config.block_size, config.top_blocks, config.top_k) == (4, 512, 2048)


@pytest.mark.parametrize(
    "overrides",
    [{"block_size": 1}, {"rotary_dim": 65}, {"rotary_dim": 256}, {"top_k": 2047}, {"input": "latent"}],
)
def test_an_indexer_the_read_cannot_run_is_refused(overrides):
    with pytest.raises(ValueError):
        QSAConfig(**overrides)


def test_a_document_shorter_than_a_block_pools_to_no_block():
    assert qsa.pool_blocks(torch.randn(3, 5), 4).shape == (0, 5)
