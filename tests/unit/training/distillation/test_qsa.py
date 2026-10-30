"""The QSA indexer as the trainer reads it - its target, its loss, its keys and its read, against naive references."""

import math
from pathlib import Path

import pytest
import torch

from lsa.baselines import qsa
from lsa.sglang.settings.design import QSAConfig
from lsa.training.distillation.config import DistillConfig, ModelConfig, ObjectiveConfig, RunConfig
from lsa.training.distillation.constants import Cache, Term
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.data.teacher_attention import teacher_attention
from lsa.training.distillation.losses import block_divergence, calibrate, qsa_block_scores, step_terms
from lsa.training.distillation.metrics import qsa_read
from lsa.training.distillation.model import QSAIndexer
from tests.helpers.training.distillation.data import fake_cache

LENGTH, MODEL_DIM, BLOCK = 256, 16, 4
QUERY_HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
CONFIG = QSAConfig(query_head_count=2, head_dim=8, rotary_dim=4, block_size=BLOCK, top_k=32)


def frequencies():
    return 1.0 / (10000 ** (torch.arange(0, CONFIG.rotary_dim, 2).float() / CONFIG.rotary_dim))


def teacher(seed=0, top=64):
    """A document's x, its sampled positions, the model's q and k at them, and QSA's targets off the teacher."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(LENGTH, MODEL_DIM, generator=generator)
    first = fake_cache.SINK + fake_cache.WINDOW
    training = (first + torch.randperm(LENGTH - 8 - first, generator=generator)[:16]).sort().values
    positions = torch.cat([training, torch.arange(LENGTH - 8, LENGTH)])
    q = torch.randn(QUERY_HEADS, len(positions), HEAD_DIM, generator=generator)
    k = torch.randn(KV_HEADS, LENGTH, HEAD_DIM, generator=generator)
    targets = teacher_attention(q, k, 1.0, positions, fake_cache.config(), top, block_size=BLOCK)
    return x, positions, q, k, targets


def document(seed=0, top=64):
    x, positions, _, _, (top_key_ids, top_key_mass, peak_key_id) = teacher(seed, top)
    blob = {
        Cache.QSA_FIELDS[0]: qsa.pool_blocks(x, BLOCK).bfloat16(),
        Cache.QSA_FIELDS[1]: x[positions].bfloat16(),
        "query_positions": positions.int(),
        "top_key_ids": top_key_ids,
        "top_key_mass": top_key_mass,
        "peak_key_id": peak_key_id,
        Cache.META: {
            "document": "source0",
            "layer": 0,
            Cache.SINK_SIZE: fake_cache.SINK,
            Cache.WINDOW_SIZE: fake_cache.WINDOW,
            Cache.CLUSTER_BLOCK_SIZE: fake_cache.WINDOW,
            Cache.LENGTH: LENGTH,
            Cache.TRAIN_QUERIES: 16,
            Cache.BLOCK_SIZE: BLOCK,
        },
    }
    return Document.from_blob(Path("tokens/train/source0_layer0.pt"), blob, fake_cache.config())


def full_attention(q, k, positions):
    """Every head's causal softmax, summed over heads [Q, T]."""
    groups = QUERY_HEADS // KV_HEADS
    logits = torch.einsum("hqd,htd->hqt", q, k.repeat_interleave(groups, 0))
    causal = torch.arange(LENGTH)[None] <= positions[:, None]
    return torch.softmax(logits.masked_fill(~causal[None], float("-inf")), -1).sum(0)


def naive_block_target(q, k, positions):
    """QSA's target - the full attention, max-pooled over closed blocks, renormalised."""
    attention = full_attention(q, k, positions)
    target = torch.zeros(len(positions), LENGTH // BLOCK)
    for row, position in enumerate(positions.tolist()):
        for block in range((position + 1) // BLOCK):
            target[row, block] = attention[row, block * BLOCK : (block + 1) * BLOCK].max()
    return target / target.sum(-1, keepdim=True)


def test_the_teacher_target_is_the_full_attention_on_closed_blocks():
    _, positions, q, k, (ids, mass, peaks) = teacher(top=LENGTH)
    attention = full_attention(q, k, positions)
    for row, position in enumerate(positions.tolist()):
        closed = (position + 1) // BLOCK * BLOCK
        kept = {int(key): float(value) for key, value in zip(ids[row], mass[row], strict=True) if value > 0}
        assert set(kept) == set(range(closed))
        assert kept[0] == pytest.approx(float(attention[row, 0]), rel=1e-2)
        assert all(int(peak) < closed for peak in peaks[row])


def test_the_block_target_is_the_max_pooled_attention_when_every_key_is_kept():
    _, positions, q, k, _ = teacher(top=LENGTH)
    one = document(top=LENGTH)
    rows = torch.arange(len(positions))
    assert torch.allclose(one.block_target(rows), naive_block_target(q, k, positions), atol=2e-3)


def test_a_cut_target_drops_whole_blocks_and_keeps_the_others_in_proportion():
    _, positions, q, k, _ = teacher(top=24)
    one = document(top=24)
    rows = torch.arange(len(positions))
    cut = one.block_target(rows)
    full = naive_block_target(q, k, positions)
    kept = cut > 0
    # Every kept block's share is the full target's share of what is kept
    renormalised = (full * kept) / (full * kept).sum(-1, keepdim=True)
    assert torch.allclose(cut, renormalised, atol=2e-3)
    assert int(kept.sum(-1).max()) <= 24


def test_block_keys_of_pooled_x_are_the_pooled_projected_keys_as_the_server_builds_them():
    torch.manual_seed(0)
    indexer = QSAIndexer(MODEL_DIM, CONFIG, frequencies(), 1.2)
    with torch.no_grad():
        indexer.k_layernorm.normal_()
    x = torch.randn(LENGTH, MODEL_DIM)
    keys = qsa.pool_blocks(x, BLOCK)
    got = indexer.index_keys(keys)
    # The server projects every token, means a block's raw keys, then norms and rotates at the block's first position
    raw = x @ indexer.index_qk_proj[CONFIG.query_head_count * CONFIG.head_dim :].t()
    pooled = raw.view(-1, BLOCK, CONFIG.head_dim).mean(1)
    normed = qsa.gemma_norm(pooled, indexer.k_layernorm, indexer.eps)
    expected = qsa.rotate(normed, torch.arange(len(pooled)) * BLOCK, frequencies(), 1.2)
    assert torch.allclose(got, expected, atol=1e-4)


def naive_divergence(scores, target, selectable):
    loss = 0.0
    for row in range(len(scores)):
        allowed = selectable[row]
        log_predicted = torch.log_softmax(scores[row][allowed], -1)
        loss -= float((target[row][allowed] * log_predicted).sum())
    return loss / len(scores)


def test_the_block_divergence_is_the_cross_entropy_over_closed_blocks():
    one = document()
    rows = one.train_rows()
    scores = torch.randn(len(rows), LENGTH // BLOCK)
    got = float(block_divergence(scores, one, rows))
    assert got == pytest.approx(
        naive_divergence(scores, one.block_target(rows), one.visible_blocks(rows)), rel=1e-5
    )


def test_training_the_indexer_lowers_its_loss_and_moves_every_weight():
    torch.manual_seed(0)
    one = document()
    rows = one.train_rows()
    indexer = QSAIndexer(MODEL_DIM, CONFIG, frequencies(), 1.0)
    calibrate(indexer, [one])
    objective = ObjectiveConfig()
    before = float(step_terms(indexer, one, rows, objective)[Term.KEY].detach())
    optimiser = torch.optim.Adam(indexer.parameters(), lr=3e-2)
    for _ in range(60):
        optimiser.zero_grad()
        step_terms(indexer, one, rows, objective)[Term.KEY].backward()
        optimiser.step()
    assert float(step_terms(indexer, one, rows, objective)[Term.KEY].detach()) < before
    for name in ("index_qk_proj", "q_layernorm", "k_layernorm"):
        assert float(getattr(indexer, name).grad.abs().sum()) > 0


@torch.no_grad()
def test_the_qsa_read_holds_the_open_block_and_its_budget_of_blocks():
    one = document()
    rows = one.eval_rows()
    indexer = QSAIndexer(MODEL_DIM, CONFIG, frequencies(), 1.0)
    index_queries = indexer.index_queries(one.query_inputs(rows), one.query_positions[rows])
    metrics = qsa_read(one, index_queries, indexer.index_keys(one.key_inputs()), CONFIG, rows, chunk_size=3)
    assert 0.0 < metrics["mass"] <= 1.0
    assert metrics["ids_read"] <= CONFIG.top_k + BLOCK - 1
    scores = qsa_block_scores(indexer, one, rows)
    assert scores.shape == (len(rows), LENGTH // BLOCK)


def test_a_qsa_run_is_refused_without_the_online_teacher():
    with pytest.raises(ValueError, match="data.teacher"):
        DistillConfig(model=ModelConfig(qsa=QSAConfig()), run=RunConfig(output="out"))


def test_the_published_shape_scores_at_its_scale():
    # Two unit-rms heads that agree everywhere score head_count * head_dim / sqrt(head_dim)
    index_queries = torch.ones(1, 4, 128)
    index_keys = torch.ones(1, 128)
    assert float(qsa.block_scores(index_queries, index_keys)) == pytest.approx(4 * 128 / math.sqrt(128))
