"""Contracts for LSA's learned projections and auxiliary gradient paths."""

import pytest
import torch
from torchtitan.models.common.aux_loss import AuxLoss

from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.lsa_indexer import LsaInnerAttention, bucket_scores, selection_table
from tests.unit.training.native.test_lsa_flat_pin import small


@pytest.fixture(autouse=True)
def denominator():
    AuxLoss.set_step_denominator(torch.tensor(32.0))
    yield
    AuxLoss._step_denominator = None


def module(training="ste", source="key_and_query", **changes):
    config = small(
        **{
            "index_input": source,
            "index_training": training,
            "index_dim": 8,
            "block_size": 8,
            "index_detached": True,
            "index_loss_weight": 1.0,
            **changes,
        }
    )
    return LsaInnerAttention.Config(
        attention=config,
        seq_len=32,
        rotary_dim=0,
        model_dim=12,
        query_head_count=4,
        kv_head_count=1,
        head_dim=8,
        steps=100,
        decay_ratio=0.4,
    ).build()


def inputs():
    torch.manual_seed(4)
    return [torch.randn(32, heads, 8, requires_grad=True) for heads in (4, 1, 1)]


@pytest.mark.parametrize("training", ["kl", "ste"])
@pytest.mark.parametrize("source", ["key_and_query", "attention_input"])
def test_each_training_path_reaches_both_projections(training, source):
    layer = module(training, source)
    tensors = inputs()
    output = layer(*tensors, attention_input=torch.randn(32, 12))
    output.square().sum().backward()
    for projection in (layer.query_projection, layer.key_projection):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.norm() > 0
    for tensor in tensors:
        assert torch.isfinite(tensor.grad).all()


def test_unlearned_wrapper_matches_the_original_bitwise():
    config = small()
    original = MultiresInnerAttention.Config(attention=config, seq_len=32, rotary_dim=0).build()
    extended = LsaInnerAttention.Config(attention=config, seq_len=32, rotary_dim=0).build()
    tensors = inputs()
    assert torch.equal(original(*tensors), extended(*tensors))


def test_summary_anneal_uses_completed_updates_and_disables_at_half_decay():
    layer = module(summaries=True)
    for step, expected in [(0, 1), (60, 1), (70, 0.5), (80, 0), (100, 0)]:
        layer.set_step(step)
        assert layer.summary_weight == expected


@pytest.mark.parametrize("training", ["kl", "ste"])
def test_summaries_reach_unselected_keys_and_remain_finite(training):
    layer = module(training, summaries=True)
    tensors = inputs()
    output = layer(*tensors)
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    for tensor in tensors:
        assert torch.isfinite(tensor.grad).all()
    layer.set_step(80)
    no_summaries = layer(*tensors)
    layer.attention.summaries = False
    assert torch.equal(no_summaries, layer(*tensors))


def test_empty_selection_has_finite_zero_kl_with_connected_projections():
    layer = module("kl", top_k=0)
    layer(*inputs()).sum().backward()
    for projection in (layer.query_projection, layer.key_projection):
        assert torch.equal(projection.weight.grad, torch.zeros_like(projection.weight))


def test_max_head_selection_rejects_multiple_key_heads():
    layer = module("none", "none", query_head_reduction="max")
    with pytest.raises(ValueError, match="one key head"):
        layer._index_vectors(torch.randn(32, 4, 8), torch.randn(32, 2, 8), None, None)


def test_bucket_scores_rebuild_unit_centroids_without_padded_members():
    queries = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    owner, scores = bucket_scores(queries, keys, torch.tensor([[[0, 1, -1]]]))
    assert torch.equal(owner, torch.tensor([[0, 0]]))
    assert torch.allclose(scores, torch.tensor([[[2**-0.5]]]))
    scores.sum().backward()
    assert keys.grad.norm() > 0


def test_padded_indices_never_select_key_zero():
    assert not selection_table(torch.full((1, 3, 4), -1), 3).any()


def test_sparse_log_mass_backward_matches_a_dense_read():
    from lsa.training.native.lsa_gradients import LogMass
    from tests.helpers.training.kernels.inputs import dense_reference

    tensors = [x.reshape(1, 32, x.shape[1], 8) for x in inputs()]
    layer = module()
    index_queries, index_keys = layer._index_vectors(*[x.flatten(0, 1) for x in tensors[:2]], None, None)
    selected, _ = layer._selected(index_queries, index_keys)
    _, lse = dense_reference(*tensors, selected, 8, 2, 8**-0.5)
    queries, keys = (x.transpose(1, 2) for x in tensors[:2])
    attached = LogMass.apply(queries, keys, selected, lse.detach(), 8, 2, 8**-0.5)
    gradient = torch.randn_like(lse)
    got = torch.autograd.grad(attached, tensors[:2], gradient, retain_graph=True)
    expected = torch.autograd.grad(lse, tensors[:2], gradient)
    for ours, want in zip(got, expected, strict=True):
        torch.testing.assert_close(ours, want, rtol=1e-5, atol=1e-6)


def test_sparse_ste_matches_zero_bias_autograd_for_both_index_vectors():
    from lsa.training.native.lsa_gradients import StraightThrough
    from tests.helpers.training.kernels.inputs import dense_reference

    tensors = [x.reshape(1, 32, x.shape[1], 8) for x in inputs()]
    layer = module()
    index_queries, index_keys = layer._index_vectors(*[x.flatten(0, 1) for x in tensors[:2]], None, None)
    index_queries = index_queries.detach().requires_grad_()
    index_keys = index_keys.detach().requires_grad_()
    selected, members = layer._selected(index_queries, index_keys)
    context, lse = dense_reference(*tensors, selected, 8, 2, 8**-0.5)
    queries, keys, values = (x.transpose(1, 2) for x in tensors)
    read = (queries.detach(), keys.detach(), values.detach(), lse.detach())
    output = StraightThrough.apply(
        context.transpose(1, 2),
        index_queries,
        index_keys,
        members,
        selected,
        read,
        (8, 2, 8**-0.5, True),
        None,
    )
    gradient = torch.randn_like(output)
    got = torch.autograd.grad(output, (index_queries, index_keys), gradient)
    owner, picked = bucket_scores(index_queries, index_keys, members)
    kept = selection_table(selected, 32)
    picked_bias = picked.gather(2, owner[:, None].expand(1, 32, 32))
    scores = index_queries @ torch.nn.functional.normalize(index_keys.float(), dim=-1).transpose(1, 2)
    positions = torch.arange(32)
    distance = positions[:, None] - positions[None]
    near = (distance >= 0) & (distance < 8) & (positions[None] >= 2)
    bias = torch.where(kept, picked_bias, 0.0) + torch.where(near, scores, 0.0)
    logits = queries.float() @ keys.float().transpose(-1, -2) * 8**-0.5
    mask = (distance >= 0) & ((distance < 8) | (positions[None] < 2))
    logits = (logits + (bias - bias.detach())[:, None]).masked_fill(
        ~(mask[None] | kept)[:, None], float("-inf")
    )
    expected_output = logits.softmax(-1) @ values.float()
    expected = torch.autograd.grad(expected_output, (index_queries, index_keys), gradient)
    for ours, want in zip(got, expected, strict=True):
        torch.testing.assert_close(ours, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rotary_dim", [0, 4])
@pytest.mark.parametrize("norm", ["l2", "rms"])
def test_index_normalization_and_rotation_keep_gradients_finite(rotary_dim, norm):
    layer = module(index_rotary_dim=rotary_dim, index_norm=norm)
    layer.init_states()
    layer(*inputs()).square().sum().backward()
    assert torch.isfinite(layer.query_projection.weight.grad).all()
    if norm == "rms":
        assert torch.isfinite(layer.query_norm_weight.grad).all()
        assert layer.query_norm_weight.grad.norm() > 0


def test_lsa_sweeps_load_every_budget_training_and_normalization_choice():
    from lsa.training.native.config import load

    flat = load("configs/training/native/qwen35_0_8b_gpt2/lsa_mqa8.yaml")
    learned = load("configs/training/native/qwen35_0_8b_gpt2/lsa_indexer.yaml")
    assert {run.attention.top_k for run in flat} == {512, 1024, 2048, 4096}
    assert len(learned) == 48
    assert len({run.run.name for run in learned}) == 48


@pytest.mark.parametrize("training", ["kl", "ste"])
def test_a_sequence_without_a_closed_block_still_backpropagates(training):
    layer = LsaInnerAttention.Config(
        attention=small(
            index_input="key_and_query", index_training=training, block_size=32, window_size=32, index_dim=8
        ),
        seq_len=32,
        rotary_dim=0,
        query_head_count=4,
        kv_head_count=1,
        head_dim=8,
    ).build()
    layer(*inputs()).sum().backward()
    for projection in (layer.query_projection, layer.key_projection):
        assert torch.isfinite(projection.weight.grad).all()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"index_input": "bad"}, "index_input"),
        ({"index_training": "bad"}, "index_training"),
        ({"index_norm": "bad"}, "index_norm"),
        ({"index_dim": 0}, "index_dim"),
        ({"index_rotary_dim": 3}, "index_rotary_dim"),
        ({"index_rotary_dim": 128}, "index_rotary_dim"),
        ({"kl_query_count": 0}, "kl_query_count"),
        ({"index_loss_weight": -1}, "index_loss_weight"),
        ({"index_input": "key_and_query"}, "index_input"),
    ],
)
def test_invalid_indexer_settings_fail_at_the_config_boundary(changes, message):
    with pytest.raises(ValueError, match=message):
        small(**changes)


def test_log_mass_backward_is_the_same_in_one_row_chunks(monkeypatch):
    from lsa.training.native import lsa_gradients
    from lsa.training.native.lsa_gradients import LogMass
    from tests.helpers.training.kernels.inputs import dense_reference

    tensors = [x.reshape(1, 32, x.shape[1], 8) for x in inputs()]
    layer = module()
    index_queries, index_keys = layer._index_vectors(*[x.flatten(0, 1) for x in tensors[:2]], None, None)
    selected, _ = layer._selected(index_queries, index_keys)
    _, lse = dense_reference(*tensors, selected, 8, 2, 8**-0.5)
    queries, keys = (x.transpose(1, 2) for x in tensors[:2])
    gradient = torch.randn_like(lse)
    grads = []
    for budget in (lsa_gradients.CPU_GATHER_BYTES, 1):
        monkeypatch.setattr(lsa_gradients, "CPU_GATHER_BYTES", budget)
        attached = LogMass.apply(queries, keys, selected, lse.detach(), 8, 2, 8**-0.5)
        grads.append(torch.autograd.grad(attached, tensors[:2], gradient))
    for whole, rows in zip(*grads, strict=True):
        torch.testing.assert_close(whole, rows, rtol=1e-5, atol=1e-6)


def test_summary_tiles_hold_at_least_a_row_and_at_most_a_block():
    from lsa.training.native.lsa_summaries import tile_rows

    assert tile_rows(1, 8, 2, 128, 16, 256, 512) == 512
    assert tile_rows(1, 8, 2, 2**30, 16, 256, 512) == 1
