"""DSA's visibility, detached teacher, indexer inputs and token-normalized loss."""

import pytest
import torch
from torchtitan.models.common.aux_loss import AuxLoss

from lsa.training.native.config import AttentionConfig
from lsa.training.native.dsa import IndexerKl, masked_kl, selected_keys
from tests.helpers.training.native.dsa import attention


def test_topk_keeps_sinks_window_and_later_keys_on_exact_ties():
    config = AttentionConfig(kind="dsa", window_size=2, sink_size=1, top_k=2)
    keep = selected_keys(torch.zeros(1, 1, 8, 8), config)[0, 0]
    assert keep[7].tolist() == [True, False, False, False, True, True, True, True]
    assert not keep.triu(1).any()
    assert keep[0, 0]


@pytest.mark.parametrize("budget", [0, 1, 20])
def test_selection_has_exactly_the_available_long_range_budget(budget):
    config = AttentionConfig(kind="dsa", window_size=2, sink_size=1, top_k=budget)
    keep = selected_keys(torch.randn(2, 2, 8, 8), config)
    for row in range(8):
        assert torch.all(keep[:, :, row].sum(-1) == min(row + 1, 3) + min(max(row - 2, 0), budget))


def test_masked_kl_matches_reference_and_only_trains_the_student():
    teacher = torch.randn(2, 2, 3, 6, dtype=torch.float64, requires_grad=True)
    student = torch.randn_like(teacher, requires_grad=True)
    mask = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
        ]
    )
    loss, stats = masked_kl(teacher, student, mask)
    expected = (
        sum(
            torch.nn.functional.kl_div(
                student[:, :, row, : row + 1].log_softmax(-1),
                teacher[:, :, row, : row + 1].detach().softmax(-1),
                reduction="sum",
            )
            for row in range(3)
        )
        / 12
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert torch.all(student.grad.masked_select(~mask.expand_as(student)) == 0)
    assert all(torch.isfinite(value) for value in stats.values())


def test_aux_loss_uses_global_step_tokens_across_microbatches():
    AuxLoss.set_step_denominator(torch.tensor(12.0))
    auxiliary = IndexerKl.Config(coeff=0.25).build()
    loss = torch.tensor(2.0, requires_grad=True)
    carrier = torch.ones(2, requires_grad=True)
    first = auxiliary(loss, carrier, 4)
    second = auxiliary(loss, carrier, 8)
    torch.testing.assert_close(first, carrier)
    (first.sum() + second.sum()).backward()
    torch.testing.assert_close(loss.grad, torch.tensor(0.25))
    torch.testing.assert_close(auxiliary.instance_acc, torch.tensor(2.0))


@pytest.mark.parametrize("source", ["attention_input", "key_and_query", "sliced_key_and_query"])
@pytest.mark.parametrize("detached", [False, True])
def test_indexer_detach_only_cuts_its_source_gradient(source, detached):
    layer = attention(index_input=source, index_detached=detached)
    inputs = [
        torch.randn(8, 16, requires_grad=True),
        torch.randn(8, 2, 8, requires_grad=True),
        torch.randn(8, 2, 8, requires_grad=True),
    ]
    vectors = layer.index_vectors(*inputs)
    sum(value.square().sum() for value in vectors).backward()
    assert all(
        parameter.grad is not None
        for parameter in (layer.query_projection.weight, layer.key_projection.weight)
    )
    source_gradient = inputs[0].grad if source == "attention_input" else inputs[1].grad
    assert (source_gradient is None) == detached


@pytest.mark.parametrize("length", [2, 8])
@pytest.mark.parametrize("selection", [False, True])
def test_forward_backward_initializes_and_trains_indexer(length, selection):
    layer = attention(length=length, kl_on_selection=selection)
    AuxLoss.set_step_denominator(torch.tensor(float(length)))
    inputs = torch.randn(length, 16, requires_grad=True)
    layer.measure = True
    output = layer(inputs, None)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )
    if length == 8:
        assert torch.isfinite(torch.tensor(list(layer.measurements.values()))).all()


def test_sparse_stage_begins_after_the_configured_completed_update_count():
    layer = attention(sparse_start_step=3)
    layer.train()
    AuxLoss.set_step_denominator(torch.tensor(8.0))
    inputs = torch.randn(8, 16)
    with torch.no_grad():
        layer.set_step(2)
        dense = layer(inputs, None)
        layer.set_step(3)
        sparse = layer(inputs, None)
    assert not torch.equal(dense, sparse)


@pytest.mark.parametrize(
    "knobs, message",
    [
        ({"index_input": "bad"}, "index_input"),
        ({"index_dim": 0}, "index_dim"),
        ({"index_head_count": 0}, "index_head_count"),
        ({"kl_query_count": 0}, "kl_query_count"),
        ({"sparse_start_step": -1}, "sparse_start_step"),
        ({"index_loss_weight": -1}, "index_loss_weight"),
        ({"index_bias": True, "index_relu": True}, "index_bias"),
    ],
)
def test_invalid_dsa_configuration_is_rejected(knobs, message):
    with pytest.raises(ValueError, match=message):
        AttentionConfig(kind="dsa", **knobs)


def test_slice_must_fit_its_attention_head():
    with pytest.raises(ValueError, match="at most head_dim"):
        attention(index_input="sliced_key_and_query", index_dim=20)


def test_bias_detached_reprojection_trains_separately_from_live_bias():
    layer = attention(index_bias=True, index_detached=True)
    queries = torch.randn(1, 2, 8, 4, requires_grad=True)
    keys = torch.randn_like(queries, requires_grad=True)
    rank_queries, rank_keys = layer.rank_vectors(queries, keys)
    (rank_queries.square().sum() + rank_keys.square().sum()).backward()
    assert queries.grad is None and keys.grad is None
    assert layer.kl_query_projection.weight.grad is not None
    assert layer.kl_key_projection.weight.grad is not None
    assert layer.index_bias_scale == 0


@pytest.mark.parametrize("detached", [False, True])
def test_bias_forward_backward_initializes_all_parameters(detached):
    layer = attention(index_bias=True, index_detached=detached)
    AuxLoss.set_step_denominator(torch.tensor(8.0))
    output = layer(torch.randn(8, 16), None)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_evaluation_collects_indexer_stats_without_injecting_a_loss():
    layer = attention()
    layer.eval()
    layer.measure = True
    with torch.no_grad():
        layer(torch.randn(8, 16), None)
        layer(torch.randn(8, 16), None)
    assert layer.measurements["passes"] == 2
    assert "dsa/kl" in layer.measurements


def test_default_dsa_source_resolves_without_changing_the_legacy_lsa_source():
    assert AttentionConfig(kind="dsa").index_input == "attention_input"
    assert AttentionConfig(kind="lsa", coarse=False).index_input == "none"


def test_sliced_query_and_key_input_is_exclusive_to_dsa():
    with pytest.raises(ValueError, match="index_input"):
        AttentionConfig(kind="lsa", coarse=False, index_input="sliced_key_and_query", index_training="kl")
