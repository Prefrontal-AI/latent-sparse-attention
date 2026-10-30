"""Hold the fast log-mass and straight-through gradients to the per-row PyTorch loops they replaced."""

import pytest
import torch
from torch.nn import functional

from lsa.training.native.lsa_gradients import LogMass, StraightThrough
from tests.helpers.training.kernels.inputs import (
    attention_inputs,
    dense_reference,
    make_buckets,
    random_selection,
    relative_error,
    unit_rows,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]

WINDOW, SINK, TOKENS = 32, 4, 256


def old_ids(selected, start, end):
    """List the selected, recent and sink positions of one query row as the old loop did."""
    batch = selected.shape[0]
    queries = torch.arange(start, end, device=selected.device)[:, None]
    recent = queries - torch.arange(WINDOW, device=selected.device)[None]
    recent = torch.where(recent >= 0, recent, -1).expand(batch, -1, -1)
    sinks = torch.arange(SINK, device=selected.device)[None].expand(end - start, -1)
    sinks = torch.where(sinks <= queries - WINDOW, sinks, -1).expand(batch, -1, -1)
    ids = torch.cat([selected[:, start:end].long(), recent, sinks], dim=-1)
    return ids, ids >= 0


def old_gathered(q, k, v, ids, scale):
    """Gather keys and values once per query head, repeating them over the group, as the old loop did."""
    batch, heads = q.shape[:2]
    batch_ids = torch.arange(batch, device=q.device)[:, None, None]
    keys = k.transpose(1, 2)[batch_ids, ids.clamp_min(0)].permute(0, 3, 1, 2, 4)
    values = v.transpose(1, 2)[batch_ids, ids.clamp_min(0)].permute(0, 3, 1, 2, 4)
    group = heads // k.shape[1]
    keys, values = keys.repeat_interleave(group, 1), values.repeat_interleave(group, 1)
    return torch.einsum("bhqd,bhqkd->bhqk", q.float(), keys.float()) * scale, keys, values


def old_log_mass_backward(q, k, selected, lse, gradient, scale):
    """Return the query and key gradients of the log mass, one query row at a time."""
    dq, dk = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32)
    batch, heads, tokens, dim = q.shape
    group = heads // k.shape[1]
    for start in range(tokens):
        end = start + 1
        ids, valid = old_ids(selected, start, end)
        logits, keys, _ = old_gathered(q[:, :, start:end], k, k, ids, scale)
        probability = (logits - lse[:, :, start:end, None]).exp().masked_fill(~valid[:, None], 0.0)
        ds = probability * gradient[:, :, start:end, None] * scale
        dq[:, :, start:end] = torch.einsum("bhqk,bhqkd->bhqd", ds, keys.float())
        key_grad = ds[..., None] * q[:, :, start:end, None].float()
        key_grad = key_grad.reshape(batch, k.shape[1], group, end - start, ids.shape[-1], dim).sum(2)
        at = ids.clamp_min(0)[:, None, :, :, None].expand(-1, k.shape[1], -1, -1, dim).flatten(2, 3)
        dk.scatter_add_(2, at, key_grad.flatten(2, 3))
    return dq, dk


def old_straight_through_backward(
    read, gradient, context, index_queries, index_keys, members, selected, recent
):
    """Return the index vectors' gradients of the straight-through bias, one query row at a time."""
    q, k, v, lse = read
    scale = q.shape[-1] ** -0.5
    batch, tokens, dim = index_keys.shape
    with torch.enable_grad():
        differentiable_keys = index_keys.detach().requires_grad_()
        units = functional.normalize(differentiable_keys.float(), dim=-1)
        batch_ids = torch.arange(batch, device=q.device)[:, None, None]
        sums = (units[batch_ids, members.clamp_min(0)] * (members >= 0)[..., None]).sum(2)
        centroids = functional.normalize(sums, dim=-1)
    centroid_grad = torch.zeros_like(centroids)
    unit_grad = torch.zeros_like(units)
    query_grad = torch.zeros_like(index_queries, dtype=torch.float32)
    owner = torch.zeros(batch, tokens + 1, dtype=torch.long, device=q.device)
    bucket_ids = torch.arange(members.shape[1], device=q.device).repeat_interleave(members.shape[2])
    owner.scatter_(1, torch.where(members >= 0, members, tokens).flatten(1), bucket_ids.expand(batch, -1))
    delta = (gradient.float() * context.float()).sum(-1)
    for start in range(tokens):
        end = start + 1
        ids, valid = old_ids(selected, start, end)
        logits, _, values = old_gathered(q[:, :, start:end], k, v, ids, scale)
        probability = (logits - lse[:, :, start:end, None]).exp().masked_fill(~valid[:, None], 0.0)
        value_gradient = torch.einsum("bhqd,bhqkd->bhqk", gradient[:, :, start:end].float(), values.float())
        scores_grad = (probability * (value_gradient - delta[:, :, start:end, None])).sum(1)
        picked_grad = scores_grad[..., : selected.shape[-1]]
        selected_ids = selected[:, start:end].long()
        buckets = owner.gather(1, selected_ids.clamp_min(0).flatten(1)).reshape_as(selected_ids)
        chosen = centroids.detach()[batch_ids, buckets]
        query_grad[:, start:end] += (picked_grad[..., None] * chosen).sum(2)
        at = buckets[..., None].expand(-1, -1, -1, dim).flatten(1, 2)
        updates = picked_grad[..., None] * index_queries[:, start:end, None].float()
        centroid_grad.scatter_add_(1, at, updates.flatten(1, 2))
        if recent:
            near_ids = ids[..., selected.shape[-1] : selected.shape[-1] + WINDOW]
            near_grad = scores_grad[..., selected.shape[-1] : selected.shape[-1] + WINDOW]
            near_grad = near_grad.masked_fill(near_ids < SINK, 0.0)
            near_units = units.detach()[batch_ids, near_ids.clamp_min(0)]
            query_grad[:, start:end] += (near_grad[..., None] * near_units).sum(2)
            updates = near_grad[..., None] * index_queries[:, start:end, None].float()
            at = near_ids.clamp_min(0)[..., None].expand(-1, -1, -1, dim).flatten(1, 2)
            unit_grad.scatter_add_(1, at, updates.flatten(1, 2))
    with torch.enable_grad():
        (key_grad,) = torch.autograd.grad((centroids, units), differentiable_keys, (centroid_grad, unit_grad))
    return query_grad, key_grad


def read_inputs(head_dim, seed):
    """Draw a small read, its selection, its buckets and its log-sum-exp."""
    q, k, v, generator = attention_inputs(1, TOKENS, 8, 2, head_dim, seed)
    selected = random_selection(1, TOKENS, 32, generator, window_size=WINDOW)
    index_queries = unit_rows((1, TOKENS, 32), generator)
    index_keys = unit_rows((1, TOKENS, 32), generator)
    _, members = make_buckets(index_keys, WINDOW, seed + 1)
    scale = head_dim**-0.5
    context, lse = dense_reference(q, k, v, selected, WINDOW, SINK, scale)
    return q, k, v, selected, index_queries, index_keys, members.long(), context, lse.detach(), scale


@pytest.mark.parametrize("head_dim", [64, 128])
def test_log_mass_gradients_match_the_row_loop(head_dim):
    q, k, v, selected, _, _, members, _, lse, scale = read_inputs(head_dim, 5)
    queries, keys = q.detach().transpose(1, 2), k.detach().transpose(1, 2)
    gradient = torch.randn_like(lse)
    want = old_log_mass_backward(queries, keys, selected, lse, gradient, scale)
    for use_members in (members, None):
        attached = LogMass.apply(
            queries.requires_grad_(), keys.requires_grad_(), selected, lse, WINDOW, SINK, scale, use_members
        )
        got = torch.autograd.grad(attached, (queries, keys), gradient)
        for ours, theirs in zip(got, want, strict=True):
            assert relative_error(ours, theirs) < 2e-2


@pytest.mark.parametrize("head_dim", [64, 128])
def test_chunked_log_mass_fallback_matches_the_row_loop(head_dim):
    q, k, v, selected, _, _, _, _, lse, scale = read_inputs(head_dim, 6)
    queries, keys = q.detach().float().transpose(1, 2), k.detach().float().transpose(1, 2)
    gradient = torch.randn_like(lse)
    want = old_log_mass_backward(queries, keys, selected, lse, gradient, scale)
    attached = LogMass.apply(
        queries.requires_grad_(), keys.requires_grad_(), selected, lse, WINDOW, SINK, scale
    )
    got = torch.autograd.grad(attached, (queries, keys), gradient)
    for ours, theirs in zip(got, want, strict=True):
        assert relative_error(ours, theirs) < 1e-5


@pytest.mark.parametrize("recent", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_straight_through_gradients_match_the_row_loop(head_dim, recent):
    q, k, v, selected, index_queries, index_keys, members, context, lse, scale = read_inputs(head_dim, 7)
    context = context.transpose(1, 2).to(q.dtype)
    queries, keys, values = (x.detach().transpose(1, 2) for x in (q, k, v))
    gradient = torch.randn_like(context)
    read = (queries, keys, values, lse)
    want = old_straight_through_backward(
        read, gradient, context, index_queries, index_keys, members, selected, recent
    )
    # Fp32 heads take the PyTorch chunks, bf16 the kernel
    for dtype in (torch.bfloat16, torch.float32):
        iq, ik = index_queries.clone().requires_grad_(), index_keys.clone().requires_grad_()
        output = StraightThrough.apply(
            context.to(dtype),
            iq,
            ik,
            members,
            selected,
            tuple(x.to(dtype) if x is not lse else x for x in read),
            (WINDOW, SINK, scale, recent),
            None,
        )
        got = torch.autograd.grad(output, (iq, ik), gradient.to(dtype))
        for ours, theirs in zip(got, want, strict=True):
            assert relative_error(ours, theirs) < 2e-2
