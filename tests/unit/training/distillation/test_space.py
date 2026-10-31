"""The learned maps - where they start, what they preserve, and what a warm start refuses."""

import math
from pathlib import Path

import pytest
import torch

from lsa.core import rownorm
from lsa.training import distilled_maps
from lsa.training.distillation.checkpoint import checkpoint, load_maps
from lsa.training.distillation.model import IndexMap, init_rows

MODEL_DIM = 16


def rows_of_x(count=64, dim=MODEL_DIM):
    torch.manual_seed(0)
    return torch.randn(count, dim).bfloat16()


def test_the_map_has_a_key_projection_and_a_query_projection_of_the_same_shape():
    index_map = IndexMap(MODEL_DIM, 4, init=rows_of_x())
    assert index_map.key_projection.shape == (MODEL_DIM, 4)
    assert index_map.query_projection.shape == (MODEL_DIM, 4)
    assert (index_map.model_dim, index_map.index_dim) == (MODEL_DIM, 4)


def test_a_map_writes_unit_rows_of_its_own_dim():
    x = rows_of_x()
    index_map = IndexMap(MODEL_DIM, 4, init=x)
    for written in (index_map.index_keys(x), index_map.index_queries(x)):
        assert written.shape == (64, 4)
        assert torch.allclose(written.norm(dim=-1), torch.ones(64), atol=1e-5)


@torch.no_grad()
def test_an_index_key_is_x_projected_and_normalized():
    x = rows_of_x()
    index_map = IndexMap(MODEL_DIM, 4)
    expected = torch.nn.functional.normalize(x.float() @ index_map.key_projection, dim=-1)
    assert torch.allclose(index_map.index_keys(x), expected, atol=1e-6)


@torch.no_grad()
def test_projecting_a_long_document_in_chunks_is_projecting_it_whole(monkeypatch):
    from lsa.training.distillation import model

    x = rows_of_x(count=100)
    index_map = IndexMap(MODEL_DIM, 4)
    whole = index_map.index_keys(x)
    monkeypatch.setattr(model.Space, "PROJECT_CHUNK", 7)
    assert torch.allclose(index_map.index_keys(x), whole, atol=1e-6)


def test_the_chunked_projection_gives_the_gradient_of_the_plain_one():
    x = rows_of_x(count=50)
    chunked = IndexMap(MODEL_DIM, 4)
    plain = IndexMap(MODEL_DIM, 4)
    plain.key_projection.data.copy_(chunked.key_projection.data)
    weights = torch.randn(50, 4)
    (chunked.index_keys(x) * weights).sum().backward()
    rows = x.float() @ plain.key_projection
    (torch.nn.functional.normalize(rows, dim=-1) * weights).sum().backward()
    assert torch.allclose(chunked.key_projection.grad, plain.key_projection.grad, atol=1e-5)


@torch.no_grad()
def test_a_full_dim_pca_init_is_a_rotation_and_keeps_every_cosine():
    x = rows_of_x()
    index_map = IndexMap(MODEL_DIM, MODEL_DIM, init=x)
    unit = torch.nn.functional.normalize(x.float(), dim=-1)
    before = unit @ unit.t()
    after = index_map.index_keys(x) @ index_map.index_keys(x).t()
    assert torch.allclose(before, after, atol=1e-4)


@torch.no_grad()
def test_the_pca_init_keeps_more_of_the_variance_than_a_random_map():
    x = rows_of_x(count=256)
    unit = torch.nn.functional.normalize(x.float(), dim=-1)
    spread = unit @ unit.t()
    learned = IndexMap(MODEL_DIM, 4, init=x).index_keys(x)
    random = IndexMap(MODEL_DIM, 4).index_keys(x)
    assert float((learned @ learned.t() - spread).pow(2).mean()) < float(
        (random @ random.t() - spread).pow(2).mean()
    )


def test_an_index_wider_than_the_x_it_reads_is_refused():
    with pytest.raises(ValueError, match="cannot be wider"):
        IndexMap(8, 16)


def test_the_temperature_is_a_trained_parameter_that_can_also_be_pinned():
    index_map = IndexMap(8, 4)
    index_map.set_tau(0.05)
    assert index_map.tau == pytest.approx(0.05)
    assert index_map.log_tau.requires_grad
    assert float(index_map.log_tau.detach()) == pytest.approx(math.log(0.05))


def test_the_init_sample_spreads_over_the_document():
    rows = torch.arange(1000).float()[:, None].expand(1000, 4)
    sampled = init_rows(rows, 10)
    assert len(sampled) == 10
    assert float(sampled[-1][0]) >= 900


def test_a_checkpoint_round_trips_every_layer():
    maps = {3: IndexMap(8, 4), 7: IndexMap(8, 4)}
    blob = checkpoint(maps, {"note": "x"})
    fresh = {3: IndexMap(8, 4), 7: IndexMap(8, 4)}
    load_maps(blob, fresh)
    assert torch.allclose(fresh[7].key_projection, maps[7].key_projection)
    assert torch.allclose(fresh[7].query_projection, maps[7].query_projection)


def test_a_checkpoint_missing_a_layer_says_which():
    blob = checkpoint({3: IndexMap(8, 4)}, {})
    with pytest.raises(ValueError, match=r"\[7\]"):
        load_maps(blob, {3: IndexMap(8, 4), 7: IndexMap(8, 4)})


def some_bases(dim=MODEL_DIM):
    torch.manual_seed(1)
    key_basis = torch.linalg.qr(torch.randn(dim, dim))[0] * torch.linspace(0.5, 2.0, dim)
    return key_basis, torch.linalg.inv(key_basis).t()


def test_a_map_in_bases_projects_what_it_reads_carried_into_them_as_the_freeze_did():
    """The freeze's cache held `normalize(key @ L)`, and normalizing before a linear map changes no direction"""
    key_basis, query_basis = some_bases()
    index_map = IndexMap(MODEL_DIM, 4, bases=(key_basis, query_basis))
    keys, queries = torch.randn(9, MODEL_DIM), torch.randn(3, MODEL_DIM)
    frozen_keys = rownorm(rownorm(keys @ key_basis) @ index_map.key_projection)
    frozen_queries = rownorm(rownorm(queries @ query_basis) @ index_map.query_projection)
    assert torch.allclose(index_map.index_keys(keys), frozen_keys, atol=1e-5)
    assert torch.allclose(index_map.index_queries(queries), frozen_queries, atol=1e-5)


def test_a_map_in_bases_starts_at_the_pca_of_the_keys_carried_into_the_key_basis():
    key_basis, query_basis = some_bases()
    rows = rows_of_x()
    in_bases = IndexMap(MODEL_DIM, 4, init=rows, bases=(key_basis, query_basis))
    carried = IndexMap(MODEL_DIM, 4, init=rownorm(rows.float() @ key_basis))
    assert torch.allclose(in_bases.key_projection, carried.key_projection)
    assert torch.allclose(in_bases.query_projection, carried.query_projection)


def test_a_checkpoint_in_bases_keeps_them_and_is_read_with_them_folded_in():
    key_basis, query_basis = some_bases(8)
    maps = {3: IndexMap(8, 4, bases=(key_basis, query_basis))}
    blob = checkpoint(maps, {"step": 0, "teacher": {"model": "glm53_flash"}})
    projection = distilled_maps.decode(blob, Path("maps.pt")).projections[3]
    served_query, served_key = maps[3].served_projections()
    assert torch.allclose(projection.key_projection, served_key.detach())
    assert torch.allclose(projection.query_projection, served_query.detach())
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_maps(blob, {3: IndexMap(8, 4)})
