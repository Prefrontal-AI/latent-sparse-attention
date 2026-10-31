"""The checkpoint's own selection, computed offline and kept beside the layer file."""

import numpy as np
import pytest
import torch

from lsa.core import RotaryLayout
from lsa.sglang.index.layer_index import KeyAndQueryProjection
from lsa.sglang.settings.design import ATTENTION_INPUT, KEY_AND_QUERY
from lsa.training.distillation.config import DocumentSet
from lsa.training.distillation.constants import Cache, Native, Target
from lsa.training.distillation.data.cache import (
    Document,
    cached_layers,
    layer_path,
    native_path,
    recorded_rotary,
    write,
)
from lsa.training.distillation.data.native import (
    KeyAndQueryReader,
    agreement,
    key_and_query,
    latent_bases,
    layer_files,
    pending,
    summed_queries,
)
from lsa.training.distillation.data.pool import Entry, require_native, stream
from lsa.training.distillation.model import IndexMap
from tests.helpers.training.distillation.data import fake_attention, fake_cache


def sidecar(path, picked):
    write(native_path(path), {Cache.NATIVE: picked.int(), Cache.META: {}})


def written(tmp_path, layers=(0,)):
    fake_cache.collection(tmp_path, documents=("source0", "source1"), layers=layers)
    return [Entry((tmp_path, "tokens", "train"), name, fake_cache.LENGTH) for name in ("source0", "source1")]


def teacher_half(path):
    ids = torch.load(path, weights_only=False)["top_key_ids"]
    return torch.cat([ids[:, : fake_cache.TOP // 2], torch.full((len(ids), 3), -1)], 1)


def test_a_sidecar_sits_beside_its_layer_file_and_is_not_a_layer(tmp_path):
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    assert native_path(path).name == "source0_layer0_native.pt"
    written(tmp_path)
    sidecar(path, teacher_half(path))
    assert cached_layers(tmp_path, DocumentSet(name="t", source="tokens", split="train")) == [0]
    assert sorted(found.name for found, _ in layer_files(path.parent)) == [
        "source0_layer0.pt",
        "source1_layer0.pt",
    ]


def test_a_document_reads_its_sidecar_when_asked_and_builds_the_target_from_it(tmp_path):
    written(tmp_path)
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    picked = teacher_half(path)
    sidecar(path, picked)

    plain = Document(path, fake_cache.config())
    document = Document(path, fake_cache.config(), native=True)
    assert plain.native_key_ids is None
    assert torch.equal(document.native_key_ids, picked.int())

    rows = document.eval_rows()
    dense = document.key_target(rows)
    target = document.key_target(rows, target=Target.DENSE_ON_NATIVE)
    members = document.native_members(rows)
    kept = dense * members
    assert torch.allclose(target, kept / kept.sum(-1, keepdim=True), atol=1e-5)
    assert not bool((target > 0)[~members].any())


def test_a_document_without_a_selection_is_refused_when_one_is_needed(tmp_path):
    written(tmp_path)
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    with pytest.raises(FileNotFoundError, match="native-picks"):
        Document(path, fake_cache.config(), native=True)


def test_a_selection_computed_for_other_positions_is_refused(tmp_path):
    written(tmp_path)
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    sidecar(path, teacher_half(path)[:-1])
    with pytest.raises(ValueError, match="other positions"):
        Document(path, fake_cache.config(), native=True)


def test_a_run_refuses_to_start_while_any_selection_is_missing(tmp_path):
    entries = written(tmp_path)
    sidecar(entries[0].path(0), teacher_half(entries[0].path(0)))
    with pytest.raises(FileNotFoundError, match=r"1 of 2.*source1_layer0_native.pt"):
        require_native(entries, [0])
    sidecar(entries[1].path(0), teacher_half(entries[1].path(0)))
    require_native(entries, [0])


def test_streaming_with_native_hands_over_documents_that_carry_it(tmp_path):
    entries = written(tmp_path)
    for entry in entries:
        sidecar(entry.path(0), teacher_half(entry.path(0)))
    documents = list(stream([entry.path(0) for entry in entries], fake_cache.config(), native=True))
    assert all(document.native_key_ids is not None for document in documents)


def test_the_pass_does_only_the_files_that_exist_and_lack_a_selection(tmp_path):
    entries = written(tmp_path)
    directory = tmp_path / "tokens"
    # A file still being written is a scratch name, which no layer file matches
    (directory / "train" / "source2_layer0.123.tmp").write_bytes(b"")
    assert sorted(path.name for path, _ in pending(directory)) == ["source0_layer0.pt", "source1_layer0.pt"]
    sidecar(entries[0].path(0), teacher_half(entries[0].path(0)))
    assert [path.name for path, _ in pending(directory)] == ["source1_layer0.pt"]


def test_agreement_tells_a_reordering_from_a_different_pick():
    indexed = torch.tensor([[1, 2, 3, -1], [4, 5, 6, 7], [8, 9, 10, 11]])
    picked = torch.tensor([[3, 2, 1, -1], [4, 5, 6, 7], [8, 9, 10, 12]])
    result = agreement(picked, indexed)
    assert result["same_order"].tolist() == [False, True, False]
    assert result["same_set"].tolist() == [True, True, False]
    assert np.allclose(result["overlap"], [1.0, 1.0, 0.75])


def a_document(tmp_path):
    written(tmp_path)
    return Document(layer_path(tmp_path, "tokens", "train", "source0", 0), fake_cache.config())


@torch.no_grad()
def test_the_key_and_query_a_run_reads_off_x_is_the_served_projections_input(tmp_path):
    document = a_document(tmp_path)
    attention = fake_attention.attention()
    read = KeyAndQueryReader({0: attention})(document)
    served = key_and_query(attention, document.x, document.query_positions.long())
    rows = torch.tensor([0, 3, 9])
    assert read.x is None and document.x is not None
    assert torch.equal(read.key_inputs(), served.key)
    assert torch.equal(read.query_inputs(rows), served.query.sum(1, dtype=torch.float32)[rows])
    assert read.input_dim == fake_attention.RANK


@torch.no_grad()
def test_a_map_trained_on_the_key_and_query_reads_as_the_server_projects_it(tmp_path):
    """The map's index keys and queries are the served projection's, its bases folded in, to bf16"""
    document = a_document(tmp_path)
    attention = fake_attention.attention()
    reader = KeyAndQueryReader({0: attention})
    read = reader(document)
    served = key_and_query(attention, document.x, document.query_positions.long())
    index_map = IndexMap(
        fake_attention.RANK, fake_cache.INDEX_DIM, init=read.key_inputs(), bases=reader.bases(0)
    )
    rows = torch.arange(len(document.query_positions))
    projection = KeyAndQueryProjection(*(tensor.detach() for tensor in index_map.served_projections()), "cpu")

    def cosines(trained, server):
        return torch.nn.functional.cosine_similarity(trained.float(), server.float(), dim=-1).min()

    assert cosines(index_map.index_keys(read.key_inputs()), projection.index_keys(served.key)) > 0.995
    assert (
        cosines(index_map.index_queries(read.query_inputs(rows)), projection.index_queries(served.query))
        > 0.995
    )


def test_the_summed_queries_are_the_same_folded_a_few_rows_at_a_time(tmp_path, monkeypatch):
    document = a_document(tmp_path)
    attention = fake_attention.attention()
    whole = summed_queries(attention, document.x, document.query_positions.long())
    monkeypatch.setattr(Native, "QUERY_ROWS", 2)
    assert torch.equal(summed_queries(attention, document.x, document.query_positions.long()), whole)


@torch.no_grad()
def test_the_latent_bases_keep_every_logit_and_make_the_keys_metric_the_plain_one():
    """`key @ L` against `query @ L^-T` is `key . query`, and `L L^T` is the metric the key maps impose"""
    attention = fake_attention.attention()
    key_basis, query_basis = latent_bases(attention)
    per_head = attention.kv_b_proj.weight.double().view(fake_attention.HEADS, -1, fake_attention.RANK)
    key_maps = per_head[:, : fake_attention.KEY_DIM]
    metric = torch.einsum("hdr,hde->re", key_maps, key_maps)
    assert torch.allclose((key_basis @ key_basis.t()).double(), metric, atol=1e-4)
    keys, queries = torch.randn(5, fake_attention.RANK), torch.randn(3, fake_attention.RANK)
    assert torch.allclose((queries @ query_basis) @ (keys @ key_basis).t(), queries @ keys.t(), atol=1e-4)


def test_a_document_holding_the_key_and_query_reads_as_it_is_and_one_holding_neither_says_so(tmp_path):
    reader = KeyAndQueryReader({0: fake_attention.attention()})
    read = reader(a_document(tmp_path))
    assert reader(read) is read
    read.key = None
    with pytest.raises(ValueError, match="no x left"):
        reader(read)


@torch.no_grad()
def test_a_file_holding_the_key_and_query_reads_what_a_run_read_off_its_x(tmp_path):
    entries = written(tmp_path)
    path = entries[0].path(0)
    attention = fake_attention.attention()
    indexer = fake_cache.config(input=KEY_AND_QUERY)
    before = KeyAndQueryReader({0: attention})(Document(path, indexer))
    blob = fake_cache.keep_key_and_query(path, attention)

    after = KeyAndQueryReader({0: attention})(Document(path, indexer))
    assert Cache.X not in blob and blob[Cache.META][Cache.INPUT] == KEY_AND_QUERY
    for name in ("key", "summed_query", "query_positions", "top_key_ids", "top_key_mass", "peak_key_id"):
        assert torch.equal(getattr(after, name), getattr(before, name)), name
    assert after.head_queries.shape == (len(after.query_positions), fake_attention.HEADS, fake_attention.RANK)
    assert torch.allclose(after.head_queries.sum(1, dtype=torch.float32), after.summed_query)


def test_a_run_on_x_refuses_a_file_holding_the_key_and_query(tmp_path):
    path = written(tmp_path)[0].path(0)
    fake_cache.keep_key_and_query(path, fake_attention.attention())
    with pytest.raises(ValueError, match="needs a cache of x"):
        Document(path, fake_cache.config(input=ATTENTION_INPUT))


@torch.no_grad()
def test_an_index_vector_cache_reads_the_key_and_query_its_index_vectors_were_carried_from(tmp_path):
    """An index-vector cache holds `rownorm(latent @ L)` and `rownorm(q @ L^-T)`, and a map reads them as it reads the ones off x"""
    document = a_document(tmp_path)
    attention = fake_attention.attention()
    reader = KeyAndQueryReader({0: attention})
    read = reader(document)
    key_basis, query_basis = reader.bases(0)
    blob = torch.load(layer_path(tmp_path, "tokens", "train", "source0", 0), weights_only=False)
    del blob[Cache.X]
    blob["index_keys"] = torch.nn.functional.normalize(read.key_inputs().float() @ key_basis, dim=-1).half()
    blob["index_queries"] = torch.nn.functional.normalize(read.summed_query @ query_basis, dim=-1).half()
    blob[Cache.META] = {
        "document": "source0",
        "layer": 0,
        "block": 256,
        "window": fake_cache.WINDOW,
        "sink": fake_cache.SINK,
        "length": fake_cache.LENGTH,
        "train_queries": fake_cache.TRAIN_QUERIES,
    }
    path = layer_path(tmp_path, "index_vectors", "train", "source0", 0)
    write(path, blob)
    indexed = Document(path, fake_cache.config())
    assert indexed.closed_block == 256 and indexed.x is None
    indexed_read = reader(indexed)
    assert indexed_read.index_keys is None and indexed_read.input_dim == fake_attention.RANK
    index_map = IndexMap(
        fake_attention.RANK, fake_cache.INDEX_DIM, init=read.key_inputs(), bases=(key_basis, query_basis)
    )
    rows = torch.arange(len(document.query_positions))

    def cosines(left, right):
        return torch.nn.functional.cosine_similarity(left, right, dim=-1).min()

    assert (
        cosines(index_map.index_keys(indexed_read.key_inputs()), index_map.index_keys(read.key_inputs()))
        > 0.99
    )
    index_queries = index_map.index_queries(indexed_read.query_inputs(rows))
    assert cosines(index_queries, index_map.index_queries(read.query_inputs(rows))) > 0.99
    assert KeyAndQueryReader({0: attention}, closed_block=64)(document).closed_block == 64


def test_a_manifest_names_the_rotation_span_dim():
    assert recorded_rotary({}) == RotaryLayout()
    assert recorded_rotary({"rotary": {"end": "prefix", "dim": 64}}) == RotaryLayout(end="prefix", dim=64)
