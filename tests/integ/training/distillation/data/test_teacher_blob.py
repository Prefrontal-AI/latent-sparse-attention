"""What the extraction writes is exactly what the trainer's reader reads, on a model output the size of a toy."""

from types import SimpleNamespace

import pytest
import torch

from lsa.evaluation.adapter.layer import TeacherView
from lsa.sglang.settings.design import KEY_AND_QUERY, IndexerConfig
from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data import teacher_plan
from lsa.training.distillation.data.cache import Document, layer_path, write
from lsa.training.distillation.data.native import KeyAndQueryReader, LatentKeyAndQuery
from lsa.training.distillation.data.teacher import attention_fields, layer_blob
from lsa.training.distillation.data.teacher_config import (
    DataConfig,
    QueryConfig,
    RunConfig,
    TeacherConfig,
    TeacherModelConfig,
)
from tests.helpers.training.distillation.data import fake_attention

LENGTH, MODEL_DIM, HEADS, KV_HEADS, HEAD_DIM = 160, 12, 4, 2, 6
QUERIES = 6, 2, 5


def an_indexer(window_size=16):
    return IndexerConfig(
        window_size=window_size, sink_size=4, bucket_size=4, bucket_cap=8, pool_size=None, top_k=8, top_p=1
    )


def a_run():
    train, tail, top = QUERIES
    return TeacherConfig(
        model=TeacherModelConfig(name="glm53_flash", indexer=an_indexer()),
        data=DataConfig(queries=QueryConfig(train=train, tail=tail, top=top)),
        run=RunConfig(output="targets"),
    )


def a_reader(layer, generator):
    """A layer's wrapper at toy size, whose view is random heads of the right shapes"""
    attention = torch.nn.Linear(1, 1)
    attention.layer_idx = layer
    attention.scaling = HEAD_DIM**-0.5

    def teacher_view(hidden, positions, embeddings):
        q = torch.randn(HEADS, len(positions), HEAD_DIM, generator=generator)
        k = torch.randn(KV_HEADS, LENGTH, HEAD_DIM, generator=generator)
        index = torch.randn(1, 1)
        return TeacherView(index, index, q, k)

    return SimpleNamespace(attention=attention, teacher_view=teacher_view)


@pytest.fixture
def written(tmp_path):
    generator = torch.Generator().manual_seed(0)
    run = a_run()
    positions = teacher_plan.query_positions(run, LENGTH, 0)
    hidden = torch.randn(1, LENGTH, MODEL_DIM, generator=generator)
    blob = layer_blob(a_reader(3, generator), hidden, None, positions, run, "doc0")
    path = layer_path(tmp_path, "tokens", "train", "doc0", 3)
    write(path, blob)
    return SimpleNamespace(run=run, positions=positions, hidden=hidden, blob=blob, path=path)


def test_a_blob_has_no_index_vectors(written):
    assert not {"index_keys", "index_queries"} & set(written.blob)


def test_x_is_the_layer_input_at_bf16(written):
    x = written.blob[Cache.X]
    assert x.dtype is torch.bfloat16
    assert torch.equal(x, written.hidden[0].bfloat16())


def test_the_dtypes_and_shapes_are_the_ones_the_cache_documents(written):
    blob = written.blob
    assert blob["query_positions"].dtype is torch.int32
    assert blob["top_key_ids"].dtype is torch.int32 and blob["top_key_ids"].shape == (8, 5)
    assert blob["top_key_mass"].dtype is torch.float16 and blob["top_key_mass"].shape == (8, 5)
    assert blob["peak_key_id"].dtype is torch.int32 and blob["peak_key_id"].shape == (8, HEADS)


def test_the_reader_takes_what_the_writer_wrote(written):
    document = Document(written.path, an_indexer())
    assert (document.name, document.layer, document.key_count, document.model_dim) == (
        "doc0",
        3,
        LENGTH,
        MODEL_DIM,
    )
    assert document.train_query_count == 6
    assert len(document.eval_rows()) == 2
    assert torch.equal(document.query_positions, written.positions)
    assert document.head_count == HEADS
    target = document.key_target(document.train_rows())
    assert torch.allclose(target.sum(-1), torch.ones(6), atol=1e-5)


def test_queries_start_past_the_sinks_and_the_window(written):
    training = written.positions[: QUERIES[0]]
    assert int(training.min()) >= 4 + 16


def test_every_key_with_mass_is_one_the_design_may_read(written):
    ids = written.blob["top_key_ids"].long()
    carries = written.blob["top_key_mass"] > 0
    positions = written.blob["query_positions"].long()[:, None].expand_as(ids)
    assert bool(carries.any())
    assert bool((ids[carries] >= 4).all())
    assert bool((positions[carries] - ids[carries] >= 16).all())


def test_a_reader_under_another_window_refuses_the_file(written):
    with pytest.raises(ValueError, match="different visibility rule"):
        Document(written.path, an_indexer(window_size=8))


@torch.no_grad()
def test_an_extraction_under_the_key_and_query_writes_what_a_run_reads_off_its_x(written):
    """`run.input: key_and_query` keeps the tensors a run on `x` reads off the same layer's `x`"""
    attention = fake_attention.attention(model_dim=MODEL_DIM)
    attention.parameters = attention.kv_a_proj_with_mqa.parameters
    attention.layer_idx, attention.scaling = 3, HEAD_DIM**-0.5
    reader = a_reader(3, torch.Generator().manual_seed(0))
    reader.attention = attention
    blob = layer_blob(
        reader, written.hidden, None, written.positions, a_run(), "doc0", reading=LatentKeyAndQuery()
    )

    with_x = {**written.blob, Cache.X: written.hidden[0].bfloat16()}
    expected = KeyAndQueryReader({3: attention}, reading=LatentKeyAndQuery())(
        Document.from_blob(written.path, with_x, an_indexer())
    )
    assert Cache.X not in blob and blob[Cache.META][Cache.INPUT] == KEY_AND_QUERY
    for name in Cache.KEY_AND_QUERY_FIELDS:
        assert torch.equal(blob[name], getattr(expected, name)), name


def test_an_extraction_under_queries_attention_keeps_the_models_own_keys_and_queries_and_reads_them_back(
    tmp_path,
):
    generator = torch.Generator().manual_seed(0)
    run = a_run()
    run.data.queries.attention = True
    positions = teacher_plan.query_positions(run, LENGTH, 0)
    reader = a_reader(3, generator)
    blob = layer_blob(
        reader, torch.randn(1, LENGTH, MODEL_DIM, generator=generator), None, positions, run, "doc0"
    )
    assert blob[Cache.ATTENTION_KEYS].shape == (LENGTH, KV_HEADS, HEAD_DIM)
    assert blob[Cache.ATTENTION_QUERIES].shape == (len(positions), HEADS, HEAD_DIM)
    assert blob[Cache.ATTENTION_KEYS].dtype == blob[Cache.ATTENTION_QUERIES].dtype == torch.bfloat16
    path = layer_path(tmp_path, "tokens", "train", "doc0", 3)
    write(path, blob)
    document = Document(path, an_indexer())
    assert torch.equal(document.attention_keys, blob[Cache.ATTENTION_KEYS])
    assert document.attention_scale == HEAD_DIM**-0.5


def test_queries_attention_refuses_query_heads_that_do_not_group_over_the_kv_heads():
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="group evenly"):
        attention_fields(
            a_reader(3, generator), torch.randn(3, 4, HEAD_DIM), torch.randn(2, LENGTH, HEAD_DIM)
        )


def test_a_cache_without_queries_attention_holds_none_of_the_models_own_keys(written):
    document = Document(written.path, an_indexer())
    assert document.attention_keys is None and document.attention_scale is None
