"""The full dense target - what it is, that it is what the extraction ranks, and that chunking does not move it."""

from types import SimpleNamespace

import pytest
import torch

from lsa.sglang.settings.design import KEY_AND_QUERY
from lsa.training.distillation.config import ObjectiveConfig
from lsa.training.distillation.constants import Cache, Target, Term
from lsa.training.distillation.data.cache import Document, layer_path
from lsa.training.distillation.data.teacher_attention import (
    full_attention_target,
    latent_attention_scale,
    teacher_attention,
)
from lsa.training.distillation.losses import key_divergence, key_scores, step_terms, teacher_entropy
from lsa.training.distillation.model import IndexMap, init_rows
from tests.helpers.training.distillation.data import fake_cache

SINK, WINDOW = 4, 32
HEADS, KEY_DIM, LENGTH = 6, 16, 200
# Two rows that read nothing - one inside the sinks, one inside the window of the first readable key - and rows that read some
POSITIONS = torch.tensor([2, 10, 36, 40, 100, 150, 199])


def latent_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(LENGTH, KEY_DIM, generator=generator).bfloat16()
    head_queries = torch.randn(len(POSITIONS), HEADS, KEY_DIM, generator=generator).bfloat16()
    return head_queries, key, KEY_DIM**-0.5


def reference(head_queries, key, scale, positions):
    """Each head's softmax over the readable keys, one row and one head at a time, averaged over heads"""
    target = torch.zeros(len(positions), len(key))
    for row, position in enumerate(positions.tolist()):
        readable = [j for j in range(len(key)) if j >= SINK and position - j >= WINDOW]
        if not readable:
            continue
        for head in range(head_queries.shape[1]):
            logits = torch.stack(
                [head_queries[row, head].double() @ key[j].double() * scale for j in readable]
            )
            target[row, readable] += (torch.softmax(logits, 0) / head_queries.shape[1]).float()
    return target


def full(chunk_keys=None, **kwargs):
    head_queries, key, scale = latent_inputs(**kwargs)
    cells = None if chunk_keys is None else len(POSITIONS) * HEADS * chunk_keys
    arguments = {} if cells is None else {"cells": cells}
    return full_attention_target(head_queries, key, scale, POSITIONS, SINK, WINDOW, **arguments)


def test_the_target_is_every_heads_softmax_over_the_readable_keys_averaged():
    head_queries, key, scale = latent_inputs()
    assert torch.allclose(full(), reference(head_queries, key, scale, POSITIONS), atol=1e-6)


def test_the_target_is_a_distribution_over_what_a_row_may_read_and_nothing_else():
    target = full()
    keys = torch.arange(LENGTH)
    readable = (keys[None] >= SINK) & (POSITIONS[:, None] - keys[None] >= WINDOW)
    assert not bool((target[~readable] != 0).any())
    has_keys = readable.any(-1)
    assert torch.allclose(target[has_keys].sum(-1), torch.ones(int(has_keys.sum())), atol=1e-5)


def test_a_row_that_reads_nothing_is_all_zero_and_never_nan():
    target = full()
    assert not bool(target.isnan().any())
    # Position 2 sits in the sinks and 10 is inside the window of the first readable key
    assert bool((target[:2] == 0).all())
    assert bool((target[2:].sum(-1) > 0).all())


@pytest.mark.parametrize("chunk_keys", [1, 7, 64, 199, 200, 1000])
def test_chunking_the_keys_does_not_move_the_target(chunk_keys):
    assert torch.allclose(full(chunk_keys), full(), atol=1e-6, rtol=1e-5)


def test_the_target_ranks_what_the_extractions_own_function_ranks():
    head_queries, key, scale = latent_inputs()
    top = 16
    # The pair of vectors the extraction dots, as the cache holds them: heads first, one shared key
    ids, mass, _ = teacher_attention(
        head_queries.transpose(0, 1).float(),
        key.float()[None],
        scale,
        POSITIONS,
        fake_cache.config(window_size=WINDOW, sink_size=SINK),
        top,
    )
    target = full()
    # The extraction's softmax over nothing is nan, so rows that read nothing are not compared
    reads = torch.tensor([bool(row.sum() > 0) for row in target])
    best = target.topk(top, dim=-1)
    assert torch.equal(best.indices[reads].int(), ids[reads])
    # Its mass is summed over heads and kept in fp16, ours averaged and kept in fp32
    assert torch.allclose(mass[reads].float() / HEADS, best.values[reads], atol=2e-4, rtol=2e-3)
    # What it cuts off is what the target keeps - the sum of its kept masses is the target's own on those keys
    assert torch.allclose(target.gather(1, ids.long())[reads], best.values[reads])


def full_document(tmp_path, **meta):
    """A cached document holding the key latent and each head's query in place of `x`"""
    blob = fake_cache.document()
    blob.pop(Cache.X)
    generator = torch.Generator().manual_seed(5)
    rows = len(blob["query_positions"])
    blob[Cache.KEY_AND_QUERY_FIELDS[0]] = torch.randn(
        fake_cache.LENGTH, KEY_DIM, generator=generator
    ).bfloat16()
    blob[Cache.HEAD_QUERIES] = torch.randn(rows, HEADS, KEY_DIM, generator=generator).bfloat16()
    blob[Cache.KEY_AND_QUERY_FIELDS[1]] = blob[Cache.HEAD_QUERIES].float().sum(1)
    blob["meta"][Cache.ATTENTION_SCALE] = KEY_DIM**-0.5
    blob["meta"].update(meta)
    return Document.from_blob(
        layer_path(tmp_path, "tokens", "train", "source0", 0), blob, fake_cache.config(input=KEY_AND_QUERY)
    )


def test_a_document_computes_the_target_of_its_rows_from_its_key_and_head_queries(tmp_path):
    one = full_document(tmp_path)
    rows = one.eval_rows()
    target = one.key_target(rows, target=Target.DENSE_FULL)
    expected = full_attention_target(
        one.head_queries[rows],
        one.key,
        one.attention_scale,
        one.query_positions[rows],
        fake_cache.SINK,
        fake_cache.WINDOW,
    )
    assert torch.equal(target, expected)
    assert torch.allclose(target.sum(-1), torch.ones(len(rows)), atol=1e-5)
    # Zero off what a row may read, as the cached targets are
    assert not bool((target[~one.visible_keys(rows)] != 0).any())
    # And mass on every one of them - the support is the visible set the cached targets and the loss read
    assert bool(((target > 0) == one.visible_keys(rows)).all())


def test_a_depth_on_the_full_target_is_refused_at_the_document(tmp_path):
    one = full_document(tmp_path)
    with pytest.raises(ValueError, match="not truncated - got depth 16"):
        one.key_target(one.eval_rows(), 16, Target.DENSE_FULL)


def test_a_cache_without_the_key_or_the_head_queries_is_refused_naming_what_is_missing(tmp_path):
    fake_cache.collection(tmp_path)
    cached = Document(layer_path(tmp_path, "tokens", "train", "source0", 0), fake_cache.config())
    with pytest.raises(
        ValueError, match=r"holds no \['key', 'head_queries'\].*written under the key and query"
    ):
        cached.key_target(cached.eval_rows(), target=Target.DENSE_FULL)


def test_a_cache_that_recorded_no_scale_takes_the_models_own(tmp_path, monkeypatch):
    from lsa.training.distillation.data import cache

    one = full_document(tmp_path)
    rows = one.eval_rows()
    recorded = one.key_target(rows, target=Target.DENSE_FULL)
    one.attention_scale = None
    asked = []
    monkeypatch.setattr(cache, "latent_attention_scale", lambda model: asked.append(model) or KEY_DIM**-0.5)
    assert torch.equal(one.key_target(rows, target=Target.DENSE_FULL), recorded)
    assert asked == [fake_cache.MODEL]


class Text:
    def __init__(self, nope, rope):
        self.qk_nope_head_dim, self.qk_rope_head_dim = nope, rope


def configured(monkeypatch, text):
    from lsa.evaluation import adapter

    latent_attention_scale.cache_clear()
    monkeypatch.setattr(adapter, "adapter_for", lambda config: SimpleNamespace(text_config=lambda c: text))


@pytest.fixture(autouse=True)
def model_that_does_not_rotate(monkeypatch):
    # Every document's model resolves to a head of KEY_DIM nope dims
    configured(monkeypatch, Text(KEY_DIM, 0))
    yield
    latent_attention_scale.cache_clear()


def test_a_recorded_scale_does_not_excuse_a_model_whose_heads_rotate(tmp_path, monkeypatch):
    one = full_document(tmp_path)
    configured(monkeypatch, Text(128, 64))
    with pytest.raises(ValueError, match="64 rope dims"):
        one.key_target(one.eval_rows(), target=Target.DENSE_FULL)


def test_the_models_scale_is_its_head_dim_to_the_minus_half(monkeypatch):
    configured(monkeypatch, Text(256, 0))
    assert latent_attention_scale("glm53_flash") == 1 / 16
    latent_attention_scale.cache_clear()


def test_a_model_whose_heads_rotate_has_no_latent_scale(monkeypatch):
    configured(monkeypatch, Text(128, 64))
    with pytest.raises(ValueError, match="64 rope dims.*does not rotate"):
        latent_attention_scale("glm53_flash")
    latent_attention_scale.cache_clear()


def test_a_document_read_for_a_qsa_indexer_has_no_full_target(tmp_path):
    one = full_document(tmp_path)
    one.block_size = 4
    with pytest.raises(ValueError, match="QSA indexer"):
        one.key_target(one.eval_rows(), target=Target.DENSE_FULL)


@torch.no_grad()
def test_the_loss_and_the_entropy_read_the_full_target_and_stay_above_it(tmp_path):
    one = full_document(tmp_path)
    rows = one.train_rows()
    index_map = IndexMap(KEY_DIM, 8, init=init_rows(one.key_inputs()))
    index_map.set_tau(0.05)
    objective = ObjectiveConfig(target=Target.DENSE_FULL)
    loss = float(step_terms(index_map, one, rows, objective)[Term.KEY])
    entropy = float(teacher_entropy([one], target=Target.DENSE_FULL))
    assert entropy > 0
    assert loss >= entropy - 1e-4
    scores = key_scores(index_map, one, rows) / index_map.log_tau.exp()
    assert float(key_divergence(scores, one, rows, target=Target.DENSE_FULL)) == pytest.approx(loss)
