"""What a cached document holds, where it waits, and what a run refuses to read."""

import pytest
import torch

from lsa.training.data.targets import extracted_documents
from lsa.training.distillation.config import DocumentSet
from lsa.training.distillation.constants import Cache, Target
from lsa.training.distillation.data.cache import (
    Document,
    cached_layers,
    documents,
    layer_path,
    write,
)
from lsa.training.distillation.data.pool import stream
from tests.helpers.training.distillation.data import fake_cache


def one(tmp_path, **overrides):
    fake_cache.collection(tmp_path)
    return Document(layer_path(tmp_path, "tokens", "train", "source0", 0), fake_cache.config(**overrides))


def test_a_document_reads_back_x_and_the_queries_come_from_it(tmp_path):
    document = one(tmp_path)
    assert document.x.dtype is torch.bfloat16
    assert document.x.shape == (fake_cache.LENGTH, fake_cache.MODEL_DIM)
    rows = document.eval_rows()
    assert torch.equal(document.query_inputs(rows), document.x[document.query_positions[rows]])


def test_a_document_reads_back_the_shapes_it_was_written_with(tmp_path):
    document = one(tmp_path)
    assert document.key_count == fake_cache.LENGTH
    assert document.model_dim == fake_cache.MODEL_DIM
    assert document.head_count == fake_cache.HEADS
    assert document.train_query_count == fake_cache.TRAIN_QUERIES
    assert len(document.eval_rows()) == fake_cache.TAIL_QUERIES


def test_training_rows_and_the_tail_never_overlap(tmp_path):
    document = one(tmp_path)
    assert set(document.train_rows().tolist()).isdisjoint(document.eval_rows().tolist())


def test_a_key_is_readable_only_past_the_sinks_and_outside_the_window(tmp_path):
    document = one(tmp_path)
    rows = document.train_rows()
    readable = document.visible_keys(rows)
    positions = document.query_positions[rows]
    keys = torch.arange(document.key_count)
    assert not bool(readable[:, : fake_cache.SINK].any())
    assert not bool(readable[positions[:, None] - keys[None] < fake_cache.WINDOW].any())
    # The edges: the first key past the sinks, and the last one a whole window behind
    assert bool(readable[0, fake_cache.SINK])
    assert bool(readable[0, positions[0] - fake_cache.WINDOW])
    assert not bool(readable[0, positions[0] - fake_cache.WINDOW + 1])


def test_the_teacher_target_is_a_distribution_over_what_is_readable(tmp_path):
    document = one(tmp_path)
    rows = document.eval_rows()
    target = document.key_target(rows)
    assert torch.allclose(target.sum(-1), torch.ones(len(rows)), atol=1e-5)
    assert not bool((target > 0)[~document.visible_keys(rows)].any())


def test_a_cache_written_under_another_visibility_rule_is_refused(tmp_path):
    fake_cache.collection(tmp_path)
    with pytest.raises(ValueError, match="different visibility rule"):
        Document(layer_path(tmp_path, "tokens", "train", "source0", 0), fake_cache.config(window_size=64))


def test_a_cache_whose_cluster_block_is_not_the_window_is_refused(tmp_path):
    blob = fake_cache.document()
    blob["meta"]["cluster_block_size"] = 8192
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    with pytest.raises(ValueError, match="cluster block"):
        Document(path, fake_cache.config())


def test_a_file_missing_a_field_says_which(tmp_path):
    blob = fake_cache.document()
    blob.pop("peak_key_id")
    path = layer_path(tmp_path, "tokens", "train", "short0", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    with pytest.raises(ValueError, match="peak_key_id"):
        Document(path, fake_cache.config())


def test_a_set_reads_one_side_and_cannot_see_the_other(tmp_path):
    """The whole point of the split being a directory - no spelling of a set reaches across it"""
    fake_cache.collection(tmp_path, documents=("arxiv0", "code0"), split="train")
    fake_cache.collection(tmp_path, documents=("arxiv1",), split="eval", seed=9)

    assert documents(tmp_path, DocumentSet(source="tokens", split="train")) == ["arxiv0", "code0"]
    assert documents(tmp_path, DocumentSet(source="tokens", split="eval")) == ["arxiv1"]


def test_sources_keep_one_domain_out_of_a_side(tmp_path):
    """What makes a per-source report expressible, which a trailing index never could"""
    fake_cache.collection(tmp_path, documents=("arxiv0", "arxiv1", "code0"))

    selected = documents(tmp_path, DocumentSet(source="tokens", split="train", sources=["arxiv"]))

    assert selected == ["arxiv0", "arxiv1"]


def test_streaming_a_set_ahead_in_threads_reads_the_same_documents_in_order(tmp_path):
    fake_cache.collection(tmp_path, documents=("arxiv0", "arxiv1", "code0", "code1"))
    paths = [
        layer_path(tmp_path, "tokens", "train", name, 0)
        for name in documents(tmp_path, DocumentSet(source="tokens"))
    ]

    one_at_a_time = list(stream(paths, fake_cache.config(), ahead=1))
    ahead = list(stream(paths, fake_cache.config(), ahead=3))

    assert [document.name for document in ahead] == ["arxiv0", "arxiv1", "code0", "code1"]
    for threaded, serial in zip(ahead, one_at_a_time, strict=True):
        assert torch.equal(threaded.x, serial.x)


def test_a_set_with_its_own_root_reads_that_cache_and_not_the_runs(tmp_path):
    """What lets a run score RULER prompts extracted into a cache apart from the one it trains on"""
    run_root, ruler_root = tmp_path / "selection", tmp_path / "ruler"
    fake_cache.collection(run_root, name="ruler", documents=("natural0",), split="eval")
    fake_cache.collection(
        ruler_root, name="ruler", documents=("cwe0", "vt0"), layers=(0, 3), split="eval", seed=9
    )
    selection = DocumentSet(source="ruler", split="eval", root=str(ruler_root))

    assert documents(run_root, selection) == ["cwe0", "vt0"]
    assert cached_layers(run_root, selection) == [0, 3]


def test_lengths_keep_one_bucket_of_a_selection(tmp_path):
    """What makes a per-length report expressible over a dataset"""
    import json

    fake_cache.collection(tmp_path, documents=("arxiv0", "arxiv1", "code0"))
    manifest = tmp_path / "tokens" / "manifest.json"
    payload = json.loads(manifest.read_text())
    for entry, length in zip(payload["documents"], (16384, 1048576, 16384), strict=True):
        entry["length"] = length
    manifest.write_text(json.dumps(payload))

    selected = documents(tmp_path, DocumentSet(source="tokens", split="train", lengths=[16384]))

    assert selected == ["arxiv0", "code0"]


def test_every_extracted_document_is_named_as_the_corpus_names_it(tmp_path):
    """Both sides of the cache, because a validation set has to leave out every document a map was fit near"""
    fake_cache.collection(tmp_path, documents=("arxiv0", "code0"), split="train")
    fake_cache.collection(tmp_path, documents=("arxiv1",), split="eval", seed=9)

    assert extracted_documents(tmp_path, "tokens") == {
        ("arxiv", "doc-arxiv0"),
        ("code", "doc-code0"),
        ("arxiv", "doc-arxiv1"),
    }


def test_a_collection_with_no_manifest_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no extraction manifest"):
        documents(tmp_path, DocumentSet(source="absent"))


def test_a_set_that_selects_nothing_says_so_rather_than_training_on_nothing(tmp_path):
    fake_cache.collection(tmp_path, documents=("arxiv0",))

    with pytest.raises(ValueError, match="selects no document"):
        documents(tmp_path, DocumentSet(name="gone", source="tokens", split="eval"))


def test_a_document_moves_without_moving_the_one_it_came_from(tmp_path):
    """A staged document is a copy, so the one waiting behind it is still readable where it waits"""
    document = one(tmp_path)
    moved = document.to("cpu")
    assert moved is not document
    assert moved.device == torch.device("cpu")
    assert torch.equal(moved.x, document.x)
    assert moved.train_query_count == document.train_query_count
    assert moved.name == document.name and moved.layer == document.layer


def test_a_documents_resident_bytes_counts_every_tensor_it_holds(tmp_path):
    document = one(tmp_path)
    assert document.resident_bytes > document.x.numel() * document.x.element_size()


def test_a_cache_without_a_native_selection_reads_as_none(tmp_path):
    assert one(tmp_path).native_key_ids is None


def test_a_native_selection_reads_back_and_moves_with_its_document(tmp_path):
    blob = fake_cache.document()
    picked = torch.randint(-1, 100, (len(blob["query_positions"]), 6), dtype=torch.int32)
    blob[Cache.NATIVE] = picked
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    document = Document(path, fake_cache.config())
    assert torch.equal(document.native_key_ids, picked)
    assert torch.equal(document.to("cpu").native_key_ids, picked)


def native(tmp_path, picked_of):
    blob = fake_cache.document()
    blob[Cache.NATIVE] = picked_of(blob).int()
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    return Document(path, fake_cache.config())


def test_a_dense_on_native_target_keeps_the_picked_keys_in_their_dense_proportion(tmp_path):
    # Half of each row's teacher keys, then empty slots
    half = fake_cache.TOP // 2
    document = native(
        tmp_path,
        lambda blob: torch.cat(
            [blob["top_key_ids"][:, :half], torch.full((len(blob["top_key_ids"]), 3), -1)], 1
        ),
    )
    rows = document.eval_rows()
    dense, restricted = document.key_target(rows), document.key_target(rows, target=Target.DENSE_ON_NATIVE)
    picked = document.native_members(rows)
    assert torch.allclose(restricted.sum(-1), torch.ones(len(rows)), atol=1e-5)
    assert not bool((restricted > 0)[~picked].any())
    kept = dense * picked
    assert torch.allclose(restricted, kept / kept.sum(-1, keepdim=True), atol=1e-5)


def test_a_dense_on_native_target_over_every_teacher_key_is_the_dense_one(tmp_path):
    document = native(tmp_path, lambda blob: blob["top_key_ids"])
    rows = document.train_rows()
    assert torch.allclose(
        document.key_target(rows, target=Target.DENSE_ON_NATIVE), document.key_target(rows), atol=1e-6
    )


def test_a_native_target_without_a_native_selection_says_how_to_get_one(tmp_path):
    with pytest.raises(ValueError, match="queries.native"):
        one(tmp_path).key_target(torch.arange(2), target=Target.DENSE_ON_NATIVE)


def with_native_mass(tmp_path, picked_of, mass_of):
    blob = fake_cache.document()
    blob[Cache.NATIVE] = picked_of(blob).int()
    blob[Cache.NATIVE_MASS] = mass_of(blob[Cache.NATIVE]).half()
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    return Document(path, fake_cache.config())


def test_a_selection_target_weighs_every_readable_pick_equally(tmp_path):
    document = native(
        tmp_path,
        lambda blob: torch.cat(
            [blob["top_key_ids"][:, :5], torch.full((len(blob["top_key_ids"]), 2), -1)], 1
        ),
    )
    rows = document.eval_rows()
    target = document.key_target(rows, target=Target.NATIVE_SELECTION)
    picked = document.native_members(rows) & document.visible_keys(rows)
    assert torch.allclose(target, picked.float() / picked.sum(-1, keepdim=True), atol=1e-6)


def test_a_served_attention_target_is_the_cached_mass_on_the_picks(tmp_path):
    length = fake_cache.LENGTH
    document = with_native_mass(
        tmp_path,
        lambda blob: torch.randint(
            fake_cache.SINK, fake_cache.SINK + fake_cache.WINDOW, (len(blob["query_positions"]), 6)
        ),
        lambda picked: torch.arange(1, 7).float().expand(len(picked), 6),
    )
    rows = document.eval_rows()
    target = document.key_target(rows, target=Target.NATIVE_ATTENTION)
    expected = torch.zeros(len(rows), length)
    expected.scatter_add_(1, document.native_key_ids[rows].long(), document.native_key_mass[rows].float())
    expected = expected * document.visible_keys(rows)
    assert torch.allclose(target, expected / expected.sum(-1, keepdim=True), atol=1e-4)


@pytest.mark.parametrize("target", [Target.NATIVE_SELECTION, Target.NATIVE_ATTENTION])
def test_a_native_target_carries_nothing_in_the_window(tmp_path, target):
    blob = fake_cache.document()
    # Every query at one position, so the same keys sit inside its window
    position = fake_cache.SINK + 2 * fake_cache.WINDOW + 3
    blob["query_positions"] = torch.full_like(blob["query_positions"], position)
    in_window = torch.tensor([position - 4, position - 10, position - 25])
    far = torch.tensor([fake_cache.SINK, fake_cache.SINK + 1, fake_cache.SINK + 2])
    picked = torch.cat([in_window, far]).expand(len(blob["query_positions"]), 6)
    blob[Cache.NATIVE] = picked.int()
    blob[Cache.NATIVE_MASS] = torch.ones(picked.shape).half()
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    document = Document(path, fake_cache.config())
    rows = document.eval_rows()
    target = document.key_target(rows, target=target)
    assert not bool(document.visible_keys(rows)[:, in_window].any())
    assert not bool(target[:, in_window].any())
    assert torch.allclose(target[:, far], torch.full((len(rows), 3), 1 / 3), atol=1e-6)


def test_a_served_attention_target_without_its_mass_says_how_to_get_it(tmp_path):
    document = native(tmp_path, lambda blob: blob["top_key_ids"])
    with pytest.raises(ValueError, match="no native attention"):
        document.key_target(document.eval_rows(), target=Target.NATIVE_ATTENTION)


def test_a_target_the_cache_cannot_build_is_refused(tmp_path):
    with pytest.raises(ValueError, match="target must be one of"):
        one(tmp_path).key_target(torch.arange(2), target="dsa")


CLOSED_BLOCK = 128


def test_a_closed_block_hides_every_key_in_a_block_still_open_at_the_query(tmp_path):
    document = one(tmp_path).with_closed_block(CLOSED_BLOCK)
    rows = document.eval_rows()
    readable = document.visible_keys(rows)
    position = int(document.query_positions[rows[0]])
    # The last block that closed before the query's own ends here
    closed_to = fake_cache.SINK + ((position - fake_cache.SINK) // CLOSED_BLOCK) * CLOSED_BLOCK
    assert not bool(readable[0, : fake_cache.SINK].any())
    assert bool(readable[0, fake_cache.SINK : closed_to].all())
    assert not bool(readable[0, closed_to:].any())


def test_a_closed_block_trains_only_the_rows_with_a_block_closed_behind_them(tmp_path):
    plain = one(tmp_path)
    document = plain.with_closed_block(CLOSED_BLOCK)
    kept = document.query_positions[document.train_rows()]
    every = plain.query_positions[plain.train_rows()]
    assert bool((kept >= fake_cache.SINK + CLOSED_BLOCK).all())
    assert len(kept) == int((every >= fake_cache.SINK + CLOSED_BLOCK).sum())
    assert plain.with_closed_block(None).closed_block is None


def test_a_closed_block_leaves_a_target_only_on_the_keys_it_reads(tmp_path):
    document = one(tmp_path).with_closed_block(CLOSED_BLOCK)
    rows = document.eval_rows()
    target = document.key_target(rows)
    assert not bool((target > 0)[~document.visible_keys(rows)].any())
    assert torch.allclose(target.sum(-1), torch.ones(len(rows)), atol=1e-5)


def test_a_closed_block_decides_which_cached_keys_a_row_may_select(tmp_path):
    document = one(tmp_path).with_closed_block(CLOSED_BLOCK)
    rows = document.eval_rows()
    selectable = document.selectable(document.top_key_ids[rows].long(), rows)
    readable = document.visible_keys(rows)
    assert torch.equal(selectable, readable.gather(1, document.top_key_ids[rows].long()))


def test_a_closed_block_larger_than_the_document_hides_every_key_and_row(tmp_path):
    document = one(tmp_path).with_closed_block(fake_cache.LENGTH)
    assert not bool(document.visible_keys(document.eval_rows()).any())
    assert len(document.train_rows()) == 0


def index_vector_document(tmp_path, **meta):
    """A document in the index-vector cache's format - index vectors in place of `x`, a coarse block, window and sinks"""
    blob = fake_cache.document()
    del blob[Cache.X]
    blob["index_keys"] = torch.nn.functional.normalize(torch.randn(fake_cache.LENGTH, 4), dim=-1).half()
    blob["index_queries"] = torch.nn.functional.normalize(
        torch.randn(len(blob["query_positions"]), 4), dim=-1
    ).half()
    blob[Cache.META] = {
        "document": "source0",
        "layer": 0,
        "block": CLOSED_BLOCK,
        "window": fake_cache.WINDOW,
        "sink": fake_cache.SINK,
        "length": fake_cache.LENGTH,
        "train_queries": fake_cache.TRAIN_QUERIES,
        **meta,
    }
    path = layer_path(tmp_path, "index_vectors", "train", "source0", 0)
    write(path, blob)
    return path


def test_an_index_vector_cache_carries_the_closed_block_rule_in_its_file(tmp_path):
    document = Document(index_vector_document(tmp_path), fake_cache.config())
    assert document.closed_block == CLOSED_BLOCK
    assert document.x is None and document.index_keys.shape == (fake_cache.LENGTH, 4)
    readable = document.visible_keys(document.eval_rows())
    assert not bool(readable[:, : fake_cache.SINK].any())
    assert bool(readable.any())


def test_an_index_vector_cache_written_under_another_window_or_sinks_is_refused(tmp_path):
    with pytest.raises(ValueError, match="different visibility rule - window 64 != 32"):
        Document(index_vector_document(tmp_path, window=64), fake_cache.config())
    with pytest.raises(ValueError, match="different visibility rule - sink 8 != 4"):
        Document(index_vector_document(tmp_path, sink=8), fake_cache.config())


def test_an_index_vector_document_is_read_on_the_key_and_query_or_refused(tmp_path):
    document = Document(index_vector_document(tmp_path), fake_cache.config())
    with pytest.raises(ValueError, match="index vectors - read it on indexer.input: key_and_query"):
        document.key_inputs()


def test_a_closed_block_of_another_size_than_the_cache_was_written_under_is_refused(tmp_path):
    document = Document(index_vector_document(tmp_path), fake_cache.config())
    assert document.with_closed_block(CLOSED_BLOCK).closed_block == CLOSED_BLOCK
    with pytest.raises(
        ValueError, match="written under a closed block of 128 keys - objective.closed_block asks for 256"
    ):
        document.with_closed_block(2 * CLOSED_BLOCK)
