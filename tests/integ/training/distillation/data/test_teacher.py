"""What the extractor samples and what it counts as readable.

The extraction itself needs a checkpoint; the rules it applies do not, and those are what decide whether
every number downstream means what it says.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lsa.training.distillation.constants import Split, Teacher
from lsa.training.distillation.data.documents import Selected
from lsa.training.distillation.data.teacher import locate_claim
from lsa.training.distillation.data.teacher_attention import readable_keys
from lsa.training.distillation.data.teacher_capture import NativeSelection, hook_native_selection
from lsa.training.distillation.data.teacher_config import (
    DataConfig,
    QueryConfig,
    RunConfig,
    TeacherConfig,
    TeacherModelConfig,
    TokenizedConfig,
)
from lsa.training.distillation.data.teacher_plan import (
    extracted_now,
    flatten_plan,
    query_positions,
    read_documents,
    training_query_count,
)

LENGTH = 40000


def a_run(**overrides):
    model = {key: overrides.pop(key) for key in ("indexer",) if key in overrides}
    data = {key: overrides.pop(key) for key in ("tokenized", "prepared", "queries") if key in overrides}
    return TeacherConfig(
        model=TeacherModelConfig(**model),
        data=DataConfig(**data),
        run=RunConfig(output="targets", **overrides),
    )


def test_training_positions_all_sit_past_the_sinks_and_the_window():
    one = a_run()
    positions = query_positions(one, LENGTH, 0)
    training = positions[: one.data.queries.train]
    assert int(training.min()) >= one.model.indexer.sink_size + one.model.indexer.window_size


def test_the_tail_is_the_end_of_the_document_and_comes_last():
    one = a_run()
    positions = query_positions(one, LENGTH, 0)
    assert positions[-one.data.queries.tail :].tolist() == list(range(LENGTH - one.data.queries.tail, LENGTH))


def test_training_positions_are_sorted_and_never_repeat():
    one = a_run()
    training = query_positions(one, LENGTH, 0)[: one.data.queries.train]
    assert torch.equal(training, training.sort().values)
    assert len(set(training.tolist())) == len(training)


def test_two_documents_are_sampled_differently_and_one_document_reproducibly():
    one = a_run()
    assert not torch.equal(query_positions(one, LENGTH, 0), query_positions(one, LENGTH, 1))
    assert torch.equal(query_positions(one, LENGTH, 3), query_positions(one, LENGTH, 3))


def test_a_fixed_count_is_the_same_at_every_length():
    one = a_run()
    assert training_query_count(one, 16384) == training_query_count(one, 1048576) == one.data.queries.train


def test_density_makes_the_count_follow_the_length_and_never_drop_below_the_floor():
    one = a_run(queries=QueryConfig(train=128, density=0.005))
    first = one.model.indexer.sink_size + one.model.indexer.window_size
    span = 1048576 - one.data.queries.tail - first

    assert training_query_count(one, 1048576) == round(0.005 * span)
    assert training_query_count(one, 16384) == 128
    assert len(query_positions(one, 1048576, 0)) == round(0.005 * span) + one.data.queries.tail


def test_a_document_with_no_room_for_its_queries_says_so_rather_than_sampling_fewer():
    one = a_run()
    with pytest.raises(ValueError, match="fewer than 512 training queries"):
        training_query_count(one, 1024)


def test_a_selection_file_gives_each_dataset_its_own_entries(tmp_path):
    entries = [
        {"dataset": "glm53_flash", "document": "a"},
        {"dataset": "edgebench_trajectories", "document": "b"},
    ]
    path = tmp_path / "documents.json"
    path.write_text(json.dumps({"documents": entries}))

    assert read_documents(path, "edgebench_trajectories") == [entries[1]]
    with pytest.raises(ValueError, match="names no document of qwen3_5"):
        read_documents(path, "qwen3_5")


def test_extracting_one_bucket_keeps_each_window_where_it_sits_in_the_whole_selection():
    """A window's queries are seeded by its place in the plan, so a bucket read alone caches the same bytes"""
    selection = [
        Selected(
            source="arxiv",
            document=f"d{index}",
            window=0,
            split=Split.TRAIN,
            name=f"d{index}",
            start=0,
            length=length,
        )
        for index, length in enumerate((16384, 32768, 16384))
    ]
    one = a_run(tokenized=TokenizedConfig(documents="documents.json", lengths=[16384]))

    kept = extracted_now(one, flatten_plan([(None, selection)]))

    assert [document.index for document in kept] == [1, 3]


def test_extracting_one_side_keeps_each_window_where_it_sits_in_the_whole_selection():
    """A training side re-extracted alone draws the rows the whole extraction drew"""
    selection = [
        Selected(
            source="arxiv",
            document=f"d{index}",
            window=0,
            split=split,
            name=f"d{index}",
            start=0,
            length=16384,
        )
        for index, split in enumerate((Split.EVAL, Split.TRAIN, Split.EVAL, Split.TRAIN))
    ]
    one = a_run(tokenized=TokenizedConfig(documents="documents.json", splits=[Split.TRAIN]))

    kept = extracted_now(one, flatten_plan([(None, selection)]))

    assert [document.index for document in kept] == [2, 4]


def test_a_side_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="tokenized.splits must name sides"):
        TokenizedConfig(documents="documents.json", splits=["validation"])


def test_the_teacher_scores_the_same_however_few_queries_a_block_holds(monkeypatch):
    """A block narrows as a document grows, and a query's row never depends on the rows beside it"""
    from lsa.training.distillation.data import teacher_attention

    one = a_run()
    torch.manual_seed(0)
    length = one.model.indexer.sink_size + 3 * one.model.indexer.window_size
    q, k = torch.randn(8, 40, 16), torch.randn(2, length, 16)
    positions = query_positions(one, length, 0)[-40:]

    whole = teacher_attention.teacher_attention(q, k, 0.25, positions, one.model.indexer, 16)
    monkeypatch.setattr(Teacher, "SCORE_CELLS", 8 * 3 * length)
    blocked = teacher_attention.teacher_attention(q, k, 0.25, positions, one.model.indexer, 16)

    for together, apart in zip(whole, blocked, strict=True):
        assert torch.equal(together, apart)


def test_a_negative_prefill_is_refused():
    with pytest.raises(ValueError, match="prefill must be nonnegative"):
        a_run(prefill=-1)


def test_lengths_without_a_selection_are_refused():
    with pytest.raises(ValueError, match="selection|documents"):
        a_run(tokenized=TokenizedConfig(lengths=[16384]))


def test_a_readable_key_is_causal_past_the_sinks_and_outside_the_window():
    one = a_run()
    positions = torch.tensor([one.model.indexer.sink_size + 3 * one.model.indexer.window_size + 7])
    readable = readable_keys(one.model.indexer, positions, LENGTH)[0]
    assert not bool(readable[: one.model.indexer.sink_size].any())
    assert not bool(readable[int(positions[0]) - one.model.indexer.window_size + 1 :].any())
    assert bool(readable[one.model.indexer.sink_size])


def test_the_last_readable_key_is_a_whole_window_behind_the_query():
    one = a_run()
    position = one.model.indexer.sink_size + 3 * one.model.indexer.window_size + 7
    readable = readable_keys(one.model.indexer, torch.tensor([position]), LENGTH)[0]
    assert bool(readable[position - one.model.indexer.window_size])
    assert not bool(readable[position - one.model.indexer.window_size + 1])
    assert int(readable.sum()) == position - one.model.indexer.window_size + 1 - one.model.indexer.sink_size


def test_a_query_within_the_sinks_and_the_window_can_read_nothing():
    one = a_run()
    position = one.model.indexer.sink_size + one.model.indexer.window_size - 1
    readable = readable_keys(one.model.indexer, torch.tensor([position]), LENGTH)[0]
    assert not bool(readable.any())


def test_an_extraction_must_cover_a_layer():
    with pytest.raises(ValueError, match="at least one layer"):
        a_run(layers=[])


def a_claim(tmp_path, name="arxiv_r0_w0"):
    return locate_claim(tmp_path, "glm53_flash", Split.TRAIN, name)


def test_one_process_takes_a_document_and_the_next_leaves_it_alone():
    """Several processes work one cache, so two must never read the same document at once"""
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        first = a_claim(Path(root))
        second = a_claim(Path(root))

        assert first.take()
        assert not second.take()


def test_a_released_claim_is_free_again():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        first = a_claim(Path(root))
        first.take()
        first.release()

        assert a_claim(Path(root)).take()


def test_a_claim_nobody_refreshed_is_taken_over():
    """What makes a job killable - whatever it was holding goes back to the survivors"""
    import os
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as root:
        killed = a_claim(Path(root))
        killed.take()
        old = time.time() - Teacher.CLAIM_STALE - 60
        os.utime(killed.path, (old, old))

        assert a_claim(Path(root)).take()


def test_a_live_claim_is_not_taken_over():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        live = a_claim(Path(root))
        live.take()

        assert not a_claim(Path(root)).take()


def a_plan(sources=("arxiv", "code"), each=3):
    """A plan shaped like `plan` returns one, without a dataset that needs a checkpoint to build"""
    selection = [
        Selected(
            source=source,
            document=f"doc-{source}-{index}",
            window=0,
            split=Split.TRAIN,
            name=f"{source}_r{index}_w0",
            start=index,
        )
        for source in sources
        for index in range(each)
    ]
    return [(None, selection)]


class Picker(torch.nn.Module):
    """A selector that picks, for each position, the keys `position` and `position - 1`"""

    def forward(self, positions):
        return torch.stack([positions, positions - 1], dim=-1).int()


class Stack:
    """A decoder stack whose layers each carry one selector"""

    def __init__(self, layer_count):
        self.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(indexer=Picker())) for _ in range(layer_count)
        ]


class SparseAdapter:
    def native_selector(self, attention):
        return attention.indexer


class DenseAdapter:
    def native_selector(self, attention):
        return None


def run_in_passes(stack, layers, length, step):
    for start in range(0, length, step):
        chunk = torch.arange(start, min(start + step, length))[None]
        for index in layers:
            stack.layers[index].self_attn.indexer(chunk)


def test_the_native_selection_is_kept_at_the_sampled_positions_across_passes():
    stack, positions = Stack(3), torch.tensor([5, 17, 18, 39])
    selection, hooks = hook_native_selection(SparseAdapter(), stack, [0, 2], positions)
    run_in_passes(stack, [0, 1, 2], 40, 16)
    for hook in hooks:
        hook.remove()
    expected = torch.stack([positions, positions - 1], dim=-1).int()
    assert torch.equal(selection.of(0, 40), expected)
    assert torch.equal(selection.of(2, 40), expected)
    assert 1 not in selection.picked


def test_a_layer_whose_selector_did_not_see_the_whole_document_is_refused():
    stack = Stack(1)
    selection, _ = hook_native_selection(SparseAdapter(), stack, [0], torch.tensor([3]))
    run_in_passes(stack, [0], 16, 16)
    with pytest.raises(ValueError, match="16 of 40 positions"):
        selection.of(0, 40)


def test_a_dense_layer_has_nothing_native_to_keep():
    with pytest.raises(ValueError, match="no sparse selector"):
        hook_native_selection(DenseAdapter(), Stack(1), [0], torch.tensor([3]))


def test_a_selector_that_changes_its_width_between_passes_is_refused():
    selection = NativeSelection(torch.tensor([1, 9]))
    selection.record(0, torch.zeros(1, 8, 4, dtype=torch.int32))
    with pytest.raises(ValueError, match="selected 2 keys a query where it had selected 4"):
        selection.record(0, torch.zeros(1, 8, 2, dtype=torch.int32))


def test_the_served_attention_is_each_heads_softmax_over_its_picks_summed():
    from lsa.training.distillation.data import teacher_attention

    torch.manual_seed(0)
    q, k = torch.randn(4, 3, 8), torch.randn(2, 50, 8)
    picked = torch.tensor([[1, 7, 30, -1], [0, 2, 4, 6], [49, -1, -1, 3]], dtype=torch.int32)

    served = teacher_attention.native_attention(q, k, 0.5, picked).float()

    for row in range(3):
        keys = picked[row][picked[row] >= 0].long()
        expected = sum(torch.softmax(q[head, row] @ k[head // 2, keys].t() * 0.5, -1) for head in range(4))
        assert torch.allclose(served[row][picked[row] >= 0], expected, atol=2e-3)
        assert bool((served[row][picked[row] < 0] == 0).all())


def test_the_served_attention_is_the_same_however_few_queries_a_block_holds(monkeypatch):
    from lsa.training.distillation.data import teacher_attention

    torch.manual_seed(0)
    q, k = torch.randn(4, 9, 8), torch.randn(2, 50, 8)
    picked = torch.randint(-1, 50, (9, 5), dtype=torch.int32)
    whole = teacher_attention.native_attention(q, k, 0.5, picked)
    monkeypatch.setattr(Teacher, "SCORE_CELLS", 2 * 5 * 8 * 2)
    assert torch.equal(teacher_attention.native_attention(q, k, 0.5, picked), whole)
