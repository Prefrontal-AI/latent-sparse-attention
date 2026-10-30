"""The rolling pool - what a run draws, what it reads, and that a step gets the document it was dealt."""

import collections
import threading

import pytest
import torch

from lsa.training.distillation.config import DocumentSet
from lsa.training.distillation.constants import Draw
from lsa.training.distillation.data.cache import Document, layer_path
from lsa.training.distillation.data.pool import (
    Entry,
    Interrupted,
    Item,
    Rolling,
    device_budget,
    drawn_documents,
    interleave,
    items_of,
    layer_steps,
    remaining,
)
from tests.helpers.training.distillation.data import fake_cache

NAMES = ("a0", "a1", "b0", "b1", "c0", "c1")


def entries(root, tokens=(10, 10, 10, 10, 10, 10)):
    return [Entry((root, "tokens", "train"), name, count) for name, count in zip(NAMES, tokens, strict=True)]


def written(tmp_path, layers=(0,)):
    fake_cache.collection(tmp_path, documents=NAMES, layers=layers)
    return entries(tmp_path)


def test_a_token_draw_takes_documents_in_proportion_to_their_tokens():
    catalog = [Entry((None, "t", "train"), str(index), tokens) for index, tokens in enumerate((1, 1, 6))]

    drawn = drawn_documents(catalog, 20000, 0, Draw.TOKEN)

    shares = [drawn.count(index) / len(drawn) for index in range(3)]
    assert shares == pytest.approx([1 / 8, 1 / 8, 6 / 8], abs=0.02)


def test_a_document_draw_takes_every_document_equally():
    catalog = [Entry((None, "t", "train"), str(index), tokens) for index, tokens in enumerate((1, 1, 6))]

    drawn = drawn_documents(catalog, 20000, 0, Draw.DOCUMENT)

    assert [drawn.count(index) / len(drawn) for index in range(3)] == pytest.approx([1 / 3] * 3, abs=0.02)


def test_the_draw_is_the_seeds_alone():
    catalog = [Entry((None, "t", "train"), str(index), index + 1) for index in range(5)]
    assert drawn_documents(catalog, 50, 3, Draw.TOKEN) == drawn_documents(catalog, 50, 3, Draw.TOKEN)
    assert drawn_documents(catalog, 50, 3, Draw.TOKEN) != drawn_documents(catalog, 50, 4, Draw.TOKEN)


def test_steps_rotate_over_the_layers_of_a_run():
    dealt = layer_steps([3, 7, 11], 7)

    assert dealt == {3: [3, 6], 7: [1, 4, 7], 11: [2, 5]}


def test_a_shard_draws_for_its_layers_what_the_whole_run_does(tmp_path):
    """Each layer's documents come off the run's one draw, so a layer trains the same whether or not its neighbours do"""
    catalog = written(tmp_path, layers=(0, 1, 2))
    drawn = drawn_documents(catalog, 30, 0, Draw.TOKEN)
    dealt = layer_steps([0, 1, 2], 30)

    whole = {layer: items_of(catalog, drawn, layer, dealt[layer], 0) for layer in (0, 1, 2)}
    alone = items_of(catalog, drawn, 1, dealt[1], 0)

    assert [(item.entry.name, item.uses) for item in alone] == [
        (item.entry.name, item.uses) for item in whole[1]
    ]


def test_each_drawn_document_arrives_once_with_the_steps_that_train_on_it(tmp_path):
    catalog = written(tmp_path)
    drawn = drawn_documents(catalog, 40, 0, Draw.TOKEN)
    steps = layer_steps([0], 40)[0]

    items = items_of(catalog, drawn, 0, steps, 0)

    assert sum(item.uses for item in items) == 40
    assert len({item.entry.name for item in items}) == len(items)
    assert {item.entry.name: item.uses for item in items} == {
        catalog[index].name: count for index, count in collections.Counter(drawn).items()
    }
    assert all(item.size > 0 for item in items)


def test_a_drawn_document_without_a_file_for_the_layer_says_so(tmp_path):
    catalog = written(tmp_path)
    drawn = drawn_documents(catalog, 40, 0, Draw.TOKEN)

    with pytest.raises(FileNotFoundError, match="layer 5"):
        items_of(catalog, drawn, 5, layer_steps([5], 40)[5], 0)


def items_with(uses, size=10):
    entry = Entry((None, "t", "train"), "x", 1)
    return [Item(0, entry, count, size) for count in uses]


def test_the_sequence_trains_each_item_exactly_the_steps_it_was_dealt():
    items = items_with([5, 1, 3, 2, 4])

    sequence = interleave(items, float("inf"), 3, 0)

    assert collections.Counter(sequence) == {index: item.uses for index, item in enumerate(items)}


def test_the_steps_of_one_item_are_mixed_with_the_others_in_its_window():
    items = items_with([6, 6, 6])

    sequence = interleave(items, float("inf"), 3, 0)

    assert sequence != sorted(sequence)
    assert sequence[:6] != [sequence[0]] * 6


def test_a_window_of_one_trains_an_item_through_before_the_next():
    items = items_with([3, 2, 4])

    assert interleave(items, float("inf"), 1, 0) == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_the_items_active_together_never_exceed_the_window_or_the_cards_budget():
    items = items_with([4] * 12, size=10)

    for window, budget in ((4, float("inf")), (8, 30), (8, 5)):
        sequence = interleave(items, budget, window, 1)
        left = {index: item.uses for index, item in enumerate(items)}
        active = set()
        for index in sequence:
            active.add(index)
            left[index] -= 1
            assert len(active) <= window
            assert len(active) * 10 <= max(budget, 10)
            if not left[index]:
                active.discard(index)


def test_the_sequence_is_worked_out_from_the_seed_alone():
    items = items_with([5, 5, 5, 5])
    assert interleave(items, float("inf"), 3, 7) == interleave(items, float("inf"), 3, 7)
    assert interleave(items, float("inf"), 3, 7) != interleave(items, float("inf"), 3, 8)


def a_pool(tmp_path, steps=24, host_bytes=float("inf"), window=3, layer=0, batch=4):
    catalog = written(tmp_path)
    drawn = drawn_documents(catalog, steps, 0, Draw.TOKEN)
    items = items_of(catalog, drawn, layer, layer_steps([layer], steps)[layer], 0)
    sequence = interleave(items, host_bytes, window, 0)
    pool = Rolling(layer, items, sequence, fake_cache.config(), "cpu", host_bytes, host_bytes, batch, 0)
    return pool, items, sequence


def test_a_step_gets_the_document_it_was_dealt_with_rows_it_may_train_on(tmp_path):
    pool, items, sequence = a_pool(tmp_path)

    for step, index in enumerate(sequence, start=1):
        drawn = pool.next(step)
        assert drawn.document.name == items[index].entry.name
        assert drawn.layer == 0 and drawn.step == step
        assert len(drawn.rows) == 4
        assert bool((drawn.rows < drawn.document.train_query_count).all())
    pool.close()


def test_a_document_is_read_once_however_many_steps_train_on_it(tmp_path):
    pool, items, sequence = a_pool(tmp_path, steps=60)
    assert len(sequence) > len(items)

    for step in range(1, len(sequence) + 1):
        pool.next(step)

    assert pool.documents_read == len(items)
    pool.close()


def test_a_pool_that_holds_two_documents_never_holds_more(tmp_path):
    catalog = written(tmp_path)
    drawn = drawn_documents(catalog, 60, 0, Draw.TOKEN)
    items = items_of(catalog, drawn, 0, layer_steps([0], 60)[0], 0)
    budget = 2 * max(item.size for item in items)
    sequence = interleave(items, budget, 8, 0)
    pool = Rolling(0, items, sequence, fake_cache.config(), "cpu", budget, budget, 4, 0)

    for step in range(1, len(sequence) + 1):
        pool.next(step)
        assert pool.host_used <= budget

    pool.close()


def test_the_pool_counts_the_bytes_it_read(tmp_path):
    pool, items, sequence = a_pool(tmp_path)
    for step in range(1, len(sequence) + 1):
        pool.next(step)

    assert pool.bytes_read == sum(
        layer_path(tmp_path, "tokens", "train", item.entry.name, 0).stat().st_size for item in items
    )
    pool.close()


def test_a_file_that_cannot_be_read_stops_the_step_that_needs_it(tmp_path):
    pool, items, sequence = a_pool(tmp_path, window=1, host_bytes=1)
    pool.next(1)
    for item in items[1:]:
        layer_path(tmp_path, "tokens", "train", item.entry.name, 0).unlink()

    with pytest.raises(RuntimeError, match="failed reading"):
        for step in range(2, len(sequence) + 1):
            pool.next(step)
    pool.close()


def test_rows_come_off_a_generator_of_the_layers_own(tmp_path):
    first, _, sequence = a_pool(tmp_path / "a")
    second, _, _ = a_pool(tmp_path / "b")

    for step in range(1, len(sequence) + 1):
        assert torch.equal(first.next(step).rows, second.next(step).rows)
    first.close()
    second.close()


def test_what_is_left_of_a_plan_trains_the_steps_left_on_the_same_documents():
    items = [
        Item(0, Entry((None, "t", "train"), name, 1), uses, 10)
        for name, uses in zip("abcd", (2, 3, 1, 2), strict=True)
    ]
    sequence = [0, 1, 0, 1, 2, 3, 1, 3]

    left_items, left_sequence = remaining(items, sequence, 3)

    # a is done, b has two steps left and is read again, c and d are untouched
    assert [item.entry.name for item in left_items] == ["b", "c", "d"]
    assert [item.uses for item in left_items] == [2, 1, 2]
    assert [left_items[index].entry.name for index in left_sequence] == [
        items[index].entry.name for index in sequence[3:]
    ]


def test_a_pool_rebuilt_from_its_state_hands_out_what_the_pool_it_replaces_would_have(tmp_path):
    whole, items, sequence = a_pool(tmp_path / "a")
    stopped, _, _ = a_pool(tmp_path / "b")
    expected = [whole.next(step) for step in range(1, len(sequence) + 1)]
    for step in range(1, 10):
        stopped.next(step)

    state = stopped.state()
    stopped.close()
    left_items, left_sequence = remaining(state["items"], state["sequence"], state["cursor"])
    resumed = Rolling(
        0, left_items, left_sequence, fake_cache.config(), "cpu", 1e12, state["device_bytes"], 4, 0
    )
    resumed.generator.set_state(state["generator"])

    for step in range(10, len(sequence) + 1):
        drawn = resumed.next(step)
        assert drawn.document.name == expected[step - 1].document.name
        assert torch.equal(drawn.rows, expected[step - 1].rows)
    whole.close()
    resumed.close()


def test_a_step_waiting_on_its_document_is_interrupted_without_moving_the_pool(tmp_path):
    catalog = written(tmp_path)
    drawn = drawn_documents(catalog, 6, 0, Draw.TOKEN)
    items = items_of(catalog, drawn, 0, layer_steps([0], 6)[0], 0)
    released = threading.Event()

    def load(item):
        released.wait()
        return Document(item.path, fake_cache.config(), "cpu")

    pool = Rolling(
        0, items, interleave(items, 1e12, 3, 0), fake_cache.config(), "cpu", 1e12, 1e12, 4, 0, load=load
    )
    stop = threading.Event()
    stop.set()

    with pytest.raises(Interrupted):
        pool.next(1, stop)
    assert pool.state()["cursor"] == 0
    released.set()
    assert pool.next(1, threading.Event()).step == 1
    pool.close()


def test_off_a_card_the_device_budget_is_unbounded_unless_configured():
    assert device_budget("cpu", None, 1) == float("inf")
    assert device_budget("cpu", 10, 2) == 5e9


def test_the_sets_a_pool_is_built_from_resolve_through_the_manifest(tmp_path):
    from lsa.training.distillation.data.cache import listing

    fake_cache.collection(tmp_path, documents=NAMES)
    entries_ = listing(tmp_path, DocumentSet(source="tokens", split="train"))

    assert [entry["name"] for entry in entries_] == list(NAMES)
    assert {entry["length"] for entry in entries_} == {fake_cache.LENGTH}
