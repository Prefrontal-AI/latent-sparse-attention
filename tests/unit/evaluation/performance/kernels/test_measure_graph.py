"""The decode graph's metadata binding, modelled without a device.

A graph keeps the address of each buffer its kernels read when it is captured.
The fake backend below keeps an eager step's metadata in a tensor it allocates on every call and frees the one before,
as a triton backend does, and keeps its graph metadata in one buffer that a graph-mode init refills.
Its graph is a stand-in that records the tensor the backend held at capture and reads it again on replay.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.measure.graph import (
    bind_capture_metadata,
    capture_indexer_call,
    check_replay,
    differing_slots,
)
from lsa.evaluation.performance.kernels.measure.snapshot import compare
from lsa.evaluation.performance.kernels.measure.variant import Variant

TOKENS = 8


class FakeBackend:
    """Eager steps allocate fresh metadata, graph steps refill one buffer - and only a capture-mode init binds it."""

    def __init__(self) -> None:
        self.graph_buffer = torch.zeros(TOKENS)
        self.metadata = self.graph_buffer
        self.captured_in_capture = 0

    def init_forward_metadata(self, batch) -> None:
        previous = self.metadata
        self.metadata = batch.slots.clone()
        # A freed block is reused by the next allocation that fits
        if previous is not self.graph_buffer:
            previous.fill_(-1.0)

    def init_forward_metadata_out_graph(self, batch, in_capture: bool = False) -> None:
        self.graph_buffer.copy_(batch.slots)
        if in_capture:
            self.captured_in_capture += 1
            self.metadata = self.graph_buffer

    def capture(self):
        held = self.metadata

        def replay():
            return held.clone()

        return replay


def replays_after_eager_steps(backend, batch, bind: bool) -> list[bool]:
    backend.init_forward_metadata(batch)
    if bind:
        bind_capture_metadata(SimpleNamespace(linear=False), backend, batch)
    replay = backend.capture()
    matches = []
    # The timed trials restore, run an eager step or a graph step, and alternate them
    for eager in (True, False, True, True, False, False, True, False):
        if eager:
            backend.init_forward_metadata(batch)
        backend.init_forward_metadata_out_graph(batch)
        matches.append(bool(torch.equal(replay(), batch.slots)))
    return matches


def test_replay_matches_eager_after_every_eager_step() -> None:
    batch = SimpleNamespace(slots=torch.arange(TOKENS, dtype=torch.float32))
    assert all(replays_after_eager_steps(FakeBackend(), batch, bind=True))


def test_capture_over_eager_metadata_replays_stale_buffers() -> None:
    batch = SimpleNamespace(slots=torch.arange(TOKENS, dtype=torch.float32))
    assert not all(replays_after_eager_steps(FakeBackend(), batch, bind=False))


def test_linear_layer_binds_nothing() -> None:
    backend = FakeBackend()
    bind_capture_metadata(SimpleNamespace(linear=True), backend, SimpleNamespace(slots=torch.zeros(TOKENS)))
    assert backend.captured_in_capture == 0


class FakeLayer:
    """The measured layer's eager step and variant, over the batch's slots."""

    def __init__(self, batch) -> None:
        self.batch = batch
        self.restored = 0
        self.variant = SimpleNamespace(linear=False, pinned_selection=self.pinned_selection)

    @contextmanager
    def pinned_selection(self, selection):
        yield

    def restore(self) -> None:
        self.restored += 1

    def execute(self) -> torch.Tensor:
        return self.batch.slots.clone()


def timed_replays_checked(bind: bool) -> int:
    batch = SimpleNamespace(slots=torch.arange(TOKENS, dtype=torch.float32))
    backend, layer = FakeBackend(), FakeLayer(batch)
    backend.init_forward_metadata(batch)
    if bind:
        bind_capture_metadata(layer.variant, backend, batch)
    replay = backend.capture()
    checked = 0
    for eager in (False, False, True, False, True, True, False):
        if eager:
            backend.init_forward_metadata(batch)
        backend.init_forward_metadata_out_graph(batch)
        check_replay(
            layer.variant,
            layer.execute,
            SimpleNamespace(restore=layer.restore),
            compare,
            batch.slots,
            replay(),
        )
        checked += 1
    return checked


def test_every_timed_replay_of_a_bound_capture_passes_the_check() -> None:
    assert timed_replays_checked(bind=True) == 7


def test_a_stale_replay_late_in_the_timed_set_fails_the_check() -> None:
    # The first replays match - the freed block still holds the metadata
    # => only a check on every replay catches the later ones
    with pytest.raises(AssertionError):
        timed_replays_checked(bind=False)


class Doubler(torch.nn.Module):
    def forward(self, x, layer_id=0):
        return x * 2


def doubling_variant(indexer) -> Variant:
    variant = Variant.__new__(Variant)
    variant.family, variant.qsa, variant.lsa, variant.indexer = Family.GLM5_NEXT, False, False, indexer
    return variant


def test_the_indexer_s_call_is_kept_as_copies_the_step_cannot_overwrite() -> None:
    indexer = Doubler()
    handle, kept = capture_indexer_call(doubling_variant(indexer))
    x = torch.arange(3.0)
    indexer(x, layer_id=5)
    handle.remove()
    # The step frees and reuses its tensors once it is done
    x.fill_(-1.0)
    positional, kwargs = kept["call"]
    assert kwargs == {"layer_id": 5}
    assert indexer(*positional, **kwargs).tolist() == [0.0, 2.0, 4.0]


class Selecting:
    """A Qwen LSA indexer, whose backend calls `select`"""

    def select(self, q, k, forward_batch):
        return q * 2


def test_a_qwen_lsa_indexer_s_select_is_kept_and_run_alone_on_the_copies() -> None:
    indexer = Selecting()
    variant = Variant.__new__(Variant)
    variant.family, variant.qsa, variant.lsa, variant.indexer = Family.QWEN3_5, False, True, indexer
    handle, kept = capture_indexer_call(variant)
    q = torch.arange(3.0)
    indexer.select(q, None, "batch")
    handle.remove()
    q.fill_(-1.0)
    positional, kwargs = kept["call"]
    assert positional[2] == "batch"
    assert variant.call_indexer(*positional, **kwargs).tolist() == [0.0, 2.0, 4.0]
    assert "select" not in vars(indexer)


def test_two_selections_are_read_as_sets_and_padding_is_not_a_slot() -> None:
    step = torch.tensor([[5, 3, 9, -1]], dtype=torch.int32)
    assert differing_slots(torch.tensor([[9, 5, 3, -1]], dtype=torch.int32), step) == 0
    assert differing_slots(torch.tensor([[9, 5, 4, -1]], dtype=torch.int32), step) == 1
