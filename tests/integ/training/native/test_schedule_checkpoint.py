"""Hold branching and mid-decay resume to real torchtitan checkpoints and native streams."""

import pytest
import torch
import torch.distributed as dist

from lsa.training.native.trainer import DecayStartCheckpointer
from tests.helpers.training.data.fake_sources import write_shards
from tests.helpers.training.schedule import CheckpointRun


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_mid_decay_resume_restores_weights_moments_schedule_and_the_next_batch(tmp_path, process_group):
    card = write_shards(tmp_path / "shards", shards=4, tokens=70)
    whole = CheckpointRun(tmp_path / "whole", card)
    interrupted = CheckpointRun(tmp_path / "resume", card)
    resumed = None
    try:
        whole.advance(10)
        interrupted.advance(8, save=True)
        assert interrupted.schedulers.state_dict() == {"last_epoch": 8}
        interrupted.checkpointer.close()
        resumed = CheckpointRun(tmp_path / "resume", card)
        assert resumed.checkpointer.load()
        assert resumed.step == 8
        resumed.advance(2)
        assert interrupted.rates + resumed.rates == whole.rates
        for name, weight in whole.model.state_dict().items():
            assert torch.equal(resumed.model.state_dict()[name], weight)
        assert resumed.schedulers.state_dict() == whole.schedulers.state_dict()
        expected = next(iter(whole.stream))
        actual = next(iter(resumed.stream))
        assert torch.equal(actual["input"], expected["input"])
        for name, value in whole.optimizers.state_dict().items():
            actual = resumed.optimizers.state_dict()[name]
            assert torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
    finally:
        whole.checkpointer.close()
        interrupted.checkpointer.close()
        if resumed is not None:
            resumed.checkpointer.close()


@pytest.mark.parametrize(
    ("seq_len", "world_size", "tokens_per_batch", "accumulation"),
    [
        (16, 1, 32, 1),
        (8, 2, 16, 2),
        (16, 2, 16, 4),
    ],
)
def test_a_branch_restarts_optimizer_schedule_and_stream_on_its_new_layout(
    tmp_path, process_group, seq_len, world_size, tokens_per_batch, accumulation
):
    trunk_card = write_shards(tmp_path / "trunk-data", shards=4, tokens=70)
    branch_card = write_shards(tmp_path / "branch-data", shards=8, tokens=99)
    trunk = CheckpointRun(tmp_path / "trunk", trunk_card, interval=99)
    branches = []
    try:
        trunk.advance(5, save=True)
        checkpoint = str(tmp_path / "trunk/checkpoint/step-5")
        for rank in range(world_size):
            branch = CheckpointRun(
                tmp_path / f"branch-{rank}",
                branch_card,
                seq_len=seq_len,
                world_size=world_size,
                rank=rank,
                tokens_per_batch=tokens_per_batch,
                initial_load_path=checkpoint,
            )
            fresh = CheckpointRun(
                tmp_path / f"fresh-{rank}",
                branch_card,
                seq_len=seq_len,
                world_size=world_size,
                rank=rank,
                tokens_per_batch=tokens_per_batch,
            )
            branches.extend([branch, fresh])
            assert branch.checkpointer.load()
            assert branch.step == 0
            assert branch.schedulers.state_dict() == fresh.schedulers.state_dict()
            assert branch.optimizers.optimizers[0].state == {}
            for name, weight in trunk.model.state_dict().items():
                assert torch.equal(branch.model.state_dict()[name], weight)
            assert torch.equal(next(iter(branch.stream))["input"], next(iter(fresh.stream))["input"])
            branch.advance(1, accumulation=accumulation)
            assert branch.step == 1
            assert all(torch.isfinite(parameter).all() for parameter in branch.model.parameters())
    finally:
        trunk.checkpointer.close()
        for branch in branches:
            branch.checkpointer.close()


def test_decay_start_is_saved_off_interval_and_survives_retention(tmp_path, process_group):
    card = write_shards(tmp_path / "shards", shards=4, tokens=70)
    run = CheckpointRun(tmp_path / "run", card, interval=2, keep_latest_k=2)
    try:
        assert isinstance(run.checkpointer, DecayStartCheckpointer)
        run.advance(10, save=True)
    finally:
        run.checkpointer.close()
    saved = {path.name for path in (tmp_path / "run/checkpoint").iterdir()}
    assert "step-5" in saved
    assert "step-10" in saved
    assert "step-2" not in saved


def test_loader_checkpoints_keep_each_ranks_position_under_distinct_keys(tmp_path, process_group):
    card = write_shards(tmp_path / "shards", shards=4, tokens=70)
    first = CheckpointRun(tmp_path / "first", card, world_size=2, rank=0)
    second = CheckpointRun(tmp_path / "second", card, world_size=2, rank=1)
    try:
        first.advance(2)
        second.advance(2)
        first_state, second_state = first.loader.state_dict(), second.loader.state_dict()
        assert set(first_state) == {"dp_world_size", "dp_rank_0"}
        assert set(second_state) == {"dp_world_size", "dp_rank_1"}
        joined = first_state | second_state
        first.loader.load_state_dict(joined)
        second.loader.load_state_dict(joined)
        with pytest.raises(ValueError, match="needs state for dp_rank_1"):
            second.loader.load_state_dict(first_state)
        with pytest.raises(ValueError, match="expected 2"):
            first.loader.load_state_dict({"dp_world_size": 1})
    finally:
        first.checkpointer.close()
        second.checkpointer.close()


def test_stochastic_mid_decay_resume_restores_query_sampling_and_weights(tmp_path, process_group):
    card = write_shards(tmp_path / "shards", shards=4, tokens=70)
    whole = CheckpointRun(tmp_path / "whole", card)
    interrupted = CheckpointRun(tmp_path / "resume", card)
    resumed = None
    try:
        torch.manual_seed(23)
        whole.advance(10, stochastic=True, validate=True)
        expected_random_state = torch.get_rng_state()
        torch.manual_seed(23)
        interrupted.advance(8, save=True, stochastic=True, validate=True)
        interrupted.checkpointer.close()
        torch.rand(17)
        resumed = CheckpointRun(tmp_path / "resume", card)
        assert resumed.checkpointer.load()
        resumed.advance(2, stochastic=True, validate=True)
        assert torch.equal(torch.get_rng_state(), expected_random_state)
        for name, weight in whole.model.state_dict().items():
            assert torch.equal(resumed.model.state_dict()[name], weight)
    finally:
        whole.checkpointer.close()
        interrupted.checkpointer.close()
        if resumed is not None:
            resumed.checkpointer.close()


def test_loading_native_training_state_requires_rank_random_generators(tmp_path, process_group):
    card = write_shards(tmp_path / "shards", shards=4, tokens=70)
    run = CheckpointRun(tmp_path / "run", card)
    try:
        with pytest.raises(ValueError, match="run.init_from"):
            run.training_state.load_state_dict({"step": 8, "ntokens_seen": 0})
    finally:
        run.checkpointer.close()
