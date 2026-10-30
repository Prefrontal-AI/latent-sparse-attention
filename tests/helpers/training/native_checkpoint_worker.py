"""Exercise native checkpoint branches and exact resume through FSDP2 on real GPUs."""

import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from lsa import assets
from tests.helpers.training.data.fake_sources import write_shards
from tests.helpers.training.schedule import CheckpointRun


def weights(run):
    return {name: value.full_tensor().cpu() for name, value in run.model.state_dict().items()}


def run_phase(root, phase):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        write_shards(root / "trunk-data", shards=4, tokens=70)
        write_shards(root / "branch-data", shards=8, tokens=99)
    dist.barrier()
    card = assets.ShardCard(
        name="toy", path=str(root / ("trunk-data" if phase == "trunk" else "branch-data")), tokenizer="gpt2"
    )
    initial = str(root / "trunk/checkpoint/step-5") if phase == "branch" else None
    if phase == "shrink":
        initial = str(root / "branch/checkpoint/step-6")
    folder = root / ("branch" if phase == "resume" else phase)
    run = CheckpointRun(
        folder,
        card,
        seq_len=8 if phase in ("trunk", "shrink") else 16,
        world_size=world_size,
        rank=rank,
        tokens_per_batch=16,
        initial_load_path=initial,
        shard=True,
    )
    try:
        if phase == "trunk":
            run.advance(5, save=True)
        elif phase == "branch":
            assert run.checkpointer.load()
            assert run.step == 0
            assert not run.optimizers.optimizers[0].state
            expected = torch.load(root / "trunk.pt", weights_only=False)["weights"]
            for name, weight in weights(run).items():
                torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)
            torch.manual_seed(1900 + rank)
            states = [None] * world_size
            dist.all_gather_object(states, torch.cuda.get_rng_state())
            assert not torch.equal(states[0], states[1])
            run.advance(6, accumulation=2, save=True, stochastic=True, validate=True)
            at_six = weights(run)
            if rank == 0:
                torch.save(at_six, root / "branch-six.pt")
            run.advance(4, accumulation=2, stochastic=True, validate=True)
        elif phase == "resume":
            print(f"rank {rank}: loading mid-decay", flush=True)
            assert run.checkpointer.load(step=6)
            print(f"rank {rank}: loaded mid-decay", flush=True)
            assert run.step == 6
            run.advance(4, accumulation=2, stochastic=True, validate=True)
            expected = torch.load(root / "branch.pt", weights_only=False)
            assert run.rates == expected["rates"][6:]
            for name, weight in weights(run).items():
                torch.testing.assert_close(weight, expected["weights"][name], rtol=0, atol=0)
        elif phase == "shrink":
            assert run.checkpointer.load()
            assert run.step == 0
            assert not run.optimizers.optimizers[0].state
            expected = torch.load(root / "branch-six.pt", weights_only=False)
            for name, weight in weights(run).items():
                torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)
            run.advance(1)
        saved = {"weights": weights(run), "rates": run.rates}
        if rank == 0:
            torch.save(saved, root / f"{phase}.pt")
    finally:
        run.checkpointer.close()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        run_phase(Path(sys.argv[1]), sys.argv[2])
    finally:
        dist.destroy_process_group()
