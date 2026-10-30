"""Check real FSDP2 checkpoint resharding and mid-decay resume across process restarts."""

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_branching_reshards_one_to_two_and_back_and_mid_decay_resume_is_exact(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 checkpoint resharding needs two GPUs")
    for phase, ranks in (("trunk", 1), ("branch", 2), ("resume", 2), ("shrink", 1)):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={ranks}",
                "--module",
                "tests.helpers.training.native_checkpoint_worker",
                str(tmp_path),
                phase,
            ],
            env={**os.environ, "OMP_NUM_THREADS": "2"},
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
