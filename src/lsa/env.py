"""What a job's environment has to be before torch or transformers starts.

Four contracts, and they are not the same thing.
`OFFLINE_HF` is what anything loading a model or a corpus needs on a cluster with no egress, `EVAL_DATA` is
where the suite's prompt files live, and `DISTRIBUTED_TORCH` is what a run spread over several GPUs needs.
`OFFLINE_HF` and `DISTRIBUTED_TORCH` meet only at `HF_HUB_OFFLINE`, and keeping them in one module is what lets a reader check that the two
spellings agree.

This is the launch path rather than the code, which is why it is not in [constants](constants.py): a value
here is read by a shell or by a launcher, and none of it reaches a tensor.
The dashboard project and the rendezvous port span are here for the same reason and stay beside the function
that spells them into an environment - a recipe read once before the process starts, not a value two modules
have to agree on.

`OFFLINE_HF` is shell lines rather than a mapping because its paths are built from the environment they are
being set in - `$USER`, `$HF_HOME` and `$LSA_ROOT` are the target node's, not the submitting host's - and
only a shell can resolve that. The shell form is also what `lsa env` prints, so an sbatch file
evaluates it rather than copying it. `DISTRIBUTED_TORCH` is a mapping, which
`exports` spells as the lines a job's shell runs.
"""

import os
import subprocess
from pathlib import Path

from lsa import paths, site


def declared_hf_home() -> str:
    """Return the hub cache the site names, unexpanded - `$USER` is the shell's or `hf_home`'s to expand.

    Returns:
        The site's `hf_home`, or an empty string where it names none and the hub's own default applies.
    """
    return str(site.load().get("hf_home") or "")


# The shared root, exported so the job's python resolves the same one its shell did
# => the site's root unless the node already has `LSA_ROOT`
ROOT_LINE: str = f'export {paths.ROOT_VARIABLE}="${{{paths.ROOT_VARIABLE}:-{paths.declared_root()}}}"'

# No egress on the cluster - the model and the datasets come from the caches or not at all
# => the HF datasets are shared, beside the prompt corpus every run reads
OFFLINE_HF: tuple[str, ...] = (
    ROOT_LINE,
    *((f'export HF_HOME="${{HF_HOME:-{declared_hf_home()}}}"',) if declared_hf_home() else ()),
    f'export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-${paths.ROOT_VARIABLE}/data/hf}}"',
    "export HF_HUB_OFFLINE=1",
    "export TRANSFORMERS_OFFLINE=1",
    "export TRANSFORMERS_VERBOSITY=error",
    "export TOKENIZERS_PARALLELISM=false",
)

# A collection streams a dataset from the hub, so it is the one job that is not offline
# => the same root and cache as the offline lines, and the hub left on
ONLINE_HF: tuple[str, ...] = (
    ROOT_LINE,
    *((f'export HF_HOME="${{HF_HOME:-{declared_hf_home()}}}"',) if declared_hf_home() else ()),
    "export HF_HUB_OFFLINE=0",
    "export TRANSFORMERS_VERBOSITY=error",
    "export TOKENIZERS_PARALLELISM=true",
)

# A teacher extraction - a span of GLM-5.3-Flash fills most of an H200, and a 1M document's late chunks ask
# the allocator for 8 GiB score tiles that a fragmented heap cannot place
TEACHER: tuple[str, ...] = (*OFFLINE_HF, "export PYTORCH_ALLOC_CONF=expandable_segments:True")

# A distillation run - the card holds a rolling window of 1M-row documents and a report's read-ahead beside it,
# and a fragmented heap OOMed every shard of the first run with 12-24 GiB reserved but unallocated
DISTILL: tuple[str, ...] = (*OFFLINE_HF, "export PYTORCH_ALLOC_CONF=expandable_segments:True")

# The suite's fixed prompt files, 4.1G at the full counts - a cache root under the shared one, filled by `prepare`
EVAL_DATA: tuple[str, ...] = (
    ROOT_LINE,
    f'export LSA_EVAL_DATA="${{LSA_EVAL_DATA:-${paths.ROOT_VARIABLE}/data/benchmarks}}"',
)

# What a run spread over several GPUs needs - unbuffered, since a block-buffered job under sbatch looks hung
DISTRIBUTED_TORCH: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "8",
    "HF_HUB_OFFLINE": "1",
}

# One Weights & Biases project for every run, the stages told apart by name
# => a run sits beside the others it is compared with
WANDB_PROJECT: str = "lsa"


def wandb_run(name: str, folder: str) -> dict[str, str]:
    """Return the environment that names one Weights & Biases run.

    Args:
        name: The run's name, which is what the dashboard shows.
        folder: Where the run writes, so an offline run's files land beside its logs.

    Returns:
        The variables to set, for a caller that lets anything already exported win.
    """
    return {"WANDB_PROJECT": WANDB_PROJECT, "WANDB_RUN_NAME": name, "WANDB_DIR": folder}


# One port per job, so two jobs on one node do not rendezvous with each other
PORT_BASE: int = 20000
PORT_SPAN: int = 20000


def hf_home() -> Path:
    """Return the hub cache root this process reads models from.

    `HF_HOME` when it is set, then the site's `hf_home` with its variables expanded, then the hub's own default.
    """
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    if declared_hf_home():
        return Path(os.path.expandvars(declared_hf_home()))
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "huggingface"


def site_hf_home() -> str | None:
    """Return the site's hub cache when this machine has the directory it sits in, and None anywhere else.

    A site names the cache of its own machine. Setting it where its parent directory is missing points a laptop
    or a CI runner at a path it cannot create, which is a failure rather than a cache miss.
    """
    if not declared_hf_home():
        return None
    home = os.path.expandvars(declared_hf_home())
    return home if os.path.isdir(os.path.dirname(os.path.dirname(home))) else None


def prefer_offline_hub() -> None:
    """Read every hub source from the cache in this process, unless the environment already says otherwise.

    The in-process half of `OFFLINE_HF`, for a package that reads a tokenizer, a dataset or an lm-eval task.
    Called before any of its modules imports one, because the hub reads its offline switches at import.
    The cluster has no egress, so every source is read from the cache.
    """
    cache = site_hf_home()
    if cache:
        os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def exports(variables: dict[str, str]) -> tuple[str, ...]:
    """Return a contract as the shell lines slurm runs before a task starts.

    Args:
        variables: Name to value.

    Returns:
        One `export NAME=value` per entry, in order.
    """
    return tuple(f"export {name}={value}" for name, value in variables.items())


def adopt_job_environment() -> None:
    """Fill in the variables `torchrun` would have set, from slurm's own.

    A task started by slurm directly has none of the variables torch's launcher publishes, and slurm has all
    four under its own names, so this is a translation. It reads them itself rather than through submitit,
    so a gate that needs four integers does not fail on an unrelated import.

    Raises:
        RuntimeError: Neither torch's launcher nor slurm has published a rank.
    """
    if "SLURM_PROCID" not in os.environ:
        raise RuntimeError("no RANK and no SLURM_PROCID - start this under srun, or set RANK yourself")

    job_id = os.environ.get("SLURM_JOB_ID", "0").split("_")[0]
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", rendezvous_host())
    os.environ.setdefault("MASTER_PORT", str(PORT_BASE + int(job_id) % PORT_SPAN))


def rendezvous_host() -> str:
    """Return the node every rank of this job rendezvouses on.

    Slurm names the allocation compressed, as `worker-[13,15]`, and only `scontrol` expands that.
    A single-node allocation is already its own hostname, which is the case worth not shelling out for.

    Returns:
        Hostname of the job's first node.

    Raises:
        RuntimeError: The allocation names several nodes and `scontrol` could not expand them.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if not nodelist:
        return "localhost"
    if "[" not in nodelist and "," not in nodelist:
        return nodelist

    try:
        expanded = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as failure:
        raise RuntimeError(f"cannot expand SLURM_JOB_NODELIST {nodelist!r} - {failure}") from failure
    if not expanded:
        raise RuntimeError(f"scontrol expanded SLURM_JOB_NODELIST {nodelist!r} to nothing")
    return expanded[0]


def main() -> None:
    """Print the offline hub contract as shell lines, for an sbatch file to evaluate."""
    print("\n".join(OFFLINE_HF))
