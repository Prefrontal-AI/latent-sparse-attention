"""Time every arm of one model's layer, both phases, one after another on one GPU of a node nothing else runs on.

    lsa evaluate isolated glm53_flash --output $LSA_ROOT/layer-perf/isolated
    lsa evaluate isolated qwen38_27b --output $LSA_ROOT/layer-perf/isolated --dry-run
    lsa evaluate isolated glm53_flash --phase decode --output $LSA_ROOT/layer-perf/isolated-decode
    lsa evaluate isolated glm53_flash --arm pisa_c8 --output $LSA_ROOT/layer-perf/isolated-pisa-c8
    lsa evaluate isolated glm53_flash --arm pisa_c16 --output $LSA_ROOT/layer-perf/isolated-pisa-c16
    lsa evaluate isolated glm53_flash --gpu b300 --output $LSA_ROOT/layer-perf/isolated-b300 --dry-run

A layer timing is only comparable across arms when the card behaved the same for all of them. Jobs of one arm each
land on whichever nodes are free and share them with other users' jobs, so an arm can be timed on a card another
job is heating or on a node with a busier power budget. This is one job that asks for a whole node - slurm's
`--exclusive` - and holds one GPU of it for every arm and phase in turn, so the arms differ in what they are
and not in where they ran. The telemetry each point keeps (see [telemetry](../telemetry.py)) says what the card did.

Every arm prefills in chunks of 32,768 tokens, the contexts are the ladder the published records hold, and the
design's arms - flat and tree - run their kernels in CuTeDSL. The image is the configs' own. An arm that fails
does not stop the ones after it, and the job says which failed when the last is done.

`--gpu b300` times the B300 arms: arm X is `b300_X.yaml` beside the H200 file, with the B300 image and backends its
own configs set. An arm with no B300 file is an error and not a silent fall back to the H200 file. The chunk is
32,768 for both phases on both cards; an arm falls back to 8,192 only when 32,768 runs out of memory at a context,
and its record says so.
"""

import argparse
import os
import traceback
from typing import TYPE_CHECKING, Any

from lsa import env, slurm
from lsa.evaluation import require_extra
from lsa.evaluation.constants import Launches
from lsa.sglang.launch import checkout_root

if TYPE_CHECKING:
    from lsa.evaluation.config import KernelsConfig

# The arms of each model, in the order they run - the configs under `configs/evaluation/kernels/<model>/`
ARMS: dict[str, tuple[str, ...]] = {
    "glm53_flash": ("full", "native", "lsa_flat", "lsa_tree", "pisa"),
    "qwen38_27b": ("full", "qsa", "lsa_flat", "lsa_tree", "pisa"),
}

# The cards an arm can be timed on, as `gpu.TOKENS` names them - the first is the default, the H200 files
GPUS: tuple[str, ...] = ("h200", "b300")

CONFIGS = checkout_root() / "configs" / "evaluation" / "kernels"

CHUNK = 32768

# The phases of a layer, in the order each arm runs them
PHASES: tuple[str, ...] = ("prefill", "decode")

# What the published records of a model were timed with beyond its config, by card - GLM's sparse attention kernels
# on H200. fa3 does not run on sm_103, so B300 sets none: its configs choose their own backends
MODEL_OVERRIDES: dict[tuple[str, str], tuple[str, ...]] = {
    ("glm53_flash", "h200"): ("data.prefill_backend=flashmla_sparse", "data.decode_backend=fa3"),
    ("glm53_flash", "b300"): (),
    ("qwen38_27b", "h200"): (),
    ("qwen38_27b", "b300"): (),
}

# Arms a job runs only when asked by `--arm`, beside the ones every run of a model times
EXTRA_ARMS: dict[str, tuple[str, ...]] = {
    "glm53_flash": ("pisa_c8", "pisa_c16"),
    "qwen38_27b": ("pisa_c8", "pisa_c16"),
}

# B300 arms whose config is one file per phase, to the file of each phase - the arm needs `--arm` like an extra one
PER_PHASE_ARMS: dict[str, dict[str, dict[str, str]]] = {
    "glm53_flash": {
        "lsa_flat_early_budget": {
            "prefill": "b300_lsa_flat_early_budget_prefill",
            "decode": "b300_lsa_flat_early_budget_decode",
        }
    },
    "qwen38_27b": {},
}

# The arms that run the design's selector, the only ones that have kernels to choose
DESIGN_ARMS: tuple[str, ...] = ("lsa_flat", "lsa_tree", "lsa_flat_early_budget")


def arm_files(model: str, arm: str, card: str = "h200") -> list[str]:
    """Return the config files one arm is loaded from.

    Args:
        model: The model's config directory.
        arm: The arm's name.
        card: The card's token, one of `GPUS`.

    Returns:
        The H200 arm's file, the B300 arm's `b300_` file, or a per-phase arm's files on B300.

    Raises:
        ValueError: The card has no file for the arm.
    """
    if card == "h200":
        return [str(CONFIGS / model / f"{arm}.yaml")]
    if arm in PER_PHASE_ARMS[model]:
        names = list(PER_PHASE_ARMS[model][arm].values())
    else:
        names = [f"b300_{arm}"]
    files = [CONFIGS / model / f"{name}.yaml" for name in names]
    if missing := [file.name for file in files if not file.exists()]:
        raise ValueError(f"no {card} config for the arm {arm!r} of {model!r} - missing {', '.join(missing)}")
    return [str(file) for file in files]


def arm_overrides(model: str, arm: str, output: str | None, card: str = "h200") -> list[str]:
    """Return the dotted overrides one arm is loaded with.

    Args:
        model: The model's config directory.
        arm: The arm's config file, without its suffix.
        output: Where the records go, or None for the config's own `run.output`.
        card: The card's token, one of `GPUS`.

    Returns:
        The chunk every arm prefills in, the model's own settings on the card, CuTeDSL kernels for a design arm and the output.
    """
    overrides = [f"data.chunk={CHUNK}", *MODEL_OVERRIDES[(model, card)]]
    if arm in DESIGN_ARMS:
        overrides.append("model.serving.kernels=cutedsl")
    if output:
        overrides.append(f"run.output={output}")
    return overrides


def load_arms(
    model: str,
    output: str | None,
    phases: tuple[str, ...] = PHASES,
    arms: tuple[str, ...] = (),
    card: str = "h200",
) -> list["KernelsConfig"]:
    """Load the arms of a model, each one's prefill and then its decode.

    Args:
        model: The model's config directory.
        output: Where the records go, or None for the config's own `run.output`.
        phases: The phases to keep of every arm, `data.phase` values.
        arms: The arms to run, or empty for every arm of the model.
        card: The card's token, one of `GPUS` - `b300` loads each arm's `b300_` file.

    Returns:
        The evaluations in the order they run.

    Raises:
        ValueError: The model has no arms listed, an arm it does not have is asked for, or the card has no file for an arm.
    """
    if model not in ARMS:
        raise ValueError(f"no arms for {model!r} - one of {', '.join(ARMS)}")
    if card not in GPUS:
        raise ValueError(f"no card {card!r} - one of {', '.join(GPUS)}")
    known = (*ARMS[model], *EXTRA_ARMS[model], *(PER_PHASE_ARMS[model] if card == "b300" else ()))
    if unknown := [arm for arm in arms if arm not in known]:
        raise ValueError(f"no arm {unknown} for {model!r} - one of {', '.join(known)}")
    from lsa.evaluation.config import KernelsConfig, load

    experiments: list[KernelsConfig] = []
    for arm in arms or ARMS[model]:
        loaded = [
            experiment
            for file in arm_files(model, arm, card)
            for experiment in load(file, arm_overrides(model, arm, output, card), "kernels")
        ]
        kernels = [
            experiment
            for experiment in loaded
            if isinstance(experiment, KernelsConfig) and experiment.data.phase in phases
        ]
        experiments += sorted(kernels, key=lambda experiment: experiment.data.phase != "prefill")
    return experiments


def run_arms(experiments: list["KernelsConfig"]) -> list[str]:
    """Run every evaluation in turn on the job's first task, whatever happens to the ones before.

    Args:
        experiments: The evaluations, as `load_arms` returns them.

    Returns:
        The names of the evaluations that failed - empty when all finished, and nothing on any task but the first.
    """
    from lsa.evaluation.run import run_experiment

    if os.environ.get("SLURM_PROCID", "0") != "0":
        return []
    failed = []
    for experiment in experiments:
        try:
            run_experiment(experiment)
        except Exception:
            traceback.print_exc()
            failed.append(experiment.name)
    print(f"failed {len(failed)} of {len(experiments)}: {failed}", flush=True)
    return failed


def job_parameters(experiments: list["KernelsConfig"], args: argparse.Namespace) -> dict[str, Any]:
    """Return what the one job asks for - one GPU on a node held whole, for the hours its arms add up to.

    Args:
        experiments: The evaluations the job runs.
        args: The parsed command line, whose slurm flags lay over the defaults.

    Returns:
        The submitit parameters.
    """
    from lsa.evaluation.config import Experiment
    from lsa.evaluation.run import bound_to_gpus
    from lsa.evaluation.run import job_parameters as shared_parameters

    everything: list[Experiment] = [*experiments]
    parameters = shared_parameters(everything, 1)
    parameters["timeout_min"] = sum(experiment.hours for experiment in experiments) * 60
    parameters["slurm_exclusive"] = True
    given = slurm.overrides(args)
    extra = given.pop("slurm_additional_parameters", {})
    parameters = {**parameters, **given}
    if extra:
        parameters["slurm_additional_parameters"] = extra
    return bound_to_gpus(parameters)


def main() -> None:
    """Parse the command line, print the evaluations, and submit them as one job unless asked only to print."""
    parser = argparse.ArgumentParser(description="Time every arm of a model's layer in one exclusive job")
    parser.add_argument(
        "model", choices=sorted(ARMS), help="the model's directory under configs/evaluation/kernels"
    )
    parser.add_argument("--output", help="the records root, instead of the configs' own")
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASES,
        help="a phase to run for every arm, repeatable - both when not given",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="an arm to run instead of the model's own, repeatable - an arm's config file name",
    )
    parser.add_argument(
        "--gpu",
        choices=GPUS,
        default="h200",
        help="the card to time on - b300 loads each arm's b300_ config, h200 the arm's own",
    )
    slurm.add_flags(parser)
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")
    experiments = load_arms(args.model, args.output, tuple(args.phase or PHASES), tuple(args.arm), args.gpu)
    for arm in args.arm or ARMS[args.model]:
        print(
            f"arm {arm}: {[os.path.relpath(file, CONFIGS) for file in arm_files(args.model, arm, args.gpu)]}"
        )
    for experiment in experiments:
        print(
            f"{experiment.name}: {experiment.data.phase}, chunk {experiment.data.chunk}, "
            f"contexts {list(experiment.data.contexts)}, backends "
            f"{experiment.data.prefill_backend}/{experiment.data.decode_backend}, "
            f"image {os.path.basename(experiment.run.image)}",
            flush=True,
        )
    parameters = job_parameters(experiments, args)
    print(f"job: {parameters}", flush=True)
    if args.dry_run:
        return

    executor = slurm.executor(
        Launches.SINGLE_JOB, [*env.OFFLINE_HF, *env.EVAL_DATA], args.folder, **parameters
    )
    job = executor.submit(run_arms, experiments)
    print(f"submitted {job.job_id}  {args.model}", flush=True)


if __name__ == "__main__":
    main()
