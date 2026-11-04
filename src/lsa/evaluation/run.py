"""Submit the evaluations one file describes, one slurm job each, and the runner every job calls.

    lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/native_tp8.yaml --dry-run
    lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/native_tp8.yaml
    lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/native_tp8.yaml --partition <partition>

An evaluation is one model read one way, asked one thing, and run one way - its kind's schema says which.
Each kind has one runner, and `RUNNERS` names them, so a runner is the entry its kind's schema is run by.
The server is started, waited for, scored or timed and stopped, and what lands here is the pointer to the
record the run wrote plus the settings that produced it.

Nothing runs on the machine the command is typed on. submitit submits `run_on_first_task` rather than a
script, so a job runs this interpreter and this checkout with nothing to activate, and a sweep becomes an
array job, one task per evaluation, which is what a slurm array is for.

What a job asks slurm for comes from the evaluations rather than from the caller - the GPUs the server
needs, the nodes its pipeline spans, the host threads their affinity requires, and the wall clock the suite
is allowed.
Each was a flag with a default that no served run wanted: `--gpus 1` against a `tp` of 8, `--cpus 8` which
put the server on one socket and showed it four of eight GPUs, and `--hours 4` against a suite measured at
5.3. All three remain as overrides, and `run.tp` and `run.hours` say the same things in the file.

The wall clock is a ceiling and not a reservation - `run_accuracy` tears the server down when scoring ends, so
a job releases its node then rather than at its limit, and the limit only decides when an overrun is killed.

Results are resolved to absolute paths before submission, so an evaluation writes where it was launched
from rather than wherever slurm happens to start it.

Several jobs can drain one run. `--workers N` submits N jobs onto one run id, each starting its own server and
claiming the requests no other job holds, and the same command with `run.id=<id>` on a run that already exists
adds workers to it. Without `--workers` a run is the single job it always was, and `lsa evaluate dispatch`
keeps a reservation full by adding workers to the runs with the most left to do.

Writes are atomic, because a rerun may overwrite `meta.json` while something else is reading it.
"""

import argparse
import copy
import os
import shlex
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lsa import env, io, site, slurm
from lsa.evaluation import gpu, launches, require_extra
from lsa.evaluation.accuracy import pool
from lsa.evaluation.accuracy.records import build_run_id, records_dir
from lsa.evaluation.constants import Evaluation, Launches, Pool, Records, Status
from lsa.sglang.constants import Server
from lsa.sglang.launch import require_image as require_launch_image

if TYPE_CHECKING:
    from lsa.evaluation.config import AccuracyConfig, Experiment, KernelsConfig, ProductionConfig


def run_experiment(experiment: "Experiment", force: bool = False) -> Path:
    """Run one evaluation through the runner its kind names, logging the launch in its run directory.

    The run directory is settled here, once, and its id pinned on the evaluation the runner is handed - an
    id left to the clock would otherwise be derived twice and could name two directories a second apart.

    Args:
        experiment: The evaluation.
        force: Rerun an evaluation that is already written.

    Returns:
        The directory the evaluation wrote into.
    """
    require_model(experiment)
    runner: Callable[[Any, bool], Path] = RUNNERS[experiment.kind]
    pinned = copy.deepcopy(experiment)
    # The card is read from the node, and the run's name ends in it
    pinned.run.gpu = gpu.settle(experiment.run.gpu)
    out = settle_launch(pinned).out
    pinned.run.id = out.name
    out.mkdir(parents=True, exist_ok=True)
    worker = pool.worker_name() if getattr(pinned.run, "pool", False) else None
    with launches.launched(out, worker):
        return runner(pinned, force)


def require_model(experiment: "Experiment") -> None:
    """Refuse an evaluation whose config names no model, before anything is served for it.

    A base or an overlay - a mesh, a ladder or a length range laid over a configuration - names none of its
    own, and run alone it would otherwise serve whatever a schema default happened to be under the base's
    serve args.

    Args:
        experiment: The evaluation.

    Raises:
        ValueError: `model.name` is empty.
    """
    if not experiment.model.name:
        raise ValueError(
            f"{experiment.name}: the config names no model - it is a base or an overlay, so run the "
            "configuration that names it under `defaults`, or set model.name"
        )


def require_image(experiment: "Experiment") -> None:
    """Refuse a submission whose site names no server image, before a job is queued for it.

    Args:
        experiment: The evaluation.

    Raises:
        NotNamed: `run.image` is empty.
    """
    require_launch_image(experiment.run.image)


def require_output(experiment: "Experiment") -> None:
    """Refuse an evaluation whose config names no records root, before anything writes for it.

    The root has no default, because an empty one would land a run in the shared root - publishing it -
    without anyone having asked for that. Each tree's base names it.

    Args:
        experiment: The evaluation.

    Raises:
        ValueError: `run.output` is empty.
    """
    if not experiment.run.output:
        raise ValueError(
            f"{experiment.name}: the config names no records root - set run.output, which each tree's base "
            "names as `results` or `${root:evaluation}`"
        )


@dataclass(frozen=True)
class Launch:
    """What every runner settles before it writes anything - the settings, the run's name and its directory.

    The id names the directory, so it is settled before the server starts and then pinned, and a rerun of
    one run lands in the directory the first one wrote.
    """

    # The indexer settings the plugin is armed with, or None for the checkpoint's own indexer
    settings: dict[str, Any] | None
    # `run.name`, which the record is grouped by
    name: str
    records: Path
    out: Path


def settle_launch(experiment: "Experiment") -> Launch:
    """Settle one evaluation's settings, name and run directory.

    The name is the config's with its card's token last, and the directory is `run.id` when the config pins
    one, and `<name>__<stamp>[-<tag>]` otherwise.

    Args:
        experiment: The evaluation, with `run.gpu` set.

    Returns:
        The launch.

    Raises:
        ValueError: `run.output` is empty, or `run.gpu` is and the run has no id to continue.
    """
    require_output(experiment)
    run = experiment.run
    root = Path(records_dir(experiment.kind, run.output)).resolve()
    # A run continued by its id is in a directory already, so the card matters to its name only once a node is reached
    name = run.launch_name() if run.gpu or not run.id else run.name
    return Launch(
        settings=experiment.model.settings,
        name=name,
        records=root,
        out=root / (run.id or build_run_id(name, run.tag)),
    )


def written_settings(launch: Launch) -> Path | None:
    """Write a launch's indexer settings into its run directory, where the record quotes them.

    Args:
        launch: The launch.

    Returns:
        The file, or None for the checkpoint's own indexer.
    """
    from lsa.sglang.launch import write_settings

    if launch.settings is None:
        return None
    return write_settings(launch.settings, (launch.out / "indexer.json").resolve())


def run_kernels(experiment: "KernelsConfig", force: bool) -> Path:
    """Time one attention layer of this evaluation's model through the harness.

    Nothing is served. The harness loads the truncated checkpoint itself, inside the same container image
    the server uses, so a kernel run and a served run measure the same build of the same kernels.

    A prefill and a decode run of one configuration are two configs, so two names and two directories, and
    a records reader can hold both. The ladder resumes into that directory, which is what makes a job that ran out
    of wall clock finishable rather than restartable.

    The record's `run.name` is the run's own name, as every other kind's is. The composition reads a path with
    no indexer to describe, `native` or `full`, by the record's setup and not by its name.

    Args:
        experiment: The evaluation.
        force: Measure every context again, discarding the samples already there.

    Returns:
        The run directory.

    Raises:
        ValueError: The data block names no checkpoint or no prompt to measure over.
    """
    from lsa.evaluation.performance.kernels.harness import run_kernels as measure_ladder

    if not experiment.data.checkpoint:
        raise ValueError("data.checkpoint must name the truncated cut of the model holding the layer")
    if not experiment.data.prompt:
        raise ValueError("data.prompt must name the prompt file the harness prefills from")

    launch = settle_launch(experiment)
    fresh_samples(launch.out, force)
    written = measure_ladder(experiment, written_settings(launch), launch.name, launch.out)

    io.write_json(launch.out / Records.META, launch_meta(experiment, launch.name, written))
    print(f"{experiment.name}: {written}", flush=True)
    return launch.out


def fresh_samples(out: Path, force: bool) -> None:
    """Create a ladder's run directory, emptying its samples first when every point is to be measured again.

    Args:
        out: The run directory.
        force: Discard the samples already there.
    """
    if force and (out / "samples").is_dir():
        shutil.rmtree(out / "samples")
    out.mkdir(parents=True, exist_ok=True)


def launch_meta(experiment: "Experiment", label: str, record: Path) -> dict[str, Any]:
    """Record what produced a run, and where the record it wrote is.

    The scores are not copied. That record is the artifact its table and any records reader read, and a second
    copy here would be a second thing to keep true.

    Args:
        experiment: The evaluation.
        label: The configuration label the record is grouped by.
        record: The record's path, inside the run directory.

    Returns:
        The metadata block, ready to write.
    """
    # Here rather than at the top, so a bare install meets `require_extra` before hydra is missing
    from lsa import config_loader
    from lsa.evaluation.config import serve_argv

    provenance = config_loader.run_provenance(experiment)
    # The server's flags are kept as the command line they became, as every record has kept them
    # => a records reader reads old and new runs alike
    provenance["run"]["run"] = {
        ("serve_args" if key == "serve" else key): serve_argv(value) if key == "serve" else value
        for key, value in provenance["run"]["run"].items()
    }
    return {
        **provenance,
        "backend": "sglang",
        "name": label,
        "run_id": record.parent.name,
        "record": str(record),
    }


def run_production(experiment: "ProductionConfig", force: bool) -> Path:
    """Time a served model's requests - first token and every token after it - through sglang's own client.

    The server is started exactly as a scored run starts it, and waited on the same way, so the plugin's
    arming check stands between a production record and a server running the checkpoint's own indexer.

    A rerun resumes into its directory and keeps every length already measured, so pinning the id is how one
    record spans a one-node and a two-node server.

    Args:
        experiment: The evaluation.
        force: Measure every length again, discarding the samples already there.

    Returns:
        The run directory.

    Raises:
        RuntimeError: The ladder stopped before its last length - the record holds the ones it reached.
    """
    from lsa.evaluation.performance.production.measure import run_ladder
    from lsa.sglang.launch import serve

    launch = settle_launch(experiment)
    out = launch.out
    fresh_samples(out, force)
    checkpoint = experiment.model.checkpoint
    with serve(experiment.server(), checkpoint, written_settings(launch), out / "server.log") as url:
        missing = run_ladder(experiment, url, launch.name, out)

    io.write_json(out / Records.META, launch_meta(experiment, launch.name, out / Records.RECORD))
    print(f"{experiment.name}: {out / Records.RECORD}", flush=True)
    if missing:
        raise RuntimeError(f"{out}: the ladder stopped at {missing[0]} - {missing} not measured")
    return out


def run_accuracy(experiment: "AccuracyConfig", force: bool) -> Path:
    """Score a served model with the accuracy suite.

    Nothing of the `evaluation` extra is reached here - the model loads in a container and the suite runs in the
    interpreter `run.python` names, so a served evaluation runs from a bare `uv sync`.

    The launch evidence and the scores share one directory, named by the run id. The run id is derived from
    the config, so a rerun lands in the run directory the first run wrote and the suite resumes it - which is
    what makes an interrupted evaluation finishable. `force` carries through to the suite as `--no-resume`,
    so a forced rerun scores every sample again into that same directory rather than restarting the server
    and scoring nothing.

    A run directory is one suite. Running a second suite that derives the same id raises rather than
    resuming, because the gate is the directory and a silent skip reports success having scored nothing.

    Args:
        experiment: The evaluation.
        force: Rerun an evaluation whose meta already exists, scoring every sample again.

    Returns:
        The run directory.

    A pooled job is one worker of the run, and it scores only what no other worker holds. It keeps a server log of
    its own, because several servers write beside each other, and it never discards samples.

    Raises:
        ValueError: The run directory holds a run of a different suite.
        ValueError: The generation block leaves out something every request needs.
        ValueError: A pooled job is asked to score every sample again.
    """
    from lsa.evaluation.accuracy import job
    from lsa.sglang.launch import serve, write_settings

    data = experiment.data
    worker = pool.worker_name() if experiment.run.pool else None
    if worker and force:
        raise ValueError(
            f"{experiment.name}: a pooled run never scores samples again - give the rerun a run id of its own"
        )
    launch = settle_launch(experiment)
    out = launch.out

    meta = out / Records.META
    if meta.exists() and not force:
        quoted = io.read_json(meta).get("run", {})
        done = quoted.get("data", {}).get("suite", "")
        if done and done != data.suite:
            raise ValueError(
                f"{out} already holds a {done} run and this one is {data.suite} - "
                f"a run directory is one suite, so give this one a run tag of its own"
            )
    if not force and _scored_everything(experiment, out):
        print(f"{experiment.name}: complete - nothing to run", flush=True)
        return out

    generation = experiment.model.generation
    missing = generation.missing(data.tasks)
    if missing:
        raise ValueError(
            f"{experiment.name}: model.generation leaves {missing} unset - the yaml that names the model "
            "says how it is sampled, and nothing fills in for it"
        )
    out.mkdir(parents=True, exist_ok=True)
    settings = written_settings(launch)
    sampling = write_settings(generation.request(), (out / "generation.json").resolve())
    log = out / (f"server.{worker}.log" if worker else "server.log")
    with serve(experiment.server(), experiment.model.checkpoint, settings, log) as url:
        # `run.port` may be 0, so the suite is pointed at the port the server took
        served = replace(experiment, run=replace(experiment.run, port=int(url.rsplit(":", 1)[1])))
        job.score(served, launch.records, out.name, force, sampling, worker)

    io.write_json(meta, launch_meta(experiment, launch.name, out / Records.RECORD))
    print(f"{experiment.name}: {out}", flush=True)
    return out


def _scored_everything(experiment: "AccuracyConfig", out: Path) -> bool:
    """Say whether a run directory's record already scored every task and length this evaluation asks for.

    A one-node run of `full` leaves the lengths past 1M unscored, and a two-node run restricted to them and
    pinned to the same run id fills them in - so a finished launch alone does not make a run complete.

    Args:
        experiment: The evaluation.
        out: Its run directory.

    Returns:
        True when every selected task's record entry scored all of its requests.
    """
    from lsa.evaluation.accuracy.data import suites

    path = out / Records.RECORD
    if not path.exists():
        return False
    tasks = io.read_json(path).get("tasks", {})
    data = experiment.data
    asked = suites.select(data.suite, data.tasks or None, data.lengths or None)
    return all(plan.task in tasks and tasks[plan.task]["scored"] >= tasks[plan.task]["n"] for plan in asked)


# The runner each kind is run by - the entry a kind's schema is handed to
RUNNERS: dict[str, Callable[[Any, bool], Path]] = {
    Evaluation.ACCURACY: run_accuracy,
    Evaluation.KERNELS: run_kernels,
    Evaluation.PRODUCTION: run_production,
}


def describe(experiment: "Experiment", parameters: Mapping[str, Any] | None = None) -> str:
    """Return what an evaluation will run - one line, and with its job's parameters what a dry run resolves.

    Args:
        experiment: The evaluation, as loaded.
        parameters: What its job asks slurm for, the command line's flags already laid over, or None for the
            summary line alone.

    Returns:
        The summary line, followed for a dry run by one indented line each for the tasks and lengths of a scored
        run, the server's command as the container runs it, and the job's resources.
    """
    from lsa.evaluation.accuracy.data import suites
    from lsa.evaluation.config import AccuracyConfig, KernelsConfig, ProductionConfig
    from lsa.sglang.launch import container_command, launch_line, with_settings

    head = f"{experiment.name}: attention {experiment.model.attention}"
    if isinstance(experiment, KernelsConfig):
        lines = [f"{head}, kernels {experiment.data.phase}, contexts {experiment.data.contexts}"]
    elif isinstance(experiment, ProductionConfig):
        lines = [f"{head}, production, lengths {experiment.data.lengths}"]
    else:
        lines = [f"{head}, suite {experiment.data.suite}"]
    if parameters is None:
        return lines[0]
    if isinstance(experiment, AccuracyConfig):
        data = experiment.data
        plans = suites.select(data.suite, data.tasks or None, data.lengths or None)
        shown = ", ".join(plan.task for plan in plans[: Evaluation.DRY_RUN_TASKS])
        hidden = len(plans) - Evaluation.DRY_RUN_TASKS
        more = f", and {hidden} more" if hidden > 0 else ""
        lines.append(f"  tasks ({len(plans)}): {shown}{more}")
        lengths = sorted({plan.length for plan in plans if plan.length})
        if lengths:
            lines.append(f"  lengths: {lengths}")
    if not isinstance(experiment, KernelsConfig):
        server = experiment.server()
        if experiment.model.settings is not None:
            server = replace(server, serve_args=with_settings(server.serve_args, experiment.model.settings))
        try:
            argv = container_command(server, launch_line(server, experiment.model.checkpoint))
        except site.NotNamed as unnamed:
            # A dry run is how a fresh clone finds out what it would run, and its site names no image or root yet
            # => say what is missing here, and leave the refusal to the submission
            lines.append(f"  server: (not resolved in site {site.current()} - {unnamed})")
        else:
            lines.append(f"  server: {shlex.join(argv)}")
    lines.append("  resources: " + ", ".join(f"{key} {value}" for key, value in parameters.items()))
    return "\n".join(lines)


def gpu_count(experiments: list["Experiment"]) -> int:
    """Return the GPUs one job needs, from the evaluations rather than from the caller.

    A served evaluation reserves `run.tp` GPUs for the server it starts, so the config already says the
    number and the command line asking for it again is a second copy that can disagree.

    An array shares one `gpus_per_node`, so a sweep whose points want different meshes has no single answer
    and says so here rather than starving the larger points of devices.

    Args:
        experiments: The evaluations about to be submitted.

    Returns:
        The tensor-parallel size every evaluation shares.

    Raises:
        ValueError: The evaluations ask for more than one mesh.
    """
    wanted = sorted({experiment.run.tp for experiment in experiments})
    if len(wanted) > 1:
        raise ValueError(
            f"one array job reserves one GPU count and these experiments want {wanted} - "
            "split the sweep by run.tp, or pass --gpus to force one"
        )
    return wanted[0]


def node_count(experiments: list["Experiment"]) -> int:
    """Return the nodes one job needs, one per pipeline stage of the server it starts.

    Args:
        experiments: The evaluations about to be submitted.

    Returns:
        The pipeline size every evaluation shares.

    Raises:
        ValueError: The evaluations ask for more than one pipeline size.
    """
    wanted = sorted({experiment.run.pp for experiment in experiments})
    if len(wanted) > 1:
        raise ValueError(f"one array job reserves one node count and these experiments want {wanted}")
    return wanted[0]


def run_on_first_task(run: "Experiment", force: bool = False) -> Path | None:
    """Run one evaluation on the job's first task, and nothing on the others.

    submitit starts the function once per node, and a job spanning nodes needs only one launcher - it
    starts the server's step on every node itself. A second launcher would start a second server.

    Args:
        run: The evaluation.
        force: Rerun an evaluation that is already written.

    Returns:
        The directory the evaluation wrote into, or None on every task but the first.
    """
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return None
    return run_experiment(run, force)


def job_parameters(experiments: list["Experiment"], gpus: int | None = None) -> dict[str, Any]:
    """Return the submitit parameters one job asks for, from the evaluations rather than from the caller.

    The threads follow the GPUs, and that is not a throughput choice. The server runs as an `srun` step
    overlapping the allocation, and an overlapping step sees the GPUs affine to the cores its job holds - too
    few and the step is confined to one socket and the server dies on a device it was told to use but
    cannot see.

    Args:
        experiments: The evaluations about to be submitted.
        gpus: GPUs per job, or None for `run.tp`. A sweep over the mesh has no single count and needs one.

    Returns:
        `run.tp` GPUs on a node per pipeline stage, `Server.CPUS_PER_GPU` threads and `Server.MEM_GB_PER_GPU`
        GB for each GPU, and the minutes the longest evaluation is allowed.

    Raises:
        ValueError: The evaluations disagree on the pipeline, or on the mesh with no GPU count given.
    """
    gpus = gpu_count(experiments) if gpus is None else gpus
    return {
        "gpus_per_node": gpus,
        "cpus_per_task": gpus * Server.CPUS_PER_GPU,
        "mem_gb": max(Server.MEM_GB_PER_GPU, gpus * Server.MEM_GB_PER_GPU),
        "timeout_min": hour_limit(experiments) * 60,
        "nodes": node_count(experiments),
    }


def hour_limit(experiments: list["Experiment"]) -> int:
    """Return the wall clock one job may have, from what the evaluations ask for.

    An evaluation's own `hours` wins, and 0 - the default - takes the ceiling its suite or its phase is
    given. So a run that is slower than its kind usually is raises the number in the file that describes
    it, and every other run says nothing at all.

    The longest in the array wins, because an array shares one limit and a ceiling that is too high costs a
    little backfill priority while one that is too low loses the run.

    Args:
        experiments: The evaluations about to be submitted.

    Returns:
        The hours the longest of them is allowed.
    """
    return max(experiment.hours for experiment in experiments)


def bound_to_gpus(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return job parameters whose threads are taken on the sockets of the job's GPUs.

    The server's overlapping step sees only the GPUs affine to the job's threads. A job holding part of a node
    can be given GPUs on both sockets and threads on one, and its step then sees one GPU of two.
    Measured: GPUs 0 and 4 with threads 0-31 left the step `CUDA_VISIBLE_DEVICES=0`, and a tensor
    parallel 2 server died on rank 1's device.
    One GPU sits on one socket and its step sees it whatever threads the job holds, and on the B300 nodes binding
    took every free thread of the GPU's socket for a one-GPU job - 96 for 16 asked - so
    a node ran two one-GPU jobs. So a one-GPU job is left unbound.

    Args:
        parameters: What each job asks for, as submitit parameters.

    Returns:
        The same parameters, with `--gres-flags=enforce-binding` among the sbatch options for more than one GPU.
    """
    if parameters.get("gpus_per_node") == 1:
        return dict(parameters)
    return {
        **parameters,
        "slurm_additional_parameters": {
            **parameters.get("slurm_additional_parameters", {}),
            "gres_flags": "enforce-binding",
        },
    }


def submit(
    experiments: list["Experiment"],
    parameters: Mapping[str, Any] | None = None,
    folder: str | Path | None = None,
    force: bool = False,
) -> list[Any]:
    """Submit one job per evaluation, print each one's id, and return them without waiting.

    `run_on_first_task` is submitted directly - it is a module-level function taking the evaluation and the
    force flag, which is exactly what an array maps over.

    Args:
        experiments: The evaluations, as `config.load` returns them.
        parameters: What each job asks for, as submitit parameters, or None for what the evaluations need.
        folder: Where submitit writes each job's output, or None for the shared log root.
        force: Rerun an evaluation that is already written, scoring every sample again.

    Returns:
        One submitit job per evaluation, in the order they were given.

    Raises:
        ValueError: An evaluation names no model or no records root, or the parameters are being derived
            and the evaluations do not agree on them.
    """
    for experiment in experiments:
        require_model(experiment)
        require_output(experiment)
        require_image(experiment)

    executor = slurm.executor(
        Launches.SINGLE_JOB,
        [*env.OFFLINE_HF, *env.EVAL_DATA],
        folder,
        **bound_to_gpus(parameters or job_parameters(experiments)),
    )
    for experiment in experiments:
        experiment.run.output = str(Path(experiment.run.output).resolve())

    jobs: list[Any] = executor.map_array(run_on_first_task, experiments, [force] * len(experiments))
    for job, experiment in zip(jobs, experiments, strict=True):
        print(f"submitted {job.job_id}  {experiment.name}", flush=True)
    return jobs


def require_no_live_single_job(out: Path) -> None:
    """Refuse to add workers to a run a single job is still running, which would send what it sends.

    A single job scores the whole suite from the files it read at its start and never looks at a claim, so a
    worker beside it would send the same requests twice. Once it ends the run is an ordinary one to add to.

    Args:
        out: The run directory.

    Raises:
        ValueError: A launch of the run that is not a pool worker is running or queued in slurm.
    """
    singles = [launch for launch in launches.read(out) if not launch.get("worker")]
    jobs = launches.open_jobs(singles)
    states = launches.accounting(jobs)
    live = [
        job
        for job in jobs
        if Status.SLURM.get(states.get(job, {}).get("State", "").split(" ")[0]) == Status.IN_PROGRESS
    ]
    if live:
        raise ValueError(
            f"{out} has a single-job launch still running or queued (job {', '.join(live)}), which does not "
            "claim requests and would be sent them again - add workers once it ends"
        )


def submit_pool(
    experiment: "AccuracyConfig",
    count: int,
    source: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    folder: str | Path | None = None,
) -> list[Any]:
    """Submit `count` workers onto one run, creating the run if it is not there.

    The run id is settled here, once, so every worker lands in one directory, and the run keeps how to submit
    another worker so a dispatcher can add one without being told the config again.

    Args:
        experiment: The evaluation, with `run.pool` set by this call.
        count: How many workers to submit.
        source: Where the config came from - its `config` path and the `overrides` laid over it - and which of
            the file's runs this is, as `index`.
        parameters: What each job asks for, as submitit parameters, or None for what the evaluation needs.
        folder: Where submitit writes each job's output, or None for the shared log root.

    Returns:
        One submitit job per worker.

    Raises:
        ValueError: The evaluation names no model or no records root, or a single job is still running the run.
    """
    require_model(experiment)
    require_output(experiment)
    require_image(experiment)
    experiment.run.output = str(Path(experiment.run.output).resolve())
    out = settle_launch(experiment).out
    require_no_live_single_job(out)
    experiment.run.id = out.name
    experiment.run.pool = True
    io.write_json(
        out / Pool.DIR / Pool.SUBMISSION,
        {**source, "output": experiment.run.output, "id": out.name, "kind": experiment.kind},
    )
    parameters = bound_to_gpus(parameters or job_parameters([experiment]))
    # A worker still queued has not launched, so its comment is the only thing that says where its run is
    parameters["slurm_additional_parameters"] = {
        **parameters.get("slurm_additional_parameters", {}),
        "comment": str(out),
    }
    executor = slurm.executor(
        f"{Launches.POOL_JOB}{out.name}",
        [*env.OFFLINE_HF, *env.EVAL_DATA],
        folder,
        **parameters,
    )
    jobs: list[Any] = executor.map_array(run_on_first_task, [experiment] * count, [False] * count)
    for job in jobs:
        print(f"submitted {job.job_id}  {experiment.name} worker of {out.name}", flush=True)
    return jobs


def main(kind: str) -> None:
    """Parse the command line and submit every evaluation of one kind the file describes.

    Args:
        kind: The kind the file is read as, one of `Evaluation.KINDS` - the command names it.
    """
    parser = argparse.ArgumentParser(description=f"Submit a file's {kind} evaluations to slurm")
    parser.add_argument("config", help=f"path to the {kind} evaluation yaml")
    slurm.add_flags(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun an evaluation that is already written, scoring every sample again",
    )
    if kind == Evaluation.ACCURACY:
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="submit this many jobs onto the run, each with a server of its own, claiming the requests no "
            "other holds - with run.id=<id> on a run that exists, this adds workers to it",
        )
        parser.add_argument(
            "--pool",
            action="store_true",
            help="submit the one job as a worker of the run's pool, as a run already being drained is "
            "added to - implied by --workers above 1 and by a run directory that has been pooled",
        )
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")
    if kind == Evaluation.ACCURACY:
        # The scorer runs in the job from this checkout's environment - checked here, not seconds into the job
        require_extra("aiohttp")
    from lsa.evaluation.config import load

    experiments = load(args.config, args.overrides, kind)
    # Only an accuracy run is pooled, and every evaluation needs an entry
    # => the dry run and the submission below walk both lists together
    pooled = [kind == Evaluation.ACCURACY and pooled_run(experiment, args) for experiment in experiments]
    for experiment in experiments:
        require_model(experiment)
        if not args.dry_run:
            print(describe(experiment), flush=True)
    if any(pooled) and args.force:
        raise SystemExit(
            "--force scores every sample again, which a pooled run never does - give the rerun a run id of its own"
        )
    parameters = {**job_parameters(experiments, args.gpus), **slurm.overrides(args)}
    if args.dry_run:
        for experiment, is_pooled in zip(experiments, pooled, strict=True):
            print(describe(experiment, parameters), flush=True)
            if is_pooled:
                print(f"{experiment.name}: would submit {args.workers} workers onto one run", flush=True)
        return

    if not any(pooled):
        submit(experiments, parameters, args.folder, args.force)
        return
    for i, (experiment, is_pooled) in enumerate(zip(experiments, pooled, strict=True)):
        source = {"config": str(Path(args.config).resolve()), "overrides": args.overrides, "index": i}
        if is_pooled:
            submit_pool(cast("AccuracyConfig", experiment), args.workers, source, parameters, args.folder)
        else:
            submit([experiment], parameters, args.folder, args.force)


def pooled_run(experiment: "Experiment", args: argparse.Namespace) -> bool:
    """Say whether an evaluation is submitted as workers of a pool.

    Args:
        experiment: The accuracy evaluation.
        args: The parsed command line, with `--workers` and `--pool`.

    Returns:
        True for `--pool`, for more than one worker, and for a run directory that has been pooled before - a
        run drained by a pool is never continued by a single job beside the claims.
    """
    if args.pool or args.workers > 1:
        return True
    run = experiment.run
    # A run with no id is a new one, and a new one is never pooled before
    if not (run.output and run.id):
        return False
    return pool.is_pooled(Path(records_dir(experiment.kind, run.output)).resolve() / run.id)
