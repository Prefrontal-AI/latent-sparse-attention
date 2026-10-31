"""What a job asks slurm for, which the experiments say rather than the caller.

Each of the three was a flag with a default no served run wanted, and the CPU one is not cosmetic: the
server runs as an overlapping `srun` step and sees only the GPUs affine to the cores its job holds, so a
default of 8 threads showed a `tp` of 8 four devices and died after loading the model.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsa import io
from lsa.evaluation import config as experiment
from lsa.evaluation import launches
from lsa.evaluation.constants import Evaluation, Kernels, Suite
from lsa.evaluation.run import (
    RUNNERS,
    bound_to_gpus,
    gpu_count,
    hour_limit,
    job_parameters,
    launch_meta,
    node_count,
    run_experiment,
    run_on_first_task,
    settle_launch,
    submit,
    submit_pool,
)
from lsa.sglang.constants import Server

MINIMAL = """
model:
  attention: lsa
  indexer:
    window_size: 512
    sink_size: 4
    top_k: 2048
    index_dim: 64
  serving:
    projections: /maps/glm53_flash.pt
data:
  suite: full
run:
  name: smoke
  image: /images/sglang.img
  tp: {tp}
"""


def write(tmp_path, text, name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def load(tmp_path, tp=8, extra="", name="experiment.yaml"):
    return experiment.load(write(tmp_path, MINIMAL.format(tp=tp) + extra, name), kind=Evaluation.ACCURACY)


def test_the_gpus_are_the_config_s_own_tp(tmp_path):
    assert gpu_count(load(tmp_path, tp=8)) == 8
    assert gpu_count(load(tmp_path, tp=2, name="two.yaml")) == 2


def test_a_sweep_agreeing_on_the_mesh_is_one_count(tmp_path):
    runs = load(tmp_path, extra="\nsweep:\n  model.indexer.top_k: [1024, 2048]\n")
    assert len(runs) == 2
    assert gpu_count(runs) == 8


def test_a_sweep_over_the_mesh_has_no_single_answer(tmp_path):
    runs = load(tmp_path, extra="\nsweep:\n  run.tp: [4, 8]\n")
    with pytest.raises(ValueError, match=r"\[4, 8\]"):
        gpu_count(runs)


def test_a_job_spans_a_node_per_pipeline_stage(tmp_path):
    assert node_count(load(tmp_path)) == 1
    runs = load(tmp_path, extra="  pp: 2\n", name="pp.yaml")
    assert node_count(runs) == 2
    assert gpu_count(runs) == 8


def test_a_sweep_over_the_pipeline_has_no_single_answer(tmp_path):
    runs = load(tmp_path, extra="\nsweep:\n  run.pp: [1, 2]\n")
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        node_count(runs)


def test_only_the_first_task_launches(monkeypatch):
    """submitit starts the function on every node, and a second launcher would start a second server"""
    launched = []
    monkeypatch.setattr("lsa.evaluation.run.run_experiment", lambda run, force: launched.append(run) or "dir")
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert run_on_first_task("run") is None
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert run_on_first_task("run") == "dir"
    assert launched == ["run"]


def test_the_cpus_follow_the_gpus(tmp_path):
    """Eight of 128 threads is one socket, and one socket of this node carries four of the eight GPUs"""
    assert job_parameters(load(tmp_path, tp=8))["cpus_per_task"] == 8 * Server.CPUS_PER_GPU
    assert job_parameters(load(tmp_path, tp=1, name="one.yaml"))["cpus_per_task"] == Server.CPUS_PER_GPU


def test_the_host_memory_follows_the_gpus(tmp_path):
    """A job asked for the whole node's 1200 GB, so two workers filled a node's memory with its GPUs idle"""
    for tp in (1, 2, 4, 8):
        parameters = job_parameters(load(tmp_path, tp=tp, name=f"tp{tp}.yaml"))
        assert parameters["mem_gb"] == tp * Server.MEM_GB_PER_GPU
    assert job_parameters(load(tmp_path, tp=1, name="one.yaml"))["mem_gb"] == 64


def test_the_cpus_follow_a_gpu_count_given_for_a_sweep_over_the_mesh(tmp_path):
    """Deriving the threads from the config used to refuse the very sweep `--gpus` exists to submit"""
    parameters = job_parameters(load(tmp_path, extra="\nsweep:\n  run.tp: [4, 8]\n"), gpus=8)
    assert (parameters["gpus_per_node"], parameters["cpus_per_task"]) == (8, 8 * Server.CPUS_PER_GPU)


def test_the_ceiling_is_the_suite_s_own(tmp_path):
    for suite, hours in Suite.HOURS.items():
        text = MINIMAL.format(tp=8).replace("suite: full", f"suite: {suite}")
        assert (
            hour_limit(experiment.load(write(tmp_path, text, f"{suite}.yaml"), kind=Evaluation.ACCURACY))
            == hours
        )


def test_the_file_may_raise_its_own_ceiling(tmp_path):
    assert hour_limit(load(tmp_path, extra="  hours: 20\n")) == 20


def test_zero_hours_falls_back_to_the_suite(tmp_path):
    assert hour_limit(load(tmp_path, extra="  hours: 0\n")) == Suite.HOURS["full"]


def test_a_ceiling_covers_the_longest_suite_in_an_array(tmp_path):
    runs = load(tmp_path, extra="\nsweep:\n  data.suite: [smoke, full]\n")
    assert hour_limit(runs) == Suite.HOURS["full"]


def test_a_negative_ceiling_is_refused(tmp_path):
    with pytest.raises(ValueError, match="run.hours"):
        load(tmp_path, extra="  hours: -1\n")


KERNELS = """
model:
  attention: native
data:
  phase: {phase}
  checkpoint: /models/cut
  prompt: /prompts/p.txt
run:
  name: kernels
  image: /images/sglang.img
  tp: 1
"""


def load_kernels(tmp_path, phase, name):
    return experiment.load(write(tmp_path, KERNELS.format(phase=phase), name), kind=Evaluation.KERNELS)


def test_a_kernel_run_takes_its_phase_s_ceiling_rather_than_a_suite_s(tmp_path):
    for phase, hours in Kernels.HOURS.items():
        assert hour_limit(load_kernels(tmp_path, phase, f"kernels_{phase}.yaml")) == hours


def test_a_kernel_run_reserves_the_one_card_its_cut_fits_on(tmp_path):
    runs = load_kernels(tmp_path, "prefill", "kernels_tp.yaml")
    assert gpu_count(runs) == 1
    assert job_parameters(runs)["cpus_per_task"] == Server.CPUS_PER_GPU


def test_a_job_spans_the_pipeline_and_takes_the_suite_s_ceiling(tmp_path):
    parameters = job_parameters(load(tmp_path, tp=4, extra="  pp: 2\n", name="pp.yaml"))
    assert (parameters["gpus_per_node"], parameters["nodes"]) == (4, 2)
    assert parameters["timeout_min"] == Suite.HOURS["full"] * 60


def test_a_launch_is_logged_in_the_directory_the_runner_writes(tmp_path, monkeypatch):
    """The id is pinned once, so the log and the runner name one directory however the clock turns"""
    (run,) = experiment.load(
        write(tmp_path, MINIMAL.format(tp=8)),
        ["model.name=zai-org/Fake-Model", f"run.output={tmp_path / 'results'}"],
        kind=Evaluation.ACCURACY,
    )
    handed = []

    def runner(pinned, force):
        handed.append(pinned)
        return settle_launch(pinned).out

    monkeypatch.setitem(RUNNERS, Evaluation.ACCURACY, runner)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_USER", "alice")
    out = run_experiment(run)
    assert handed[0].run.id == out.name
    assert run.run.id == ""
    (logged,) = launches.read(out)
    assert (logged["job"], logged["user"], logged["outcome"]) == ("4242", "alice", "completed")


def test_a_runner_that_raises_is_logged_as_failed(tmp_path, monkeypatch):
    (run,) = experiment.load(
        write(tmp_path, MINIMAL.format(tp=8)),
        ["model.name=zai-org/Fake-Model", "run.id=pinned", f"run.output={tmp_path / 'results'}"],
        kind=Evaluation.ACCURACY,
    )

    def runner(pinned, force):
        raise RuntimeError("server never came up")

    monkeypatch.setitem(RUNNERS, Evaluation.ACCURACY, runner)
    with pytest.raises(RuntimeError, match="never came up"):
        run_experiment(run)
    (logged,) = launches.read(settle_launch(replace_gpu(run)).out)
    assert logged["error"] == "RuntimeError: server never came up"


STAMPED = r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}"


def replace_gpu(run, gpu="h200"):
    run.run.gpu = gpu
    return run


def test_a_run_directory_is_the_run_name_the_card_and_the_start_stamp(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n")
    launch = settle_launch(replace_gpu(run))
    assert re.fullmatch(f"smoke_h200__{STAMPED}", launch.out.name)
    assert launch.name == "smoke_h200"


def test_the_card_closes_the_name_whichever_way_the_config_spells_it(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n")
    assert settle_launch(replace_gpu(run, "NVIDIA B300 SXM6 AC")).name == "smoke_b300"


def test_a_launch_that_knows_no_card_is_refused(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n")
    with pytest.raises(ValueError, match="run.gpu is empty"):
        settle_launch(run)


def test_a_run_continued_by_its_id_needs_no_card_to_find_its_directory(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n  id: pinned\n")
    assert settle_launch(run).out.name == "pinned"


def test_a_tag_closes_the_run_id_and_a_pinned_id_replaces_it(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n  tag: rerun\n")
    replace_gpu(run)
    assert re.fullmatch(f"smoke_h200__{STAMPED}-rerun", settle_launch(run).out.name)
    run.run.id = "pinned"
    launch = settle_launch(run)
    assert (launch.out.name, launch.name) == ("pinned", "smoke_h200")


def test_a_launch_meta_names_the_record_by_the_run_name(tmp_path):
    [run] = load(tmp_path, extra="  output: results\n")
    launch = settle_launch(replace_gpu(run))
    meta = launch_meta(run, launch.name, launch.out / "record.json")
    assert meta["name"] == "smoke_h200"
    assert meta["run_id"] == launch.out.name


def test_a_launch_meta_keeps_the_server_s_flags_as_the_command_line_they_became(tmp_path):
    [run] = load(
        tmp_path,
        extra="  output: results\n  serve:\n    page-size: 1\n    enable-metrics: true\n    watchdog-timeout: null\n",
    )
    meta = launch_meta(run, "smoke_h200", tmp_path / "record.json")
    assert meta["run"]["run"]["serve_args"] == ["--page-size", "1", "--enable-metrics"]
    assert "serve" not in meta["run"]["run"]


def test_a_launch_reads_the_card_from_the_node_and_pins_it_for_the_runner(tmp_path, monkeypatch):
    (run,) = experiment.load(
        write(tmp_path, MINIMAL.format(tp=8)),
        ["model.name=zai-org/Fake-Model", f"run.output={tmp_path / 'results'}"],
        kind=Evaluation.ACCURACY,
    )
    monkeypatch.setattr("lsa.evaluation.gpu.detect", lambda: "NVIDIA B300 SXM6 AC")
    handed = []
    monkeypatch.setitem(RUNNERS, Evaluation.ACCURACY, lambda pinned, force: handed.append(pinned) or Path())

    run_experiment(run)

    assert handed[0].run.gpu == "NVIDIA B300 SXM6 AC"
    assert re.fullmatch(f"smoke_b300__{STAMPED}", handed[0].run.id)
    assert run.run.gpu == ""


def test_a_launch_on_another_card_than_the_one_submitted_for_is_refused(tmp_path, monkeypatch):
    (run,) = experiment.load(
        write(tmp_path, MINIMAL.format(tp=8)),
        ["model.name=zai-org/Fake-Model", "run.gpu=b300", f"run.output={tmp_path / 'results'}"],
        kind=Evaluation.ACCURACY,
    )
    with pytest.raises(ValueError, match="submitted for"):
        run_experiment(run)


class FakeExecutor:
    """submitit's executor as a list of what it was asked to map."""

    def __init__(self):
        self.mapped = []

    def map_array(self, function, *arguments):
        self.mapped.append((function, arguments))
        return [SimpleNamespace(job_id=str(1000 + i)) for i in range(len(arguments[0]))]


@pytest.fixture
def executors(monkeypatch):
    made = []

    def build(name, environment, folder, **parameters):
        made.append({"name": name, "parameters": parameters, "executor": FakeExecutor()})
        return made[-1]["executor"]

    monkeypatch.setattr("lsa.evaluation.run.slurm.executor", build)
    return made


def pooled(tmp_path, extra=""):
    (run,) = experiment.load(
        write(tmp_path, MINIMAL.format(tp=8)),
        [
            "model.name=zai-org/Fake-Model",
            "run.gpu=h200",
            f"run.output={tmp_path / 'results'}",
            *([extra] if extra else []),
        ],
        kind=Evaluation.ACCURACY,
    )
    return run


SOURCE = {"config": "x.yaml", "overrides": [], "index": 0}


def test_a_pool_is_named_at_submission_so_it_must_say_its_card(tmp_path, executors):
    run = pooled(tmp_path)
    run.run.gpu = ""
    with pytest.raises(ValueError, match="run.gpu is empty"):
        submit_pool(run, 2, SOURCE)
    assert executors == []


def test_workers_are_jobs_of_one_run_and_the_run_remembers_how_to_add_another(tmp_path, executors):
    run = pooled(tmp_path)
    jobs = submit_pool(run, 3, SOURCE)
    assert [job.job_id for job in jobs] == ["1000", "1001", "1002"]
    (made,) = executors
    function, (experiments, forces) = made["executor"].mapped[0]
    assert function is run_on_first_task
    # One run id, settled once, and each worker a copy of the same pooled evaluation
    assert {worker.run.id for worker in experiments} == {run.run.id}
    assert all(worker.run.pool for worker in experiments)
    assert made["name"] == f"lsa-pool-{run.run.id}"
    out = settle_launch(run).out
    # A worker still queued says where its run is, for the in flight list
    assert made["parameters"]["slurm_additional_parameters"]["comment"] == str(out)
    submission = io.read_json(out / "pool" / "submission.json")
    assert (submission["id"], submission["index"], submission["kind"]) == (out.name, 0, "accuracy")


def test_a_job_s_threads_are_bound_to_its_gpus_beside_its_other_sbatch_options():
    parameters = {"gpus_per_node": 2, "slurm_additional_parameters": {"reservation": "lsa"}}
    bound = bound_to_gpus(parameters)
    assert bound["slurm_additional_parameters"] == {"reservation": "lsa", "gres_flags": "enforce-binding"}
    assert bound["gpus_per_node"] == 2
    assert parameters["slurm_additional_parameters"] == {"reservation": "lsa"}


def test_a_one_gpu_job_is_left_unbound():
    parameters = {"gpus_per_node": 1, "slurm_additional_parameters": {"reservation": "lsa"}}
    assert bound_to_gpus(parameters) == parameters


def test_pool_workers_and_single_jobs_both_bind_their_threads_to_their_gpus(tmp_path, executors):
    run = pooled(tmp_path)
    submit_pool(run, 1, SOURCE, {"gpus_per_node": 2, "slurm_additional_parameters": {"reservation": "lsa"}})
    submit([pooled(tmp_path)], {"gpus_per_node": 2})
    pool_job, single_job = executors
    assert pool_job["parameters"]["slurm_additional_parameters"]["gres_flags"] == "enforce-binding"
    assert pool_job["parameters"]["slurm_additional_parameters"]["reservation"] == "lsa"
    assert single_job["parameters"]["slurm_additional_parameters"] == {"gres_flags": "enforce-binding"}


def test_pooled_workers_keep_the_default_free_port_unless_the_run_names_one(tmp_path, executors):
    submit_pool(pooled(tmp_path), 2, SOURCE)
    submit_pool(pooled(tmp_path, "run.port=61300"), 2, SOURCE)
    default, named = (made["executor"].mapped[0][1][0] for made in executors)
    assert {worker.run.port for worker in default} == {0}
    assert {worker.run.port for worker in named} == {61300}


def test_workers_added_later_land_in_the_run_already_there(tmp_path, executors):
    first = pooled(tmp_path, "run.id=fixed")
    submit_pool(first, 1, SOURCE)
    later = pooled(tmp_path, "run.id=fixed")
    submit_pool(later, 2, SOURCE)
    assert settle_launch(later).out == settle_launch(first).out
    assert len(executors[1]["executor"].mapped[0][1][0]) == 2


def test_a_run_a_single_job_is_still_running_is_not_given_workers(tmp_path, executors, monkeypatch):
    run = pooled(tmp_path, "run.id=fixed")
    out = settle_launch(run).out
    out.mkdir(parents=True)
    launches.append(out, {"job": "77", "user": "alice", "host": "n1", "started": "2026-10-05T01:00:00+00:00"})
    monkeypatch.setattr(launches, "accounting", lambda jobs, fields=("State",): {"77": {"State": "RUNNING"}})
    with pytest.raises(ValueError, match="single-job launch still running"):
        submit_pool(run, 2, SOURCE)
    assert executors == []
    # Once it ended, the run is an ordinary one to add to
    monkeypatch.setattr(
        launches, "accounting", lambda jobs, fields=("State",): {"77": {"State": "CANCELLED"}}
    )
    submit_pool(run, 2, SOURCE)
    assert len(executors) == 1


def test_a_pool_worker_logs_itself_by_name_and_a_single_job_does_not(tmp_path, monkeypatch):
    run = pooled(tmp_path, "run.id=fixed")
    monkeypatch.setitem(RUNNERS, Evaluation.ACCURACY, lambda pinned, force: settle_launch(pinned).out)
    monkeypatch.setenv("SLURM_JOB_ID", "55")
    run_experiment(run)
    run.run.pool = True
    run_experiment(run)
    single, worker = launches.read(settle_launch(run).out)
    assert "worker" not in single
    assert worker["worker"] == "55"
