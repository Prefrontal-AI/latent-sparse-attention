"""What a job asks slurm for, read off the sbatch parameters submitit would write.

A fixture site names the launcher while Submitit builds the actual scheduler parameters.
No scheduler installation or submission is required.
"""

import argparse
import shlex
import sys
from pathlib import Path

import pytest
import submitit

from lsa import paths, site, slurm
from tests.helpers.sites import use_site

PARAMETERS = {"gpus_per_node": 4, "cpus_per_task": 32, "mem_gb": 600, "timeout_min": 150}


@pytest.fixture(autouse=True)
def on_slurm(monkeypatch, tmp_path):
    use_site(monkeypatch, tmp_path, "root: /test/root\nlauncher: slurm\n")
    monkeypatch.setattr(submitit.SlurmExecutor, "affinity", classmethod(lambda cls: 2))


def on_site(monkeypatch, tmp_path, text):
    use_site(monkeypatch, tmp_path, f"root: /test/root\n{text}")


def sbatch(executor):
    return executor._executor.parameters


def parse(*argv):
    parser = argparse.ArgumentParser()
    slurm.add_flags(parser)
    return parser.parse_args(argv)


def test_the_executor_asks_for_the_parameters_it_is_given():
    executor = slurm.executor("lsa-test", ["export A=1"], "/logs", **PARAMETERS)
    assert sbatch(executor) == {
        "job_name": "lsa-test",
        "time": 150,
        "gpus_per_node": 4,
        "cpus_per_task": 32,
        "mem": "600GB",
        "setup": [
            "export A=1",
            f'export PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"',
            "export LSA_SITE=test",
        ],
    }
    assert str(executor.folder) == "/logs"


def test_slurm_parameters_pass_through_unchanged():
    executor = slurm.executor(
        "lsa-test",
        [],
        "/logs",
        nodes=2,
        tasks_per_node=4,
        slurm_partition="sweep",
        slurm_additional_parameters={"requeue": True, "dependency": "afterok:152895"},
    )
    parameters = sbatch(executor)
    assert (parameters["nodes"], parameters["ntasks_per_node"], parameters["partition"]) == (2, 4, "sweep")
    assert parameters["additional_parameters"] == {"requeue": True, "dependency": "afterok:152895"}


def test_a_parameter_at_none_is_left_unset():
    assert "partition" not in sbatch(slurm.executor("lsa-test", [], "/logs", slurm_partition=None))


def test_the_job_runs_under_the_python_it_is_given():
    executor = slurm.executor("lsa-test", [], "/logs", slurm_python="/snapshot/.venv/bin/python")
    assert executor._executor.python == "/snapshot/.venv/bin/python"


def test_the_job_puts_its_pythons_bin_directory_on_path_so_the_venvs_tools_are_found():
    executor = slurm.executor("lsa-test", [], "/logs", slurm_python="/snapshot/my venv/bin/python")
    assert sbatch(executor)["setup"][0] == "export PATH='/snapshot/my venv/bin':\"$PATH\""


def test_the_job_without_a_python_of_its_own_uses_the_one_that_submitted_it():
    setup = sbatch(slurm.executor("lsa-test", [], "/logs"))["setup"]
    assert f'export PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"' in setup


def test_the_folder_defaults_to_the_log_root_of_lsa_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROOT_VARIABLE, str(tmp_path))
    assert slurm.executor("lsa-test", []).folder == tmp_path / "logs"


def test_a_command_line_that_says_nothing_overrides_nothing():
    args = parse()
    assert slurm.overrides(args) == {}
    assert args.folder is None


def test_every_flag_given_becomes_the_submitit_parameter_it_sets():
    args = parse(
        "--gpus", "8", "--cpus", "128", "--mem-gb", "750", "--hours", "0.5",
        "--partition", "sweep", "--reservation", "res", "--folder", "/logs",
    )  # fmt: skip
    assert slurm.overrides(args) == {
        "gpus_per_node": 8,
        "cpus_per_task": 128,
        "mem_gb": 750,
        "timeout_min": 30,
        "slurm_partition": "sweep",
        "slurm_additional_parameters": {"reservation": "res"},
    }
    assert args.folder == "/logs"


def test_an_override_changes_only_what_was_given():
    executor = slurm.executor("lsa-test", [], "/logs", **PARAMETERS)
    executor.update_parameters(**slurm.overrides(parse("--gpus", "8")))
    parameters = sbatch(executor)
    assert (parameters["gpus_per_node"], parameters["cpus_per_task"], parameters["time"]) == (8, 32, 150)


def test_a_second_additional_parameters_dict_replaces_the_first():
    """Why a submitter merges its own sbatch options into the reservation rather than passing them after"""
    executor = slurm.executor("lsa-test", [], "/logs", slurm_additional_parameters={"requeue": True})
    executor.update_parameters(**slurm.overrides(parse("--reservation", "res")))
    assert sbatch(executor)["additional_parameters"] == {"reservation": "res"}


def test_a_merged_requeue_keeps_the_reservation_beside_it():
    parameters = {**PARAMETERS, **slurm.overrides(parse("--reservation", "res"))}
    parameters["slurm_additional_parameters"] = {"requeue": True, **parameters["slurm_additional_parameters"]}
    executor = slurm.executor("lsa-test", [], "/logs", **parameters)
    assert sbatch(executor)["additional_parameters"] == {"requeue": True, "reservation": "res"}


def test_a_dry_run_says_what_it_would_ask_for_and_submits_nothing(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run built an executor")

    monkeypatch.setattr(slurm, "executor", refuse)
    job = slurm.submit(parse("--dry-run", "--hours", "1"), "lsa-test", [], PARAMETERS, print, "unused")
    assert job is None
    shown = capsys.readouterr().out
    assert "builtins.print" in shown and "'timeout_min': 60" in shown


def test_a_site_that_sets_no_slurm_value_adds_none(monkeypatch):
    asked = slurm.site_parameters(PARAMETERS)
    assert asked == PARAMETERS
    assert "additional_parameters" not in sbatch(slurm.executor("lsa-test", [], "/logs", **PARAMETERS))


def test_the_sites_values_are_laid_under_a_job_that_names_none(monkeypatch, tmp_path):
    on_site(
        monkeypatch,
        tmp_path,
        "launcher: slurm\nslurm:\n  partition: shared\n  account: lab\n  qos: long\n  constraint: b300\n"
        "  exclude: [n1, n2]\n",
    )
    parameters = sbatch(slurm.executor("lsa-test", [], "/logs", **PARAMETERS))
    assert (parameters["partition"], parameters["account"], parameters["qos"], parameters["constraint"]) == (
        "shared",
        "lab",
        "long",
        "b300",
    )
    assert parameters["additional_parameters"] == {"exclude": "n1,n2"}


def test_a_flag_wins_over_the_site_and_the_excluded_nodes_of_both_are_kept(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: slurm\nslurm:\n  partition: shared\n  exclude: n1\n")
    given = {**PARAMETERS, **slurm.overrides(parse("--partition", "sweep", "--reservation", "pool"))}
    given["slurm_additional_parameters"]["exclude"] = "n2,n1"
    parameters = sbatch(slurm.executor("lsa-test", [], "/logs", **given))
    assert parameters["partition"] == "sweep"
    assert parameters["additional_parameters"] == {"reservation": "pool", "exclude": "n1,n2"}


def test_a_site_with_an_unknown_slurm_key_is_refused(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: slurm\nslurm:\n  partiton: shared\n")
    with pytest.raises(ValueError, match="partiton"):
        site.slurm_values()


def test_a_site_without_a_launcher_is_refused(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "")
    with pytest.raises(site.NotNamed, match="launcher"):
        slurm.executor("lsa-test", [], "/logs")


def test_the_local_launcher_builds_the_local_executor_and_passes_no_slurm_flag(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: local\n")
    executor = slurm.executor("lsa-test", ["export A=1"], str(tmp_path / "logs"), **PARAMETERS)
    assert isinstance(executor._executor, submitit.LocalExecutor)
    assert executor._executor.parameters["setup"] == [
        "export A=1",
        f'export PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"',
        "export LSA_SITE=test",
    ]
    assert "partition" not in executor._executor.parameters


def test_the_local_launcher_refuses_more_than_one_node(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: local\n")
    with pytest.raises(site.Unsupported, match="2 nodes need slurm"):
        slurm.executor("lsa-test", [], str(tmp_path), nodes=2)


def test_the_local_launcher_refuses_a_reservation(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: local\n")
    parameters = slurm.overrides(parse("--reservation", "pool"))
    with pytest.raises(site.Unsupported, match="reservation needs slurm"):
        slurm.executor("lsa-test", [], str(tmp_path), **parameters)


def test_a_slurm_query_is_refused_where_there_is_no_slurm(monkeypatch, tmp_path):
    on_site(monkeypatch, tmp_path, "launcher: local\n")
    with pytest.raises(site.Unsupported, match="the queue needs slurm"):
        slurm.queue()
    with pytest.raises(site.Unsupported, match="a reservation needs slurm"):
        slurm.free_nodes("pool")


def test_the_roofline_job_takes_the_open_partition_the_site_names(monkeypatch, tmp_path):
    from lsa.evaluation.performance.roofline import run

    on_site(monkeypatch, tmp_path, "launcher: slurm\nslurm:\n  open_partition: shared\n")
    assert run.parameters() == {**run.PARAMETERS, "slurm_partition": "shared"}
    on_site(monkeypatch, tmp_path, "launcher: slurm\n")
    assert run.parameters() == run.PARAMETERS


def test_a_job_reads_the_site_its_submitter_selected(monkeypatch):
    monkeypatch.delenv(site.SITE_VARIABLE, raising=False)
    executor = slurm.executor("lsa-test", ["export A=1"], "/logs", **PARAMETERS)
    assert sbatch(executor)["setup"] == [
        "export A=1",
        f'export PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"',
        "export LSA_SITE=test",
    ]


def test_a_process_with_no_selected_site_reads_the_one_the_job_exports(monkeypatch):
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.setenv(site.SITE_VARIABLE, "elsewhere")
    assert site.current() == "elsewhere"
    site.select("chosen")
    assert site.current() == "chosen"
