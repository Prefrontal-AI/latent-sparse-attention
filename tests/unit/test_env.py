"""The launch contract - what a task started by slurm has to find in its environment.

`adopt_job_environment` is the one piece of this that runs on a compute node with no torchrun in front of
it, and a gate that needs four integers must not fail on an unrelated import.
"""

import os
import subprocess
from pathlib import Path

import pytest

from lsa import env, site

SLURM = {
    "SLURM_PROCID": "3",
    "SLURM_LOCALID": "3",
    "SLURM_NTASKS": "4",
    "SLURM_JOB_ID": "152895",
    "SLURM_JOB_NODELIST": "node-13",
}


def slurm_task(monkeypatch, **overrides):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**SLURM, **overrides}.items():
        monkeypatch.setenv(name, value)


def test_slurms_own_variables_become_torchs(monkeypatch):
    slurm_task(monkeypatch)
    env.adopt_job_environment()
    assert (os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"]) == ("3", "3", "4")
    assert os.environ["MASTER_ADDR"] == "node-13"
    assert os.environ["MASTER_PORT"] == str(env.PORT_BASE + 152895 % env.PORT_SPAN)


def test_a_launcher_that_already_ranked_us_wins(monkeypatch):
    slurm_task(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    env.adopt_job_environment()
    assert os.environ["RANK"] == "0"


def test_two_jobs_on_one_node_do_not_share_a_port(monkeypatch):
    slurm_task(monkeypatch, SLURM_JOB_ID="152895")
    env.adopt_job_environment()
    first = os.environ["MASTER_PORT"]
    slurm_task(monkeypatch, SLURM_JOB_ID="152896")
    env.adopt_job_environment()
    assert os.environ["MASTER_PORT"] != first


def test_an_array_task_rendezvouses_with_its_own_job(monkeypatch):
    """submitit spells an array task `152895_3`, and the port comes from the job rather than the task"""
    slurm_task(monkeypatch, SLURM_JOB_ID="152895_3")
    env.adopt_job_environment()
    assert os.environ["MASTER_PORT"] == str(env.PORT_BASE + 152895 % env.PORT_SPAN)


def test_nothing_to_translate_says_so(monkeypatch):
    slurm_task(monkeypatch)
    monkeypatch.delenv("SLURM_PROCID")
    with pytest.raises(RuntimeError, match="no RANK and no SLURM_PROCID"):
        env.adopt_job_environment()


def test_one_named_node_needs_no_scontrol(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node-13")
    monkeypatch.setattr(env.subprocess, "run", _refuse)
    assert env.rendezvous_host() == "node-13"


def test_a_compressed_nodelist_that_cannot_be_expanded_raises(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NODELIST", "worker-[13,15]")
    monkeypatch.setattr(env.subprocess, "run", _refuse)
    with pytest.raises(RuntimeError, match="cannot expand"):
        env.rendezvous_host()


def _refuse(*args, **kwargs):
    raise OSError("no scontrol here")


def test_no_allocation_is_a_local_run(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    assert env.rendezvous_host() == "localhost"


def exported(lines, **environment):
    """Run shell lines in a clean environment and return what they export"""
    script = "\n".join([*lines, "env"])
    base = {"PATH": os.environ["PATH"], "USER": "someone"}
    shown = subprocess.run(
        ["bash", "-c", script], env={**base, **environment}, capture_output=True, text=True, check=True
    ).stdout
    return dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)


def make_site(tmp_path, monkeypatch, text):
    (tmp_path / "here.yaml").write_text(text)
    monkeypatch.setattr(site, "SITES", tmp_path)
    monkeypatch.setattr(site, "DEFAULT_SITE", "here")
    monkeypatch.setattr(site, "_selected", None)


def test_the_hub_cache_is_hf_home_then_the_site_then_the_hubs_own_default(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "ada")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    make_site(tmp_path, monkeypatch, "hf_home: /caches/$USER/hf\n")
    monkeypatch.setenv("HF_HOME", "/given")
    assert env.hf_home() == Path("/given")
    monkeypatch.delenv("HF_HOME")
    assert env.hf_home() == Path("/caches/ada/hf")
    make_site(tmp_path, monkeypatch, "root: /r\n")
    assert env.hf_home() == Path("~/.cache/huggingface").expanduser()


def test_a_site_without_a_hub_cache_leaves_it_to_the_hub(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    make_site(tmp_path, monkeypatch, "root: /r\n")
    assert env.declared_hf_home() == ""
    assert env.site_hf_home() is None


def test_the_sites_hub_cache_is_adopted_only_where_its_parent_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    make_site(tmp_path, monkeypatch, f"hf_home: {tmp_path}/users/ada/caches/hf_home\n")
    assert env.site_hf_home() is None
    (tmp_path / "users" / "ada").mkdir(parents=True)
    assert env.site_hf_home() == f"{tmp_path}/users/ada/caches/hf_home"
