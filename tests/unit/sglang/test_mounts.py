"""A path the container reads or writes must lie under a directory it mounts, checked before a job is queued."""

import re

import pytest

from lsa import paths
from lsa.sglang.launch import require_mounted
from tests.helpers.sites import use_site


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    """Point the site at a root and one declared mount, with no hub cache or extra mount beside them."""
    root = tmp_path / "root"
    root.mkdir()
    users = tmp_path / "users"
    users.mkdir()
    use_site(monkeypatch, tmp_path, f"root: {root}\nmounts:\n  - {users}\n")
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no_hub"))
    return root, users


def test_a_path_under_the_project_root_is_seen(mounted):
    root, _ = mounted
    require_mounted(root / "evaluation" / "not_yet_made", "run.output")


def test_a_path_under_a_site_mount_is_seen(mounted):
    _, users = mounted
    require_mounted(users / "ada" / "results", "run.output")


def test_a_path_outside_every_mount_is_refused_naming_it_and_the_mounts(mounted, tmp_path):
    root, users = mounted
    elsewhere = tmp_path / "elsewhere" / "results"
    with pytest.raises(ValueError, match="outside every directory the container mounts") as refused:
        require_mounted(elsewhere, "run.output")
    message = str(refused.value)
    assert f"run.output {elsewhere} is" in message
    assert str(root) in message
    assert str(users) in message


def test_a_link_inside_a_mount_that_resolves_outside_is_refused(mounted, tmp_path):
    _, users = mounted
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = users / "results"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match=re.escape(f"resolves to {outside}")):
        require_mounted(link / "performance", "run.output")


def test_a_link_outside_the_mounts_that_resolves_inside_one_is_seen(mounted, tmp_path):
    root, _ = mounted
    link = tmp_path / "shortcut"
    link.symlink_to(root)
    require_mounted(link / "evaluation", "run.output")


def test_a_directory_the_operator_adds_is_seen(mounted, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("LSA_EXTRA_MOUNTS", str(extra))
    require_mounted(extra / "results", "run.output")
