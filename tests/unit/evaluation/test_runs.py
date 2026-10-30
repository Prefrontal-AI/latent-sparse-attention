"""A run named on a command line is its directory or its record, and anything else is refused at the boundary."""

import argparse
import sys

import pytest

from lsa.evaluation import rename
from lsa.evaluation.runs import run_directory, unrequired_run_directory


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "record.json").write_text("{}")
    return directory


def test_a_directory_names_itself(run):
    assert run_directory(str(run)) == run


def test_a_record_names_its_directory(run):
    assert run_directory(run / "record.json") == run


@pytest.mark.parametrize("missing", ["nowhere", "nowhere/record.json"])
def test_a_path_that_is_not_there_is_refused(tmp_path, missing):
    with pytest.raises(argparse.ArgumentTypeError, match="neither a run directory"):
        run_directory(tmp_path / missing)


def test_a_directory_without_a_record_is_refused(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError, match="record.json"):
        run_directory(tmp_path)


def test_a_file_that_is_not_a_record_is_refused(tmp_path):
    other = tmp_path / "meta.json"
    other.write_text("{}")
    with pytest.raises(argparse.ArgumentTypeError, match="neither a run directory"):
        run_directory(other)


def test_a_run_still_to_write_its_record_passes_where_none_is_required(tmp_path):
    assert unrequired_run_directory(tmp_path) == tmp_path


def test_a_reader_refuses_what_is_neither_with_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rename", str(tmp_path), "--name", "x", "--into", str(tmp_path)])
    with pytest.raises(SystemExit) as failure:
        rename.main()
    assert failure.value.code == 2
    assert "neither a run directory" in capsys.readouterr().err
