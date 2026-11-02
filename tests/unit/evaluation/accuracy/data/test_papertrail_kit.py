"""PaperTrail's kit is the one the card pins, and an edited copy of it is refused."""

import shutil
from pathlib import Path

import pytest

from lsa.evaluation.accuracy.data import papertrail_card
from lsa.evaluation.accuracy.data.papertrail import KIT, check_kit, kit_digest
from lsa.evaluation.accuracy.data.papertrail.rebuild import read_kit


def test_the_kit_in_the_repository_is_the_one_the_card_pins() -> None:
    assert check_kit(KIT) == papertrail_card().revision


def test_the_kit_holds_every_sample_the_card_names() -> None:
    kit = read_kit(KIT, papertrail_card().tasks)
    assert set(kit.samples) == set(papertrail_card().tasks)
    assert kit.papers


def test_an_edited_kit_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "kit"
    shutil.copytree(KIT, copy)
    check_kit(copy)
    questions = copy / papertrail_card().tasks[0] / "questions.jsonl"
    questions.write_text(questions.read_text() + "\n")
    with pytest.raises(ValueError, match="pins"):
        check_kit(copy)


def test_the_digest_follows_a_files_name_as_well_as_its_bytes(tmp_path: Path) -> None:
    copy = tmp_path / "kit"
    shutil.copytree(KIT, copy)
    sample = copy / papertrail_card().tasks[0] / "nic.jsonl"
    sample.rename(sample.with_name("renamed.jsonl"))
    assert kit_digest(copy) != papertrail_card().revision


def test_a_path_that_is_not_a_kit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        check_kit(tmp_path / "missing")
