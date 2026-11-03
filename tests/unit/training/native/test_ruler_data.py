import pytest

from lsa.training.native import ruler_data
from lsa.training.native.benchmark import tasks


def test_the_rendering_is_read_from_the_root_the_preparer_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("LSA_RULER_ROOT", str(tmp_path))
    assert ruler_data.root_default() == str(tmp_path) == tasks.ruler_root()
    monkeypatch.delenv("LSA_RULER_ROOT")
    assert ruler_data.root_default().endswith("data/benchmarks-gpt2")


def test_a_task_outside_the_card_is_refused_before_anything_is_generated(tmp_path):
    with pytest.raises(ValueError, match="not a RULER task"):
        ruler_data.render("niah_single_9", 4096, 10, None, str(tmp_path))


def test_a_dry_run_submits_nothing_and_fetches_nothing(capsys, tmp_path):
    ruler_data.main(["--root", str(tmp_path), "--lengths", "4096", "--dry-run"])
    assert "would submit" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
