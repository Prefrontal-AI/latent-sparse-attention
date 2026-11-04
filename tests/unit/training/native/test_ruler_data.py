"""The RULER rendering, on a tokenizer that needs no download and fixtures small enough to render offline."""

import json
from types import SimpleNamespace

import pytest

from lsa.training.native import ruler_data
from lsa.training.native.benchmark import tasks

LENGTH = 1024
SAMPLES = 3


class Chars:
    """A tokenizer that counts one token for every four characters, about what GPT-2 does to English."""

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[0] * (len(text) // 4 + 1))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A root holding a few SQuAD and HotpotQA records in the shapes lm-eval reads."""
    (tmp_path / "sources").mkdir()
    paragraphs = [
        {
            "context": f"Paragraph {i} tells of the town of Place{i} and its river, its mill and its market. "
            * 3,
            "qas": [
                {
                    "question": f"What is the town of paragraph {i} called?",
                    "answers": [{"text": f"Place{i}"}, {"text": f"the town of Place{i}"}],
                    "is_impossible": False,
                }
            ],
        }
        for i in range(12)
    ]
    squad = {"data": [{"paragraphs": paragraphs}]}
    hotpot = [
        {
            "question": f"Who built the mill of Place{i}?",
            "answer": f"Builder{i}",
            "context": [
                [f"Mill{i}", [f"Builder{i} built the mill of Place{i}. "]],
                [f"Town{i}", ["A town by a river. "]],
            ],
        }
        for i in range(12)
    ]
    (tmp_path / "sources" / "squad.json").write_text(json.dumps(squad))
    (tmp_path / "sources" / "hotpotqa.json").write_text(json.dumps(hotpot))
    from lm_eval.tasks.ruler import qa_utils

    # The readers are cached on their url, and the render patches the downloader for good
    for reader in (qa_utils.read_squad, qa_utils.read_hotpotqa):
        reader.cache_clear()
    monkeypatch.setattr(qa_utils, "download_json", qa_utils.download_json)
    yield str(tmp_path)
    for reader in (qa_utils.read_squad, qa_utils.read_hotpotqa):
        reader.cache_clear()


def cell(task, root, length=LENGTH, samples=SAMPLES):
    return ruler_data.render(task, length, samples, Chars(), root)


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


@pytest.mark.parametrize(
    "task",
    ["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt", "cwe", "fwe", "qa_squad", "qa_hotpot"],
)
def test_a_cell_has_the_fields_the_benchmark_reads_within_its_length(task, root):
    rows = cell(task, root)
    assert len(rows) == SAMPLES
    assert [row["index"] for row in rows] == list(range(SAMPLES))
    for row in rows:
        assert all(field in row for field in ruler_data.FIELDS)
        assert row["max_length"] == LENGTH
        assert row["outputs"]
        assert 0 < row["length"] <= LENGTH
    assert len({row["input"] for row in rows}) == SAMPLES


@pytest.mark.parametrize(
    "task", ["niah_single_1", "niah_multikey_2", "vt", "cwe", "fwe", "qa_squad", "qa_hotpot"]
)
def test_a_cell_is_the_same_twice_in_one_process(task, root):
    assert cell(task, root) == cell(task, root)


def test_a_needle_cell_asks_for_the_needle_it_hid():
    [row, *_] = cell("niah_single_1", None)
    [answer] = row["outputs"]
    assert answer in row["input"]
    assert row["input"].startswith("A special magic number is hidden")


def test_a_cwe_cell_does_not_depend_on_which_cwe_cell_came_before_it(root):
    first = [cell("cwe", root, length, 2) for length in (512, 1024)]
    second = [cell("cwe", root, length, 2) for length in (1024, 512)][::-1]
    assert first == second


def test_a_fwe_cell_leaves_room_for_the_words_it_asks_for(root):
    rows = cell("fwe", root)
    assert all(row["length"] <= LENGTH for row in rows)
    assert all(row["max_length"] == LENGTH for row in rows)


def test_a_question_whose_paragraphs_do_not_fit_is_left_out_and_too_few_are_refused(root):
    with pytest.raises(ValueError, match="only 0 questions fit"):
        cell("qa_hotpot", root, 30, 1)


def test_ruler_cells_are_refused_for_a_model_with_another_tokenizer():
    names = tasks.expand("ruler-4k")
    tasks.check_ruler_tokenizer(names, "gpt2")
    tasks.check_ruler_tokenizer(["hellaswag"], "qwen3.5-0.8b")
    with pytest.raises(ValueError, match="mis-sized"):
        tasks.check_ruler_tokenizer(names, "qwen3.5-0.8b")
