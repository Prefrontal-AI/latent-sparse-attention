import json

import pytest

from lsa.evaluation.accuracy.data import ruler_card, ruler_path
from lsa.training.native.benchmark import tasks


def test_core_expands_to_the_short_tasks():
    assert tasks.expand("core") == [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "lambada_openai",
        "wikitext",
        "sciq",
    ]


def test_ruler_is_the_card_s_thirteen_tasks_at_the_trained_lengths_and_32k_is_apart():
    names = tasks.expand("ruler")
    assert sorted(tasks.RULER_TASKS) == sorted(ruler_card().tasks)
    assert len(names) == 13 * 3
    assert tasks.lengths_of(names) == [4096, 8192, 16384]
    assert tasks.lengths_of(tasks.expand("ruler-32k")) == [32768]
    assert names[0] == "ruler_niah_single_1_4096"


def test_a_list_mixes_presets_and_tasks_without_repeating_one():
    names = tasks.expand("hellaswag, core,ruler_niah_single_1_4096")
    assert names.count("hellaswag") == 1
    assert names[-1] == "ruler_niah_single_1_4096"


def test_a_cell_name_splits_into_its_task_and_length_and_a_bad_one_is_refused():
    assert tasks.parse_ruler("ruler_niah_multikey_2_16384") == ("niah_multikey_2", 16384)
    with pytest.raises(ValueError, match="not a RULER cell"):
        tasks.parse_ruler("ruler_nothing_4096")


def test_a_ruler_cell_is_graded_by_the_official_string_match(tmp_path):
    path = ruler_path("niah_single_1", 4096, str(tmp_path))
    rows = [
        {
            "index": 0,
            "input": "haystack one",
            "outputs": ["1234567"],
            "gen_prefix": "The number is",
            "length": 100,
            "max_length": 4096,
        },
        {
            "index": 1,
            "input": "haystack two",
            "outputs": ["7654321"],
            "gen_prefix": "The number is",
            "length": 100,
            "max_length": 4096,
        },
    ]
    tmp_path.joinpath("ruler").mkdir()
    with open(path, "w") as sink:
        sink.write("\n".join(json.dumps(row) for row in rows) + "\n")
    [task] = [tasks.build_ruler("ruler_niah_single_1_4096", str(tmp_path))]
    docs = list(task.eval_docs)
    assert len(docs) == 2
    assert task.doc_to_text(docs[0]) == "haystack one"
    assert task.doc_to_target(docs[0]) == ["1234567"]
    assert task.process_results(docs[0], [" 1234567."]) == {"score": 1.0}
    assert task.process_results(docs[1], [" 123"]) == {"score": 0.0}


def test_a_qa_cell_scores_the_fraction_of_references_found_as_the_paper_did(tmp_path):
    path = ruler_path("qa_squad", 4096, str(tmp_path))
    row = {
        "index": 0,
        "input": "documents and a question",
        "outputs": ["Denver Broncos", "the Broncos"],
        "gen_prefix": "Answer:",
        "length": 100,
        "max_length": 4096,
    }
    tmp_path.joinpath("ruler").mkdir()
    with open(path, "w") as sink:
        sink.write(json.dumps(row) + "\n")
    [task] = [tasks.build_ruler("ruler_qa_squad_4096", str(tmp_path))]
    [doc] = list(task.eval_docs)
    # Official RULER would count either reference alone as a full answer
    assert task.process_results(doc, [" Denver Broncos"]) == {"score": 0.5}
    assert task.process_results(doc, [" the Denver Broncos, the Broncos"]) == {"score": 1.0}


def test_ruler_4k_is_the_thirteen_tasks_at_4096_under_a_short_run_folder_name():
    names = tasks.expand("ruler-4k")
    assert len(names) == 13
    assert tasks.lengths_of(names) == [4096]
    assert [tasks.parse_ruler(name)[0] for name in names] == list(tasks.RULER_TASKS)
    # A run folder is named after the `--tasks` value, and the thirteen cells spelled out pass a file name's 255
    # => the preset is how the paper's column is run
    assert len("+".join(names)) > 255
