"""How a source's documents are built and kept - the first collector's rules, on rows a test writes."""

import pytest

from lsa import assets
from lsa.assets.constants import Collector
from lsa.data import collect
from lsa.data.collect import (
    Collection,
    digest,
    long_documents,
    render_trajectory,
    repository_documents,
    text_documents,
    trace_documents,
)
from lsa.data.lock import compare

COLLECTED = ["arxiv", "doab", "gutenberg", "hansard", "swe", "usgpo", "wikimedia", "code"]


def file(repository, path, content, licences=("MIT",)):
    return {
        "max_stars_repo_name": repository,
        "max_stars_repo_path": path,
        "content": content,
        "max_stars_repo_licenses": list(licences),
    }


def test_a_text_row_is_a_document_named_by_its_id():
    rows = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]

    assert list(text_documents(rows, "books", "id", "text")) == [("a", "one"), ("b", "two")]


def test_a_row_without_an_id_is_numbered_under_the_source():
    rows = [{"id": "a", "text": "one"}, {"id": "", "text": "two"}, {"text": "three"}]

    assert [doc_id for doc_id, _ in text_documents(rows, "books", "id", "text")] == [
        "a",
        "books-1",
        "books-2",
    ]


def test_a_repository_is_its_files_in_path_order_under_a_header_each():
    rows = [file("o/r", "b.py", "B"), file("o/r", "a.py", "A"), file("o/other", "x.py", "X")]

    documents = dict(repository_documents(lambda language: rows, ["python"], buffer_chars=10**6))

    assert documents == {"python/o/r": "# a.py\nA\n\n# b.py\nB", "python/o/other": "# x.py\nX"}


def test_a_file_is_kept_by_its_first_listed_licence_and_when_it_lists_none():
    rows = [
        file("o/gpl", "a.py", "A", ["GPL-3.0"]),
        file("o/mixed", "a.py", "A", ["MIT", "GPL-3.0"]),
        file("o/reversed", "a.py", "A", ["GPL-3.0", "MIT"]),
        file("o/none", "a.py", "A", []),
    ]

    documents = dict(repository_documents(lambda language: rows, ["java"], buffer_chars=10**6))

    assert sorted(documents) == ["java/o/mixed", "java/o/none"]


def test_a_language_is_read_until_its_files_fill_the_buffer():
    read = []

    def rows():
        for index in range(5):
            read.append(index)
            yield file("o/r", f"{index}.py", "x" * 10)

    documents = dict(repository_documents(lambda language: rows(), ["python"], buffer_chars=25))

    assert read == [0, 1, 2]
    assert documents["python/o/r"].count("# ") == 3


def test_a_language_is_opened_only_when_the_one_before_it_is_used_up():
    opened = []

    def open_rows(language):
        opened.append(language)
        return [file("o/r", "a.py", "A")]

    stream = repository_documents(open_rows, ["python", "java", "go"], buffer_chars=10**6)
    next(stream)

    assert opened == ["python"]


def test_a_trajectory_is_kept_for_a_permissive_licence_and_named_by_its_config():
    rows = [
        {"license": "MIT", "trajectory_id": "t1", "trajectory": [{"role": "user", "content": "hi"}]},
        {"license": "GPL-3.0", "trajectory_id": "t2", "trajectory": []},
        {"license": "Apache-2.0", "instance_id": "i3", "trajectory": [{"role": "user", "content": "yo"}]},
    ]

    documents = dict(trace_documents(lambda config: rows, ["openhands"]))

    assert documents == {"openhands:t1": "<user>\nhi", "openhands:i3": "<user>\nyo"}


def test_a_trajectory_renders_each_turns_role_reasoning_content_and_calls():
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "reasoning_content": "think", "content": "ok", "tool_calls": [{"name": "ls"}]},
        {"content": "no role"},
    ]

    assert render_trajectory(messages) == (
        '<user>\nfix it\n<assistant>\nthink\nok\n[{"name": "ls"}]\n<?>\nno role'
    )


def counted(texts):
    # Three characters to a token, which is above the 2.5 the prefilter allows for
    return [len(text) // 3 for text in texts]


def test_a_document_is_kept_at_the_minimum_and_over():
    documents = [("short", "a" * 10), ("below", "a" * 299), ("exact", "a" * 300), ("long", "a" * 900)]

    kept = list(long_documents(documents, counted, min_tokens=100))

    # "below" is long enough in characters to be counted, and is dropped for its 99 tokens
    assert [(doc_id, tokens) for doc_id, _, tokens in kept] == [("exact", 100), ("long", 300)]


def test_a_document_too_short_in_characters_is_never_tokenized():
    seen = []

    def count(texts):
        seen.extend(texts)
        return [len(text) for text in texts]

    list(long_documents([("a", "x" * 10), ("b", "x" * 249), ("c", "x" * 250)], count, min_tokens=100))

    assert [len(text) for text in seen] == [250]


def test_collection_stops_after_the_batch_that_reaches_the_cap(monkeypatch):
    monkeypatch.setattr(Collection, "BATCH", 2)
    documents = [(f"d{index}", "a" * 300) for index in range(10)]

    kept = list(long_documents(documents, counted, min_tokens=100, max_tokens=250))

    # 250 is passed in the second batch, which is whole, so the kept tokens end at 400 rather than at 250
    assert [doc_id for doc_id, _, _ in kept] == ["d0", "d1", "d2", "d3"]


def test_a_limit_counts_the_documents_read_before_any_is_dropped():
    documents = [("short", "a"), ("long1", "a" * 300), ("long2", "a" * 300)]

    kept = list(long_documents(documents, counted, min_tokens=100, limit_docs=2))

    assert [doc_id for doc_id, _, _ in kept] == ["long1"]


def test_a_digest_is_the_sha256_of_the_utf8_text():
    assert digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert digest("é") != digest("e")


def test_two_collections_of_the_same_documents_in_another_order_are_told_apart():
    locked = [("a", "1"), ("b", "2"), ("c", "3")]

    assert compare("s", locked, list(locked)).ok
    swapped = compare("s", locked, [("b", "2"), ("a", "1"), ("c", "3")])
    assert swapped.reordered and not (swapped.missing or swapped.extra or swapped.changed)


def test_a_comparison_names_what_is_missing_extra_and_changed():
    result = compare("s", [("a", "1"), ("b", "2"), ("c", "3")], [("a", "1"), ("c", "9"), ("d", "4")])

    assert (result.missing, result.extra, result.changed) == (["b"], ["d"], ["c"])
    assert not result.ok
    assert "1 missing: b" in result.report()


@pytest.mark.parametrize("name", COLLECTED)
def test_every_collected_source_pins_the_tokenizer_that_counts_it(name):
    card = assets.source(name)

    assert card.collector in Collector.ALL
    assert card.min_tokens == 65536
    assert card.count_tokenizer == "HuggingFaceTB/SmolLM3-3B-Base"
    assert len(card.count_tokenizer_revision) == 40


def test_the_sources_capped_at_three_billion_tokens_are_the_three_that_were():
    capped = {name for name in COLLECTED if assets.source(name).max_tokens}

    assert capped == {"usgpo", "code", "swe"}
    assert {assets.source(name).max_tokens for name in capped} == {3_000_000_000}


def test_code_is_read_unpinned_because_its_revision_was_never_recorded():
    card = assets.source("code")

    assert card.revision == ""
    assert card.collector == Collector.REPOSITORIES
    assert card.subsets == ["python", "java"]


def test_the_in_house_sources_are_not_collected_from_the_hub():
    assert assets.source("games").collector == ""
    assert assets.source("edgebench").collector == ""


def test_a_collected_source_must_name_what_counts_its_documents():
    with pytest.raises(ValueError, match="the tokenizer"):
        assets.SourceCard(
            name="x", path="/x", licence="MIT", dataset="a/b", collector=Collector.TEXT, min_tokens=10
        )
    with pytest.raises(ValueError, match="collector must be one of"):
        assets.SourceCard(name="x", path="/x", licence="MIT", dataset="a/b", collector="scrape")


def test_the_preparer_runs_online_and_every_other_one_does_not():
    assert "export HF_HUB_OFFLINE=0" in collect.ENVIRONMENT
    assert "export HF_HUB_OFFLINE=1" not in collect.ENVIRONMENT
