"""What a selection chooses - equal tokens per bucket, equal shares per domain, and no document read twice."""

import pytest

from lsa.training.distillation.constants import Split
from lsa.training.distillation.data import selection
from lsa.training.distillation.data.selection import (
    Candidate,
    SelectionConfig,
    allocate,
    tier,
    water_fill,
)

LENGTHS = [16, 32, 64]
BUDGET = 64 * 8


def a_selection(**overrides):
    values = dict(
        datasets=["tokens"],
        lengths=LENGTHS,
        tokens_per_bucket=BUDGET,
        tokens_per_tier=BUDGET // 4,
        domains={"books": ["gutenberg"], "academic": ["doab", "arxiv"], "code": ["code"]},
    )
    values.update(overrides)
    return SelectionConfig(**values)


def a_pool(counts, tokens=64, held_out=False):
    """Documents per source, every one long enough for every bucket unless `tokens` says otherwise"""
    return [
        Candidate("tokens", source, f"{source}-{index}", tokens, f"{source}-{index}", held_out)
        for source, count in counts.items()
        for index in range(count)
    ]


def tokens_by(entries, key):
    totals = {}
    for entry in entries:
        totals[entry[key]] = totals.get(entry[key], 0) + entry["length"]
    return totals


def test_water_fill_splits_evenly_and_gives_a_short_key_what_it_has():
    assert water_fill(9, {"a": 10, "b": 10, "c": 1}) == {"a": 4, "b": 4, "c": 1}


def test_every_bucket_holds_the_same_tokens():
    chosen = allocate(
        a_selection(), a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60, "code": 60}), Split.TRAIN, None
    )

    assert tokens_by(chosen, "length") == dict.fromkeys(LENGTHS, BUDGET)


def test_a_bucket_with_its_own_budget_holds_that_many_tokens_and_the_rest_the_shared_one():
    chosen = allocate(
        a_selection(tokens_per_length={64: 2 * BUDGET}),
        a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60, "code": 60}),
        Split.TRAIN,
        None,
    )

    assert tokens_by(chosen, "length") == {16: BUDGET, 32: BUDGET, 64: 2 * BUDGET}


def test_a_budget_for_a_length_the_selection_lacks_is_refused():
    with pytest.raises(ValueError, match="tokens_per_length"):
        a_selection(tokens_per_length={128: BUDGET})


def test_a_document_is_read_in_one_bucket_only():
    chosen = allocate(
        a_selection(), a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60, "code": 60}), Split.TRAIN, None
    )

    documents = [entry["document"] for entry in chosen]
    assert len(documents) == len(set(documents))


def test_domains_share_a_bucket_equally_however_many_sources_they_arrive_in():
    chosen = allocate(
        a_selection(), a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60, "code": 60}), Split.TRAIN, None
    )

    by_domain = tokens_by([entry for entry in chosen if entry["length"] == 16], "domain")
    assert max(by_domain.values()) - min(by_domain.values()) <= 16


def test_a_domain_with_nothing_that_long_gives_its_share_to_the_others():
    pool = a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60}) + a_pool({"code": 60}, tokens=32)
    chosen = allocate(a_selection(), pool, Split.TRAIN, None)

    longest = [entry for entry in chosen if entry["length"] == 64]
    assert "code" not in {entry["domain"] for entry in longest}
    assert sum(entry["length"] for entry in longest) == BUDGET


def test_a_source_stays_out_of_buckets_shorter_than_its_floor():
    chosen = allocate(
        a_selection(min_length={"code": 64}),
        a_pool({"gutenberg": 60, "doab": 60, "code": 60}),
        Split.TRAIN,
        None,
    )

    assert {entry["length"] for entry in chosen if entry["source"] == "code"} == {64}


def test_a_window_starts_at_the_front_only_for_a_source_that_asks():
    pool = a_pool({"gutenberg": 60, "code": 60}, tokens=640)
    chosen = allocate(a_selection(from_start=["code"]), pool, Split.TRAIN, None)

    assert {entry["window"] for entry in chosen if entry["source"] == "code"} == {0}
    assert len({entry["window"] for entry in chosen if entry["source"] == "gutenberg"}) > 1


def test_a_bucket_takes_one_document_per_group():
    pool = [
        Candidate("tokens", "code", f"task{index % 3}:{index}", 64, f"task{index % 3}", False)
        for index in range(30)
    ]
    chosen = allocate(a_selection(group_by={"code": ":"}), pool, Split.TRAIN, None)

    for length in LENGTHS:
        groups = [entry["document"].split(":")[0] for entry in chosen if entry["length"] == length]
        assert len(groups) == len(set(groups))


def test_scoring_takes_the_cap_per_domain_and_no_more():
    chosen = allocate(
        a_selection(), a_pool({"gutenberg": 60, "doab": 60, "code": 60}, held_out=True), Split.EVAL, 2
    )

    for length in LENGTHS:
        by_domain = {}
        for entry in chosen:
            if entry["length"] == length:
                by_domain[entry["domain"]] = by_domain.get(entry["domain"], 0) + 1
        assert by_domain == {"books": 2, "academic": 2, "code": 2}


def test_a_tier_is_rebalanced_out_of_the_full_selection_and_keeps_its_scoring_side():
    one = a_selection()
    full = allocate(one, a_pool({"gutenberg": 60, "doab": 60, "arxiv": 60, "code": 60}), Split.TRAIN, None)
    full += allocate(one, a_pool({"gutenberg": 9, "code": 9}, held_out=True), Split.EVAL, 2)

    small = tier(one, full, BUDGET // 4)

    assert all(entry in full for entry in small)
    assert [entry for entry in small if entry["split"] == Split.EVAL] == [
        entry for entry in full if entry["split"] == Split.EVAL
    ]
    training = [entry for entry in small if entry["split"] == Split.TRAIN]
    assert tokens_by(training, "length") == dict.fromkeys(LENGTHS, BUDGET // 4)


def test_a_task_with_a_scored_trajectory_trains_on_none_of_its_trajectories(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lsa.assets import TokenizedCard

    documents = [f"task{index % 4}:run{index}" for index in range(400)]
    pq.write_table(
        pa.table({"doc_id": documents, "row_start": list(range(400)), "n_samples": [1] * 400}),
        tmp_path / "code.manifest.parquet",
    )
    card = TokenizedCard(
        name="tokens", path=str(tmp_path), sources=["code"], tokenizer="glm53_flash", sample_length=64
    )
    monkeypatch.setattr(selection.assets, "tokenized", lambda name: card)

    found = selection.candidates(a_selection(domains={"code": ["code"]}, group_by={"code": ":"}))

    scored = {one.group for one in found if one.held_out}
    assert scored
    assert not [one for one in found if not one.held_out and one.group in scored]


def test_a_tier_as_large_as_the_selection_is_refused():
    with pytest.raises(ValueError, match="smaller than tokens_per_bucket"):
        a_selection(tokens_per_tier=BUDGET)


def test_a_source_in_two_domains_is_refused():
    with pytest.raises(ValueError, match="two domains"):
        a_selection(domains={"books": ["gutenberg"], "more": ["gutenberg"]})
