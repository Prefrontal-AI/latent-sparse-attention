import json

import pytest

from lsa.io import (
    Claim,
    file_sha256,
    git_sha,
    git_state,
    read_json,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)


def test_write_json_replaces_whole_and_leaves_no_scratch(tmp_path):
    target = tmp_path / "nested" / "record.json"
    write_json(target, {"a": 1})
    write_json(target, {"b": "é"}, indent=1)

    assert json.loads(target.read_text()) == {"b": "é"}
    assert [path.name for path in target.parent.iterdir()] == ["record.json"]


def test_a_failed_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "record.json"
    write_json(target, {"a": 1})

    with pytest.raises(TypeError):
        write_json(target, {"bad": object()})

    assert json.loads(target.read_text()) == {"a": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["record.json"]


def test_read_json_returns_the_payload_and_a_missing_file_returns_only_a_given_default(tmp_path):
    target = tmp_path / "record.json"
    write_json(target, {"a": 1})

    assert read_json(target) == {"a": 1}
    assert read_json(tmp_path / "missing.json", None) is None
    assert read_json(tmp_path / "missing.json", {}) == {}
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


def test_utc_now_spells_seconds_or_keeps_the_microseconds():
    assert len(utc_now()) == len("2026-10-02T12:00:00+00:00")
    assert utc_now().endswith("+00:00")
    assert "." in utc_now("microseconds")


def test_jsonl_round_trips_and_a_missing_file_reads_empty(tmp_path):
    target = tmp_path / "rows.jsonl"
    write_jsonl(target, [{"i": 0}, {"i": 1}])

    assert read_jsonl(target) == [{"i": 0}, {"i": 1}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_file_sha256_hashes_bytes_or_returns_none(tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"abc")

    assert file_sha256(target) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(tmp_path / "missing") is None


def test_the_sha_is_none_outside_a_repository(tmp_path):
    assert git_sha(tmp_path) is None


def test_the_state_outside_a_repository_is_an_empty_sha(tmp_path):
    assert git_state(tmp_path) == {"sha": "", "dirty": False}


def test_a_claim_is_held_by_one_process_until_released(tmp_path):
    first = Claim(tmp_path / "nested" / "doc.claim", stale_after=900.0)
    second = Claim(tmp_path / "nested" / "doc.claim", stale_after=900.0)

    assert first.take()
    assert not second.take()
    first.release()
    assert second.take()
    second.release()
    assert not second.path.exists()


def test_a_claim_nobody_refreshed_is_taken_over(tmp_path):
    held = Claim(tmp_path / "doc.claim", stale_after=0.0)
    assert held.take()

    assert Claim(tmp_path / "doc.claim", stale_after=0.0).take()
