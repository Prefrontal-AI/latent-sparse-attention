"""Preparing a card - what each kind is refused for, and agent sessions rendered as documents.

A session is a fake one on disk in the records' own layout, and the template is a stub that writes each turn's reasoning where a test can see it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsa import assets
from lsa.data.tokenize import prepare
from lsa.data.trajectories import collect_session, run_name

SHARD = "qwen-1m-tc-keptcot-s3-20260913T061723Z"
RUN = "qwen-1m-tc-keptcot-20260913T061723Z"
SESSION = "455e8dee-555d-5651-a214-1a9bcda896ec"
AGENT = "You are Qwen Code, a non-interactive CLI agent."
SUMMARIZER = "You are the component that summarizes a conversation."


class StubTokenizer:
    """One word per token, and a template that writes reasoning before content."""

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, clear_thinking):
        turns = []
        for message in messages:
            reasoning = message.get("reasoning_content") or ""
            calls = " ".join(
                call["function"]["arguments"]["command"] for call in message.get("tool_calls") or []
            )
            turns.append(f"<{message['role']}> {reasoning} {message.get('content') or ''} {calls}")
        return " ".join(turns)

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=text.split())


def assistant(content, reasoning, command=None):
    calls = [
        {
            "id": "c",
            "type": "function",
            "function": {"name": "run", "arguments": json.dumps({"command": command})},
        }
    ]
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": calls if command else [],
    }


def write_record(completions, name, system, history, response, prompt_tokens):
    record = {
        "completion_id": name,
        "model_used": "Qwen/Qwen3.8-27B",
        "original_request": {
            "model": "m",
            "messages": [{"role": "system", "content": system}, *history],
            "tools": [],
        },
        "response": {
            "choices": [{"message": response, "prompt_token_ids": [1] * prompt_tokens}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3},
        },
    }
    (completions / name).write_text(json.dumps(record))


@pytest.fixture
def session(tmp_path):
    """A session that compacts once, with a side call before the collapse and an empty record last"""
    root = tmp_path / SHARD / "rollout" / "task_apple_game" / "sessions" / SESSION
    completions = root / "completions"
    completions.mkdir(parents=True)
    task = {"role": "user", "content": "solve the task"}
    early = [
        task,
        assistant("looked", "early thought " * 50, command="ls"),
        {"role": "tool", "content": "out " * 100},
    ]

    write_record(
        completions,
        "0001-msg_a.json",
        AGENT,
        early,
        assistant("before collapse", "last thought", command="make"),
        1000,
    )
    write_record(
        completions, "0002-msg_b.json", SUMMARIZER, early, assistant("side call " * 500, "summary"), 1200
    )
    snapshot = [{"role": "user", "content": "<state_snapshot> done so far </state_snapshot>"}]
    write_record(completions, "0003-msg_c.json", AGENT, snapshot, assistant("resumed", "fresh"), 100)
    write_record(completions, "0004-msg_d.json", AGENT, snapshot, assistant("final", "end"), 150)
    (completions / "0005-msg_e.json").write_bytes(b"")
    return root


def test_a_session_is_its_longest_call_before_a_compaction(session):
    document = collect_session(StubTokenizer(), session, assets.source("edgebench"))

    assert "before collapse" in document.text
    assert "final" not in document.text
    assert "side call" not in document.text
    assert document.model == "Qwen/Qwen3.8-27B"


def test_the_document_id_names_task_run_and_session(session):
    document = collect_session(StubTokenizer(), session, assets.source("edgebench"))

    assert document.doc_id == f"apple_game:{RUN}/{SESSION}"
    assert (document.task, document.run, document.source) == ("apple_game", RUN, "edgebench")


def test_as_generated_drops_only_the_history_reasoning(session):
    document = collect_session(StubTokenizer(), session, assets.source("edgebench"))

    assert "early thought" in document.text
    assert "last thought" in document.text
    assert document.n_tokens - document.n_tokens_as_generated == 100


def test_the_run_is_the_shard_without_its_index():
    assert run_name(Path(f"/logs/{SHARD}")) == RUN
    assert run_name(Path("/logs/flashctx-qwen-1m-s2-20260902T175426Z")) == "flashctx-qwen-1m-20260902T175426Z"


@pytest.mark.parametrize("name", ["arxiv", "games"])
def test_a_source_collected_elsewhere_has_nothing_to_prepare_here(name):
    """From the hub, or from another directory on the cluster"""
    with pytest.raises(ValueError, match="predates this repository"):
        prepare(assets.dataset(name))


def test_a_benchmark_has_nothing_to_prepare():
    with pytest.raises(ValueError, match="the harness that reads it prepares it"):
        prepare(assets.dataset("ruler"))


def test_the_trajectories_need_the_runs_they_are_collected_from():
    with pytest.raises(ValueError, match="--sessions"):
        prepare(assets.dataset("edgebench"))


def test_a_tokenized_card_tokenizes_only_its_own_sources():
    with pytest.raises(ValueError, match="has no source"):
        prepare(assets.dataset("qwen3_5"), sources=["edgebench"])
