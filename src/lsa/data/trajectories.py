"""Render the agent trajectories a harness recorded into the documents one source card describes.

A shard holds `rollout/task_<task>/sessions/<session>/completions/NNNN-msg_<id>.json`, one record per model call
in call order. One session becomes one document: its main-thread call whose rendering is longest, with that call's
response appended as the final assistant turn, through the teacher's own chat template and with every
assistant turn's reasoning kept. Nothing else is stripped - harness reminders and nudges are context the
model really read.

The main thread is every call whose system prompt is the agent's own, and everything else is a side call.
Auto-compaction collapses the prompt to a summary and starts a new segment, so the candidates are the last
main-thread call before each collapse and the session's last one, and only those are rendered. A record runs
to tens of megabytes because it carries token ids, so a call is classified from its first bytes and sized
from its last `usage` block, and only a candidate is ever parsed.
"""

import json
import mmap
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from lsa import assets
from lsa.data.constants import Collected
from lsa.data.tokenizer import load_tokenizer

# Agent's own system prompt, within the record's first bytes
MAIN_THREAD = b'"messages": [{"role": "system", "content": "You are Qwen Code'
HEAD_BYTES = 4096

# Response usage - last in the record, at the end or before the prompt's token ids
USAGE_KEY = b'"usage": {"prompt_tokens": '
USAGE = re.compile(rb'"usage": \{"prompt_tokens": (\d+)')

# Prompt shorter than this share of the previous one
# => compaction
COMPACTION_RATIO = 0.2

DOCUMENTS_PER_SHARD = 16
SHARD_INDEX = re.compile(r"-s\d+-")


class Trajectories:
    """The agent trajectories this repository renders itself, which their source card does not say"""

    # The model whose chat template renders a session, which is the teacher that reads it
    TEMPLATE: str = "glm53_flash"


@dataclass(frozen=True)
class Call:
    """One main-thread model call, sized without being parsed."""

    path: Path
    prompt_tokens: int


@dataclass(frozen=True)
class Document:
    """One session rendered, as the columns of a collected shard."""

    doc_id: str
    source: str
    license: str
    # GLM tokens with every assistant turn's reasoning
    n_tokens: int
    # GLM tokens with the final turn's reasoning alone
    n_tokens_as_generated: int
    model: str
    task: str
    run: str
    text: str


def collect_trajectories(card: assets.SourceCard, shards: list[Path], workers: int = 32) -> Path:
    """Render every session under the run directories as the documents one source card describes.

    Args:
        card: The source the sessions become, which names its folder and its licence.
        shards: The harness's run shard directories.
        workers: Sessions rendered at once.

    Returns:
        The source's folder, now holding its shards.
    """
    template = assets.model(Trajectories.TEMPLATE).checkpoint
    sessions = sorted(session for shard in shards for session in shard.glob("rollout/task_*/sessions/*"))
    print(f"rendering {len(sessions)} sessions from {len(shards)} shards", flush=True)
    documents = []
    with ProcessPoolExecutor(workers) as pool:
        rendered = pool.map(collect_session_at, [template] * len(sessions), sessions, [card] * len(sessions))
        for session, document in zip(sessions, rendered, strict=True):
            if document is None:
                print(f"  {session}: no main-thread call", flush=True)
                continue
            documents.append(document)
            print(
                f"  {document.doc_id}: {document.n_tokens} tokens, {document.n_tokens_as_generated} as generated",
                flush=True,
            )

    documents.sort(key=lambda document: document.doc_id)
    folder = Path(card.path)
    write_shards(documents, folder)
    print(f"{len(documents)} documents written to {folder}")
    return folder


def run_name(shard: Path) -> str:
    """Name the run a shard belongs to - its directory name without the shard index."""
    return SHARD_INDEX.sub("-", shard.name, count=1)


def size_call(path: Path) -> Call | None:
    """Classify one record and read its prompt length.

    Args:
        path: A completion record.

    Returns:
        The call, or None for an empty record, a side call or a call with no usage.
    """
    if path.stat().st_size == 0:
        return None
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as record:
        if record[:HEAD_BYTES].find(MAIN_THREAD) < 0:
            return None
        start = record.rfind(USAGE_KEY)
        usage = USAGE.match(record[start : start + 64]) if start >= 0 else None
    if usage is None:
        return None
    return Call(path=path, prompt_tokens=int(usage.group(1)))


def pick_candidates(calls: list[Call]) -> list[Call]:
    """Keep the last call before each compaction and the session's last call.

    Args:
        calls: The session's main-thread calls, in call order.

    Returns:
        The calls whose rendering could be the session's longest.
    """
    if not calls:
        return []
    before_compaction = [
        call
        for call, following in zip(calls, calls[1:], strict=False)
        if following.prompt_tokens < COMPACTION_RATIO * call.prompt_tokens
    ]
    return [*before_compaction, calls[-1]]


def arguments_of(function: dict[str, Any]) -> dict[str, Any]:
    """Return a tool call's arguments as the mapping the template iterates.

    The records carry them as the JSON string the API sends.
    One that does not parse to an object is kept whole under one key, so the text the model wrote survives.
    """
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"arguments": arguments}


def assistant_turn(message: dict[str, Any], keep_reasoning: bool) -> dict[str, Any]:
    """Normalise one assistant message for the template.

    Reasoning arrives as `reasoning_content`, `reasoning` or both.
    An empty string rather than an absent key is what makes the template render an empty think block instead of looking for one inside the content.
    """
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    tool_calls = [
        {**call, "function": {**call["function"], "arguments": arguments_of(call["function"])}}
        for call in message.get("tool_calls") or []
    ]
    return {
        "role": "assistant",
        "content": message.get("content"),
        "reasoning_content": reasoning if keep_reasoning else "",
        "tool_calls": tool_calls,
    }


def conversation(record: dict[str, Any], keep_history_reasoning: bool) -> list[dict[str, Any]]:
    """Return the messages a teacher reads - the request's, then its response as the final turn.

    Args:
        record: One parsed completion record.
        keep_history_reasoning: Keep the reasoning of the assistant turns before the response.

    Returns:
        The messages in order, the response keeping its reasoning either way.
    """
    history = [
        assistant_turn(message, keep_history_reasoning) if message["role"] == "assistant" else message
        for message in record["original_request"]["messages"]
    ]
    response = record["response"]["choices"][0]["message"]
    return [*history, assistant_turn(response, keep_reasoning=True)]


def render(tokenizer: Any, record: dict[str, Any], keep_history_reasoning: bool) -> tuple[str, int]:
    """Render one call through the chat template and count its tokens.

    GLM's template keeps every turn's reasoning under `clear_thinking=False`.
    Its `clear_thinking=True` drops only what precedes the last user message, which in an agent session is little, so history is dropped on the messages instead.

    Args:
        tokenizer: A tokenizer carrying the teacher's chat template.
        record: One parsed completion record.
        keep_history_reasoning: Keep the reasoning of the assistant turns before the response.

    Returns:
        The rendered text, and its length in tokens as the tokenizer counts it.
    """
    text = tokenizer.apply_chat_template(
        conversation(record, keep_history_reasoning),
        tools=record["original_request"].get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
        clear_thinking=False,
    )
    return text, len(tokenizer(text, add_special_tokens=False).input_ids)


def read_record(path: Path) -> dict[str, Any]:
    """Parse one completion record whole."""
    record: dict[str, Any] = json.loads(path.read_bytes())
    return record


def collect_session(tokenizer: Any, session: Path, card: assets.SourceCard) -> Document | None:
    """Render one session as its longest main-thread call.

    Args:
        tokenizer: A tokenizer carrying the teacher's chat template.
        session: A `rollout/task_<task>/sessions/<session>` directory.
        card: The source the session becomes a document of.

    Returns:
        The document, or None for a session with no main-thread call.
    """
    completions = sorted((session / "completions").glob("*.json"))
    candidates = pick_candidates([call for call in map(size_call, completions) if call is not None])
    if not candidates:
        return None

    rendered = {
        candidate.path: render(tokenizer, read_record(candidate.path), keep_history_reasoning=True)
        for candidate in candidates
    }
    longest = max(rendered, key=lambda path: rendered[path][1])
    text, n_tokens = rendered[longest]
    record = read_record(longest)
    _, n_tokens_as_generated = render(tokenizer, record, keep_history_reasoning=False)

    task = session.parents[1].name.removeprefix("task_")
    run = run_name(session.parents[3])
    return Document(
        doc_id=f"{task}:{run}/{session.name}",
        source=card.name,
        license=card.licence,
        n_tokens=n_tokens,
        n_tokens_as_generated=n_tokens_as_generated,
        model=record["model_used"],
        task=task,
        run=run,
        text=text,
    )


@cache
def cached_tokenizer(path: str) -> Any:
    """Open a tokenizer once per worker process."""
    return load_tokenizer(path, require_fast=True)


def collect_session_at(tokenizer_path: str, session: Path, card: assets.SourceCard) -> Document | None:
    """Render one session in a worker, with the tokenizer named by its path."""
    return collect_session(cached_tokenizer(tokenizer_path), session, card)


def write_shards(documents: list[Document], folder: Path) -> None:
    """Write documents as collected parquet shards and mark the source done.

    Args:
        documents: Every document of the source, in the order the shards hold them.
        folder: The source's folder of shards.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(documents), DOCUMENTS_PER_SHARD)):
        rows = [asdict(document) for document in documents[start : start + DOCUMENTS_PER_SHARD]]
        pq.write_table(pa.Table.from_pylist(rows), folder / f"shard_{index:05d}.parquet")
    (folder / Collected.DONE_MARKER).touch()
