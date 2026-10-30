"""Files every package reads and writes the same way - atomic json, jsonl rows, a content hash, the checkout's sha and a claim file.

A write goes to a scratch file beside its target and is renamed over it, so a reader never sees half a file
and two processes rewriting one file leave one of the two whole - on one node or on two, since the scratch
name carries the host, the process and the thread.
"""

import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

# Marks a `read_json` caller who named no default, since None is a default some callers want
_REQUIRED: Any = object()


def write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> None:
    """Write json to a file atomically, creating its parents.

    Args:
        path: Destination file.
        payload: Anything `json.dumps` accepts.
        indent: Spaces per nesting level.
    """
    with _scratch(path) as target:
        target.write(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")


def read_json(path: str | os.PathLike[str], default: Any = _REQUIRED) -> Any:
    """Read a whole json file.

    Args:
        path: Source file.
        default: What to return when there is no file. Without one, a missing file raises.

    Returns:
        The parsed payload, or `default` when there is no file.

    Raises:
        FileNotFoundError: There is no file at `path` and no default was given.
    """
    if default is not _REQUIRED and not os.path.isfile(path):
        return default
    with open(path) as source:
        return json.load(source)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> None:
    """Write one json row per line to a file atomically, creating its parents.

    Args:
        path: Destination file.
        rows: Anything `json.dumps` accepts, one per line.
    """
    with _scratch(path) as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Read every non-blank line of a jsonl file.

    Args:
        path: Source file.

    Returns:
        The parsed rows in file order, or an empty list when there is no file.
    """
    if not os.path.exists(path):
        return []
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def utc_now(timespec: str = "seconds") -> str:
    """Return the UTC time now, with its offset, as a record spells a timestamp.

    Args:
        timespec: What `datetime.isoformat` takes - `"seconds"` for an evaluation's records, `"auto"` for a
            run's metadata, which keeps the microseconds.

    Returns:
        The iso timestamp.
    """
    return datetime.now(UTC).isoformat(timespec=timespec)


def file_sha256(path: str | os.PathLike[str]) -> str | None:
    """Hash a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex sha256, or None when there is no file at the path.
    """
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(directory: str | os.PathLike[str] | None = None) -> str | None:
    """Return the short sha of a checkout.

    Args:
        directory: Any path inside the checkout, or None for the working directory.

    Returns:
        The short sha, or None outside a repository.
    """
    command = ["git", "rev-parse", "--short", "HEAD"]
    if directory is not None:
        command[1:1] = ["-C", str(directory)]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(directory: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the commit a checkout is at, in full, and whether its tracked files differ from it.

    A run's metadata keeps this rather than the short sha, because a short sha is not unique for long and a
    run from a checkout with uncommitted edits ran code that no commit holds.

    Args:
        directory: Any path inside the checkout, or None for the working directory.

    Returns:
        `{"sha": ..., "dirty": ...}`, with an empty sha outside a repository.
    """
    command = ["git"] + (["-C", str(directory)] if directory is not None else [])
    try:
        sha = subprocess.check_output(
            [*command, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        changed = subprocess.check_output(
            [*command, "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "", "dirty": False}
    return {"sha": sha, "dirty": bool(changed.strip())}


class Claim:
    """One process's hold on a piece of work, kept as a file so several processes may share it without being handed slices.

    A slice belongs to whichever job was given it, so killing that job orphans its slice until somebody
    works out which one it was. A claim is a file, taken before the work starts and released after, and a
    claim nobody refreshed within `stale_after` seconds is taken over - so the survivors finish what a
    killed job was holding.
    """

    def __init__(self, path: Path, stale_after: float) -> None:
        """Hold the claim file for one piece of work.

        Args:
            path: Where the claim lives.
            stale_after: Seconds without a refresh after which the claim is nobody's.
        """
        self.path = path
        self.stale_after = stale_after

    def take(self) -> bool:
        """Try to become the process responsible for this work.

        Returns:
            True when this process took it, False when something alive already holds it.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._fresh():
                return False
            # Whoever held this is gone - take it over rather than leaving the work undone
            self.release()
            return self.take()
        os.write(handle, f"{os.getpid()}@{socket.gethostname()} {time.time()}\n".encode())
        os.close(handle)
        return True

    def _fresh(self) -> bool:
        """Return whether the existing claim is young enough to still be someone's."""
        try:
            return time.time() - self.path.stat().st_mtime < self.stale_after
        except FileNotFoundError:
            return False

    def release(self) -> None:
        """Give the claim up, whether the work was done or failed."""
        self.path.unlink(missing_ok=True)


@contextmanager
def _scratch(path: str | os.PathLike[str]) -> Iterator[TextIO]:
    """Open a scratch file beside a target, and rename it over the target once the block finishes cleanly.

    Args:
        path: Destination file, whose parents are created.

    Yields:
        The scratch file, open for text.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Named by host, process and thread, since workers on two nodes can share a pid and a thread a process
    scratch = target.with_name(
        f"{target.name}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(scratch, "w") as handle:
            yield handle
        os.replace(scratch, target)
    finally:
        scratch.unlink(missing_ok=True)
