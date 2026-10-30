"""Count the requests a server decoded together at each context of a production run, from the server's own log.

The client's count of requests in flight bins them into whole seconds, so two requests that run back to back
inside one second read as two. The server's `#running-req` on its "Decode batch" lines is the batch it actually
ran, so that is what is counted.

The client flushes the server's cache after its warmup and before the requests it measures, once per context.
Those flushes mark where each measured stretch starts, in the order the contexts ran, and the context's own
duration says where it ends. A window is its flush to its flush plus its duration and `WINDOW_SLACK_SECONDS`, so
the warmup before it and the next context's warmup after it fall outside.

A log line carries the rank that wrote it. Every rank of a context-parallel or pipeline-parallel server logs the
same batch, so the count at one second is the highest any rank reports, and a context's count is the highest of its
seconds. A server with data-parallel attention runs a different batch per rank and would be undercounted - none of
the recorded runs is one. A decode step is logged every few dozen steps, so a wave that decoded for fewer steps
than that can be missed.

A count is None whenever the log cannot say it: no log, no flush markers, a different number of flushes than
contexts, a flush before the previous context could have finished, or no decode line inside the window. The
client's own count is never a stand-in.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

# A whole-second log stamp, then the ranks that wrote the line, then the message
LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: [^\]]*)?\] (.*)$")
DECODE = re.compile(r"Decode batch, #running-req: (\d+)")
STAMP = "%Y-%m-%d %H:%M:%S"

# What a flush logs - on a hybrid model, one line per rank
FLUSH_MARKERS = ("Reset HybridReqToTokenPool", "Cache flushed")

# Flush lines of one flush land within a few seconds of each other
FLUSH_GAP_SECONDS = 5

# A log stamp is a whole second and a duration is the client's own clock
WINDOW_SLACK_SECONDS = 2


def read_log(path: Path) -> tuple[list[datetime], dict[datetime, int]]:
    """Read the flush times and the requests running at each second from a server log.

    Args:
        path: The server's log.

    Returns:
        The time of each flush in order, and the most requests running at each second any rank logged.
    """
    flushes: dict[str, list[datetime]] = {marker: [] for marker in FLUSH_MARKERS}
    running: dict[datetime, int] = {}
    with path.open(errors="replace") as lines:
        for line in lines:
            if "Decode batch" not in line and not any(marker in line for marker in FLUSH_MARKERS):
                continue
            match = LINE.match(line.rstrip("\n"))
            if not match:
                continue
            stamp, message = datetime.strptime(match.group(1), STAMP), match.group(2)
            decode = DECODE.match(message)
            if decode:
                running[stamp] = max(running.get(stamp, 0), int(decode.group(1)))
                continue
            for marker in FLUSH_MARKERS:
                if message.startswith(marker):
                    flushes[marker].append(stamp)
    marked = next((times for times in flushes.values() if times), [])
    starts: list[datetime] = []
    for time in sorted(marked):
        if not starts or time - starts[-1] > timedelta(seconds=FLUSH_GAP_SECONDS):
            starts.append(time)
    return starts, running


def wave_peaks(path: Path, durations: list[float]) -> list[int | None]:
    """Count the requests decoded together at each context, in the order the contexts ran.

    Args:
        path: The server's log.
        durations: Each context's measured seconds, in the order they ran.

    Returns:
        The most requests running together in each context's window, None where the log cannot say.
    """
    nothing: list[int | None] = [None] * len(durations)
    if not path.is_file() or not durations:
        return nothing
    starts, running = read_log(path)
    if len(starts) != len(durations):
        return nothing
    ends = [
        start + timedelta(seconds=duration + WINDOW_SLACK_SECONDS)
        for start, duration in zip(starts, durations, strict=True)
    ]
    if any(
        end - timedelta(seconds=WINDOW_SLACK_SECONDS) > following
        for end, following in zip(ends, starts[1:], strict=False)
    ):
        return nothing
    peaks: list[int | None] = []
    for start, end in zip(starts, ends, strict=True):
        inside = [count for time, count in running.items() if start <= time <= end]
        peaks.append(max(inside) if inside else None)
    return peaks
