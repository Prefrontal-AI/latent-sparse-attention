"""Sample the GPUs a performance run measures on, so a record says what clocks and power it ran under.

    <samples>/<point>.gpu-<host>.csv      nvidia-smi's own rows over the point's timed window, one file a node

The clocks of a shared card cannot be locked without root, so a timing is only as steady as the card was, and
the only honest answer is to write down what the card did. A sampler runs `nvidia-smi` in the background for
the window and its raw rows stay beside the point's samples. `summarize` folds them into the facts a record keeps,
with no verdict - what a reader makes of a clock that dipped is theirs.

A throttle reason counts when it is anything but idle or the application-clocks setting, which a card holds
whenever it is not busy or its clocks were set by hand. The power cap is kept apart from the rest - a card at its
own limit is working as built, and any other reason is a slowdown its environment imposed. This module reads no GPU itself and imports nothing but
the standard library, because it runs inside the serving container too.
"""

import contextlib
import csv
import glob
import io
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

# One row per GPU per interval
FIELDS: tuple[str, ...] = (
    "timestamp",
    "index",
    "uuid",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks_throttle_reasons.active",
)

INTERVAL_MS = 100

# NVML's clock event reason bits
REASONS: dict[int, str] = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}

# Held by a card at rest or with its clocks set by hand, not a card slowed down
IGNORED: frozenset[str] = frozenset({"gpu_idle", "applications_clocks_setting"})

# A card at its own power limit under a power-bound kernel, which is the card working and not the environment
POWER_CAP = "sw_power_cap"

Reading = dict[str, Any]


def command(uuids: Sequence[str] = ()) -> list[str]:
    """Build the nvidia-smi command that samples until it is stopped.

    Args:
        uuids: The cards to sample, every visible card when empty.

    Returns:
        The argv.
    """
    argv = [
        "nvidia-smi",
        f"--query-gpu={','.join(FIELDS)}",
        "--format=csv,nounits",
        f"-lms={INTERVAL_MS}",
    ]
    if uuids:
        argv.append(f"--id={','.join(uuids)}")
    return argv


def reason_names(mask: int) -> list[str]:
    """Name the throttle reasons a bitmask holds, in bit order."""
    return [name for bit, name in REASONS.items() if mask & bit]


def number(text: str) -> float | None:
    """Read one csv cell as a number, None for `[N/A]` or anything else that is not one."""
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse(text: str) -> list[Reading]:
    """Read the rows nvidia-smi wrote, leaving out its header lines and any row cut short.

    A sampler that is stopped mid-write leaves a last line that is partial, and a loop may repeat the header.

    Args:
        text: The csv a sampler wrote.

    Returns:
        One reading per complete row - `uuid`, `index`, `sm_mhz`, `mem_mhz`, `power_w`, `power_limit_w`,
        `temperature_c` and `reasons`, the names set in the throttle bitmask. A cell that was not a number is None.
    """
    readings = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) != len(FIELDS) or row[0].strip() == FIELDS[0]:
            continue
        cells = [cell.strip() for cell in row]
        try:
            mask = int(cells[8], 16)
        except ValueError:
            continue
        readings.append(
            {
                "uuid": cells[2],
                "index": int(cells[1]) if cells[1].isdigit() else None,
                "sm_mhz": number(cells[3]),
                "mem_mhz": number(cells[4]),
                "power_w": number(cells[5]),
                "power_limit_w": number(cells[6]),
                "temperature_c": number(cells[7]),
                "reasons": reason_names(mask),
            }
        )
    return readings


def summarize(readings: Iterable[Reading]) -> dict[str, dict[str, Any]]:
    """Fold readings into the facts a record keeps, one block per card.

    Args:
        readings: What `parse` returned, of one or several files.

    Returns:
        Each card's uuid to its sample count, minimum and median SM clock, median memory clock, maximum power
        and its limit, maximum temperature, the fraction of samples at the power cap, the fraction with any
        other reason set besides idle and the application-clocks setting - a slowdown - with that fraction per slowdown
        reason, and the fraction per reason of either kind.
    """
    by_card: dict[str, list[Reading]] = {}
    for reading in readings:
        by_card.setdefault(reading["uuid"], []).append(reading)
    summary = {}
    for uuid, rows in sorted(by_card.items()):
        clocks = [row["sm_mhz"] for row in rows if row["sm_mhz"] is not None]
        memory = [row["mem_mhz"] for row in rows if row["mem_mhz"] is not None]
        power = [row["power_w"] for row in rows if row["power_w"] is not None]
        limits = [row["power_limit_w"] for row in rows if row["power_limit_w"] is not None]
        heat = [row["temperature_c"] for row in rows if row["temperature_c"] is not None]
        counted = [[name for name in row["reasons"] if name not in IGNORED] for row in rows]
        slowdowns = [[name for name in reasons if name != POWER_CAP] for reasons in counted]
        names = sorted({name for reasons in counted for name in reasons})
        summary[uuid] = {
            "index": rows[0]["index"],
            "samples": len(rows),
            "sm_mhz_min": min(clocks) if clocks else None,
            "sm_mhz_median": statistics.median(clocks) if clocks else None,
            "mem_mhz_median": statistics.median(memory) if memory else None,
            "power_w_max": max(power) if power else None,
            "power_limit_w": max(limits) if limits else None,
            "temperature_c_max": max(heat) if heat else None,
            "power_capped_fraction": round(
                sum(1 for reasons in counted if POWER_CAP in reasons) / len(rows), 4
            ),
            "slowdown_fraction": round(sum(1 for reasons in slowdowns if reasons) / len(rows), 4),
            "slowdown_reasons": {
                name: round(sum(1 for reasons in slowdowns if name in reasons) / len(rows), 4)
                for name in sorted({name for reasons in slowdowns for name in reasons})
            },
            "throttle_reasons": {
                name: round(sum(1 for reasons in counted if name in reasons) / len(rows), 4) for name in names
            },
        }
    return summary


def sample_path(stem: Path, host: str) -> Path:
    """Return where one node's rows of a point go, beside the point's samples as `<stem>.gpu-<host>.csv`."""
    return stem.with_name(f"{stem.name}.gpu-{host}.csv")


def read_point(stems: Iterable[Path]) -> dict[str, Any] | None:
    """Summarize every node's rows of one point, over all the files it wrote.

    A prefill window measures several chunks and writes a file for each, and a card's summary is over the rows of
    them all.

    Args:
        stems: The point's sample paths without their suffix, which `sample_path` extends.

    Returns:
        The hosts, each card's summary with the host it sat on, and the csv files read, by name - or None where
        the point has no rows, as one measured before the sampler existed.
    """
    files = sorted(path for stem in stems for path in stem.parent.glob(f"{glob.escape(stem.name)}.gpu-*.csv"))
    readings: dict[str, list[Reading]] = {}
    for path in files:
        host = path.name.split(".gpu-", 1)[1][: -len(".csv")]
        readings.setdefault(host, []).extend(parse(path.read_text()))
    gpus: dict[str, Any] = {}
    for host, rows in readings.items():
        for uuid, card in summarize(rows).items():
            gpus[uuid] = {"host": host, **card}
    if not gpus:
        return None
    return {
        "hosts": sorted({card["host"] for card in gpus.values()}),
        "gpus": gpus,
        "files": [path.name for path in files],
    }


class Sampler:
    """Run nvidia-smi over a window and keep its rows, as a context manager.

    A node that fails to start nvidia-smi leaves an empty file rather than failing the measurement, because the
    timing is the thing being measured and the telemetry is only beside it.
    """

    def __init__(self, path: Path, uuids: Sequence[str] = (), prefix: Sequence[str] = ()) -> None:
        """Set where the rows go and which cards.

        Args:
            path: The csv to write.
            uuids: The cards to sample, every visible card when empty.
            prefix: A command to put in front of nvidia-smi's, such as an `srun` step for another node.
        """
        self.path = path
        self.argv = [*prefix, *command(uuids)]
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "Sampler":
        """Start sampling, or do nothing if the command cannot run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as out:
            try:
                self.process = subprocess.Popen(self.argv, stdout=out, stderr=subprocess.DEVNULL)
            except OSError:
                self.process = None
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Stop the sampling process, killing it if it does not end in time."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def sample_nodes(stem: Path, hosts: Sequence[str], here: str, gpus: int) -> contextlib.ExitStack:
    """Start a sampler on every node of a run and return the stack that stops them.

    This node's runs directly and another's through an `srun` step that overlaps the job's own, one task on the node.

    Args:
        stem: The point's sample path without its suffix.
        hosts: Every node the run's servers span.
        here: This node's name.
        gpus: The GPUs the run's servers hold on each node, which a step must ask for to see them.

    Returns:
        An open stack - leave it to stop the samplers.
    """
    stack = contextlib.ExitStack()
    for host in hosts:
        step = (
            []
            if host == here
            else [
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={host}",
                "--mem=0",
                "--cpu-bind=none",
                f"--gres=gpu:{gpus}",
            ]
        )
        stack.enter_context(Sampler(sample_path(stem, host), prefix=step))
    return stack
