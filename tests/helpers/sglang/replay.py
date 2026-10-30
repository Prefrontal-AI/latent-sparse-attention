"""Replay the freeze's goldens on the bucket cache and compare - the table of design by input.

The goldens (`data/goldens` under the shared root, see its README) are the key ids each design selected on the
freeze, given fixed unit index keys and index queries. A replay rebuilds the store from the same inputs in the same
order - advance to a chunk's start, read the chunk, settle, then sixteen decode steps - and a read matches when its
sorted ids are equal. Only the rows the manifest lists as unstable may differ, and by at most a bucket, and the rows
where two buckets score exactly alike and the exact pick reads the lower first.
The one change from the freeze is the trees' read just past a growth (invariant 19): a row the golden read short
may come out full, with every id the golden had.

    python -m tests.helpers.sglang.replay [--designs ...] [--inputs ...]

Runs on a GPU, in the serving image the goldens were taken in or in the uv env, whose Triton builds partition alike.
Every top-k the reads run is held to a stable sort of the same scores, so a row that parts from its golden is shown
to come from an exact pick; `--no-audit` leaves the picks unchecked.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F

from lsa import paths
from lsa.core.kernels.triton.keys import unpack_ids
from lsa.core.kernels.triton.top_k import TopK
from lsa.sglang.index import weights
from lsa.sglang.index.layer_index import KeyAndQueryProjection
from lsa.sglang.index.selector import Selector
from lsa.sglang.settings.design import DesignSettings, Shape

GOLDENS = paths.goldens()
# The request row, rows 0 and 2 of the 3-row table staying empty
ROW = 1
REQUEST_ROWS = 3
# The served chunked-prefill-size
CHUNK = 8192
DECODE_STEPS = 16
# A prefill read keeps every STRIDE-th row of its chunk and the last
STRIDE = 16
# Most ids a row on the freeze's radix cut may differ by a side - one bucket
BUCKET_CAP = 16
# Rows of scores the audit sorts at a time
AUDIT_ROWS = 64

BASE_INDEXER = {
    "index_dim": 64,
    "pool_size": 4,
    "bucket_size": 8,
    "bucket_cap": 16,
    "top_k": 4096,
    "top_p": 1024,
}
FIXED_INDEXER = {"top_level": 1, "node_bucket_size": 4, "node_cluster_block_size": 1024, "beam_width": 2048}
TREE_INDEXER = {
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 4096,
    "beam_width": 2048,
    "start_nodes": 32768,
}

# The node bucket shape of tree (node shape) - rank 2, z 2, a sign basis over the keys beneath
NODE_SHAPE = {"rank": 2, "deviations": 2.0}

# Design name -> (model.indexer, model.serving) blocks
DESIGNS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "flat": (BASE_INDEXER, {}),
    "fixed": ({**BASE_INDEXER, **FIXED_INDEXER}, {"node_centroid_precision": "bf16"}),
    "tree": ({**BASE_INDEXER, **TREE_INDEXER}, {"node_centroid_precision": "fp8"}),
    "flat_k2048": ({**BASE_INDEXER, "top_k": 2048, "top_p": 512}, {}),
    "flat_no_pooling": ({**BASE_INDEXER, "pool_size": None}, {}),
    "tree_descend": (
        {**BASE_INDEXER, **TREE_INDEXER},
        {"node_centroid_precision": "fp8", "decode": "descend"},
    ),
    "tree_node_shape": (
        {**BASE_INDEXER, **TREE_INDEXER, "node_bucket_shape": NODE_SHAPE},
        {"node_centroid_precision": "fp8", "shape_basis": "sign"},
    ),
}
INPUTS = (
    "synth-64k",
    "synth-256k",
    "synth-512k",
    "synth-2m",
    "synth-6m",
    "probe-multikey2-1m-l35",
    "probe-cwe-512k-l11",
)
# The real probes' sources - the freeze's probe of GLM-5.3-Flash, a layer's key latents and each head's absorbed
# queries, and the map it served, re-exported for the key and query input
PROBES = Path(paths.under_root("experiments/relax-buckets/probe"))
LATENT_MAP = paths.maps() / "glm53_flash__long-16k-1M-latent.pt"
PROBE_SOURCES = {
    "probe-multikey2-1m-l35": ("ruler__niah_multikey_2_1000000_s0", 35),
    "probe-cwe-512k-l11": ("ruler__cwe_524288_s0", 11),
}
# How the goldens drew a probe's queries: the tensor-parallel ranks the heads split over, the noise the rows off the
# stride took, its seed, the stride a chunk's index into the query pool moves by, and the decode steps' stride
PROBE_RANKS = 8
PROBE_NOISE = 0.02
PROBE_SEED = 7
PROBE_CHUNK_STRIDE = 7919
PROBE_DECODE_STRIDE = 13
# The layers GLM-5.3-Flash runs an indexer at, which a map holds
GLM_INDEXER_LAYERS = range(3, 44, 4)
# Key latents projected at a time, as the goldens did
PROBE_PIECE = 65536
# The manifest's spelling of the design's knobs, where it differs from the code names
MANIFEST_NAMES = {"bucket_size_avg": "bucket_size", "bucket_size_max": "bucket_cap"}
# The designs that grow a hierarchy. The freeze's tree read short just past a growth (invariant 19), so on these a row
# the golden read short may now come out full - the one change from the freeze a replay allows
TREES = ("fixed", "tree", "tree_descend", "tree_node_shape")


def design_settings(name: str) -> DesignSettings:
    """Build a design's settings from its table entry."""
    indexer, serving = DESIGNS[name]
    return DesignSettings.from_blocks(indexer, serving)


def manifest_disagreements(name: str, manifest: dict[str, Any]) -> list[str]:
    """List the knobs whose value differs between a design here and the manifest's record of it.

    Args:
        name: The design.
        manifest: The goldens' manifest.
    """
    recorded = manifest["designs"][name]["design_names"]
    settings = design_settings(name)
    mine = {**vars(settings.indexer), **vars(settings.serving), "window_size": settings.indexer.window_size}
    differing = []
    for key, value in recorded.items():
        key = MANIFEST_NAMES.get(key, key)
        here = mine.get(key)
        if isinstance(here, Shape):
            # The manifest spells a node shape as its rank, z and what it is formed over
            here = f"rank {here.rank}, z {here.deviations:g}, keys beneath"
        if key in mine and key != "beam_width" and isinstance(value, (int, str)) and here != value:
            differing.append(f"{key}: manifest {value!r}, here {here!r}")
    return differing


class Replayer:
    """A selector over a fixed key store, standing in for the engine around it."""

    def __init__(self, name: str, keys: torch.Tensor, device: str = "cuda") -> None:
        """Build the selector for a design over keys the request already holds in its KV slots.

        Args:
            name: The design.
            keys: The input's index keys [T, index_dim] bf16, KV slot equal to position.
            device: Where it runs.
        """
        self.keys = keys
        self.device = device
        total = keys.shape[0]
        self.table = (
            torch.arange(total, device=device, dtype=torch.int32).expand(REQUEST_ROWS, -1).contiguous()
        )
        self.selector = Selector(
            design_settings(name),
            kv_size=total,
            request_count=REQUEST_ROWS,
            max_context=total,
            prefill_chunk=CHUNK,
            device=device,
            slot_table=lambda: self.table,
        )

    def index_keys_of(self, row: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the lookup of a position's index key - the stored key, normalized again as the freeze does."""

        def lookup(positions: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.keys[self.table[row, positions.long()].long()].float(), dim=-1).to(
                torch.bfloat16
            )

        return lookup


class Audit:
    """Every top-k a selector runs, held to a stable descending sort of the same scores."""

    def __init__(self) -> None:
        self.read = ""
        self.calls = 0
        self.rows = 0
        # Rows whose picked columns are not the sort's, by the read they were picked in
        self.inexact: dict[str, int] = {}

    @contextmanager
    def active(self) -> Iterator[None]:
        """Check the best k of every listing a `TopK` makes inside the block."""
        listed, called = TopK.listed, TopK.__call__

        def checked_listing(pick: TopK, scores: torch.Tensor, lengths: torch.Tensor) -> Any:
            listing = listed(pick, scores, lengths)
            self.check(scores, lengths, unpack_ids(pick.best()))
            return listing

        def checked_keep(pick: TopK, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            kept = called(pick, scores, lengths)
            self.check(scores, lengths, kept, ascending=True)
            return kept

        with patch.object(TopK, "listed", checked_listing), patch.object(TopK, "__call__", checked_keep):
            yield

    def check(
        self, scores: torch.Tensor, lengths: torch.Tensor, ids: torch.Tensor, ascending: bool = False
    ) -> None:
        """Count the rows whose picked columns are not a stable sort's best - as a set, when they come ascending."""
        width = ids.shape[1]
        columns = torch.arange(scores.shape[1], device=scores.device)
        slots = torch.arange(width, device=scores.device)
        self.calls += 1
        self.rows += scores.shape[0]
        for first in range(0, scores.shape[0], AUDIT_ROWS):
            part = slice(first, first + AUDIT_ROWS)
            masked = scores[part].masked_fill(columns >= lengths[part, None], -torch.inf)
            best = masked.sort(dim=1, descending=True, stable=True).indices[:, :width]
            best = F.pad(best, (0, width - best.shape[1]))
            expected = torch.where(slots < lengths[part, None], best, -1).to(torch.int32)
            if ascending:
                held = torch.where(expected >= 0, expected, torch.iinfo(torch.int32).max).sort(dim=1).values
                expected = torch.where(held == torch.iinfo(torch.int32).max, -1, held)
            wrong = int((expected != ids[part]).any(1).sum())
            if wrong:
                self.inexact[self.read] = self.inexact.get(self.read, 0) + wrong


def row_sample(count: int) -> torch.Tensor:
    """The rows of a chunk a golden keeps - every STRIDE-th and the last."""
    rows = torch.arange(0, count, STRIDE)
    return torch.unique(torch.cat([rows, torch.tensor([count - 1])]))


def digest(replayer: Replayer) -> dict[str, Any]:
    """The partition the cache holds - a hash of its buckets as sorted key sets, and the hierarchy's shape."""
    import numpy as np

    cache = replayer.selector.cache
    visible = cache.visible(ROW)
    rows = cache.logical_rows(ROW, visible)[:visible]
    members = cache.bucket_members[rows].sort(dim=1, descending=True).values.cpu().numpy()
    order = np.lexsort(members.T[::-1])
    state = dict(buckets=int(visible), partition_sha256=hashlib.sha256(members[order].tobytes()).hexdigest())
    hierarchies = cache.hierarchies
    if hierarchies is None:
        return {**state, "tree": None}
    hierarchy = hierarchies[ROW]
    nodes = [
        len(hierarchy.open_node_buckets[i]) + len(hierarchy.closed_node_buckets[i])
        for i in range(hierarchy.levels)
    ]
    tree = dict(
        levels=hierarchy.levels,
        covered_buckets=hierarchy.covered_buckets,
        nodes_per_level=nodes,
        start_level=hierarchy.start_level() if hierarchy.levels else None,
    )
    return {**state, "tree": tree}


def replay(
    name: str, golden_input: dict[str, Any], device: str = "cuda", audit: Audit | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay one design over one input.

    Args:
        name: The design.
        golden_input: The input file's contents.
        device: Where it runs.
        audit: Where to hold every top-k to a sort, or None.

    Returns:
        The reads in the goldens' layout, and the cache's digest after the prefill and at the end.
    """
    meta = golden_input["meta"]
    keys = golden_input["keys"].to(device)
    replayer = Replayer(name, keys, device)
    selector, cache = replayer.selector, replayer.selector.cache
    index_keys_of = replayer.index_keys_of(ROW)
    active = {ROW}
    reads: list[dict[str, Any]] = []
    for chunk, start in enumerate(range(0, meta["prefill"], CHUNK)):
        stop = min(start + CHUNK, meta["prefill"])
        cache.ensure(ROW, "golden", chunk)
        if chunk not in golden_input["chunks"]:
            cache.advance(ROW, stop, index_keys_of, active)
            continue
        selector.maintain(["golden"], [ROW], [start], replayer.index_keys_of)
        if audit is not None:
            audit.read = f"prefill {chunk}"
        positions = torch.arange(start, stop, device=device, dtype=torch.int32)
        lists = selector.prefill(
            ROW, golden_input["chunks"][chunk].to(device), positions, index_keys_of, active
        )
        torch.cuda.synchronize()
        rows = row_sample(stop - start)
        reads.append(
            dict(
                kind="prefill",
                chunk=chunk,
                rows=rows,
                positions=positions.cpu()[rows],
                lists=lists.cpu()[rows],
            )
        )
    after_prefill = digest(replayer)
    for step in range(DECODE_STEPS):
        position = meta["prefill"] + step
        selector.maintain(["golden"], [ROW], [position], replayer.index_keys_of)
        if audit is not None:
            audit.read = f"decode {step}"
        lists = selector.decode(
            golden_input["decode"][step : step + 1].to(device),
            torch.tensor([position], device=device, dtype=torch.int32),
            torch.tensor([ROW], device=device),
        )
        torch.cuda.synchronize()
        reads.append(
            dict(
                kind="decode",
                step=step,
                rows=torch.zeros(1, dtype=torch.long),
                positions=torch.tensor([position], dtype=torch.int32),
                lists=lists.cpu(),
            )
        )
    return reads, dict(after_prefill=after_prefill, final=digest(replayer))


def read_key(read: dict[str, Any]) -> str:
    """Name a read as the manifest's unstable rows do."""
    return f"{read['kind']} {read['chunk'] if read['kind'] == 'prefill' else read['step']}"


@dataclass
class Comparison:
    """How a replay stands against its golden."""

    reads: int = 0
    rows: int = 0
    # Rows whose sorted ids differ, split by whether the manifest lists them as unstable and the cut allows it
    unstable_rows: int = 0
    # Rows the golden read short, which the replay fills to the budget with the same ids and more
    refilled_rows: int = 0
    mismatched_rows: int = 0
    # Rows that differ off the manifest's list, in a read whose every pick the audit found to be the exact top-k
    exact_rows: int = 0
    # Picks of the audited reads that parted from the sort, and the rows of them
    inexact_picks: int = 0
    # Reads whose shape or positions differ
    broken_reads: int = 0
    notes: list[str] = field(default_factory=list)
    # Rows of scores the audit sorted
    sorted_rows: int = 0

    @property
    def status(self) -> str:
        """exact, unstable-only, refilled, exact-picks or mismatch."""
        if self.broken_reads or self.mismatched_rows or self.inexact_picks:
            return "mismatch"
        if self.exact_rows:
            return "exact-picks"
        if self.refilled_rows:
            return "refilled"
        return "unstable-only" if self.unstable_rows else "exact"


def recorded_variants(path: Path) -> dict[tuple[str, int], list[list[int]]]:
    """Read the answers the freeze gave on its unstable rows, pass 1's and the later passes'.

    Args:
        path: The run's `.unstable.pt`, which only a run with unstable rows has.

    Returns:
        The sorted ids of each answer, by read and row of the chunk, for the first 64 rows of a read.
    """
    variants: dict[tuple[str, int], list[list[int]]] = {}
    if not path.exists():
        return variants
    for entry in torch.load(path, weights_only=False):
        key = read_key(entry)
        for index, row in enumerate(entry["rows"].tolist()):
            variants.setdefault((key, row), []).append(
                sorted(i for i in entry["pass1"][index].tolist() if i >= 0)
            )
        for later in entry["later"]:
            for index, row in enumerate(later["rows"].tolist()):
                variants.setdefault((key, row), []).append(
                    sorted(i for i in later["lists"][index].tolist() if i >= 0)
                )
    return variants


def compare_run(
    golden: dict[str, Any],
    reads: list[dict[str, Any]],
    unstable: dict[str, list[int]],
    variants: dict[tuple[str, int], list[list[int]]] | None = None,
    refill: bool = False,
    audit: Audit | None = None,
) -> Comparison:
    """Compare a replay with a golden, strictly except at the manifest's unstable rows.

    A row passes when its sorted ids equal the golden's. A row the manifest lists as unstable may also take another
    answer the freeze gave it in a later pass, or where none is recorded differ by at most one bucket of ids a side.

    Args:
        golden: The golden file's contents.
        reads: The replay's reads.
        unstable: The manifest's unstable rows of the run, by read.
        variants: The freeze's other answers on those rows, from `recorded_variants`.
        audit: What the replay's picks were held to - a row off the manifest's list that differs from the golden then
            passes where its read held no pick that parted from the exact top-k, since the freeze's radix top-k
            is where the golden parted from it.
        refill: Whether a row the golden read short may come out full - a tree's read just past a growth, which the
            freeze under-read. Such a row must hold every id the golden's did.

    Returns:
        The comparison.
    """
    result = Comparison()
    for want, got in zip(golden["reads"], reads, strict=True):
        key = read_key(want)
        result.reads += 1
        result.rows += len(want["rows"])
        if want["lists"].shape != got["lists"].shape or not torch.equal(want["positions"], got["positions"]):
            result.broken_reads += 1
            result.notes.append(f"{key}: shape or positions differ")
            continue
        a, b = want["lists"].sort(dim=1).values, got["lists"].sort(dim=1).values
        for index in (a != b).any(1).nonzero().flatten().tolist():
            chunk_row = int(want["rows"][index])
            sa, sb = set(a[index][a[index] >= 0].tolist()), set(b[index][b[index] >= 0].tolist())
            worst = max(len(sa - sb), len(sb - sa))
            width = want["lists"].shape[1]
            if refill and len(sa) < width and sa <= sb and len(sb) == width == int((b[index] >= 0).sum()):
                result.refilled_rows += 1
                result.notes.append(f"{key}: row {chunk_row} was short at {len(sa)} ids, now full")
                continue
            recorded = (variants or {}).get((key, chunk_row))
            on_the_cut = worst <= BUCKET_CAP if recorded is None else sorted(sb) in recorded
            if chunk_row in unstable.get(key, []) and on_the_cut:
                result.unstable_rows += 1
                answer = "" if recorded is None else ", an answer the freeze gave in another pass"
                result.notes.append(f"{key}: row {chunk_row} on the radix cut, {worst} ids a side{answer}")
            elif audit is not None and key not in audit.inexact:
                result.exact_rows += 1
                result.notes.append(
                    f"{key}: row {chunk_row} is the exact pick, {worst} ids a side from the golden"
                )
            else:
                result.mismatched_rows += 1
                result.notes.append(f"{key}: row {chunk_row} differs, {worst} ids a side")
    if audit is not None:
        result.inexact_picks = sum(audit.inexact.values())
        for key, rows in audit.inexact.items():
            result.notes.append(f"{key}: {rows} pick rows part from the exact top-k")
    return result


def digest_notes(info: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    """List the ways the replay's partition or hierarchy differs from the golden's."""
    notes = []
    for stage in ("after_prefill", "final"):
        if info[stage] != stats["pass1"][stage]:
            notes.append(
                f"{stage}: partition or hierarchy differs - {info[stage]} against {stats['pass1'][stage]}"
            )
    return notes


def probe_input(input_name: str, device: str = "cuda") -> dict[str, Any]:
    """Rebuild a real probe's input from its key latents and head queries, through the key and query projection.

    The goldens' recipe, with the serving code's projection in place of the freeze's: every key latent projected to
    its index key, and each rank's eight heads summed and projected to an index query, then drawn into the chunks'
    rows and the decode steps as the goldens drew them.

    Args:
        input_name: One of `PROBE_SOURCES`.
        device: Where it runs.

    Returns:
        The input in the goldens' layout, its `meta` the golden input's.
    """
    document, layer = PROBE_SOURCES[input_name]
    golden_input = torch.load(GOLDENS / "inputs" / f"{input_name}.pt", weights_only=False)
    meta = golden_input["meta"]
    blob = torch.load(
        PROBES / document / f"layer{layer}.pt", map_location="cpu", mmap=True, weights_only=False
    )
    query_projection, key_projection = weights.read(LATENT_MAP, 512, 64, GLM_INDEXER_LAYERS)[layer]
    projection = KeyAndQueryProjection(query_projection, key_projection, device)
    total = len(golden_input["keys"])
    keys = torch.cat(
        [
            projection.index_keys(blob["latent"][first : min(first + PROBE_PIECE, total)].to(device))
            for first in range(0, total, PROBE_PIECE)
        ]
    )
    # [heads, positions, key_dim] -> each rank's heads as [positions, heads a rank, key_dim], rank by rank
    heads = blob["queries"].to(device).to(torch.bfloat16)
    per_rank = heads.view(PROBE_RANKS, -1, heads.shape[1], heads.shape[2]).transpose(1, 2)
    pool = torch.cat([projection.index_queries(rank) for rank in per_rank])
    generator = torch.Generator(device).manual_seed(PROBE_SEED)
    chunks = {}
    for chunk in meta["recorded"]:
        rows = min((chunk + 1) * CHUNK, meta["prefill"]) - chunk * CHUNK
        drawn = pool[(torch.arange(rows, device=device) + chunk * PROBE_CHUNK_STRIDE) % len(pool)]
        noisy = (torch.arange(rows, device=device) % (2 * STRIDE) != 0)[:, None]
        noise = PROBE_NOISE * torch.randn(drawn.shape, device=device, generator=generator) * noisy
        chunks[chunk] = F.normalize(drawn + noise, dim=-1).cpu()
    decode = pool[(torch.arange(DECODE_STEPS, device=device) * PROBE_DECODE_STRIDE) % len(pool)].cpu()
    return {"keys": keys.cpu(), "chunks": chunks, "decode": decode, "meta": meta}


def run_one(
    name: str,
    input_name: str,
    device: str = "cuda",
    audit: bool = True,
    golden_input: dict[str, Any] | None = None,
) -> tuple[Comparison, float]:
    """Replay a design over an input and compare it with its golden.

    Args:
        name: The design.
        input_name: The input.
        device: Where it runs.
        audit: Whether to hold every top-k of the replay to a sort.
        golden_input: The input to replay in place of the input file's - a probe rebuilt by `probe_input`.

    Returns:
        The comparison, with the cache's digest differences among its notes, and the seconds the replay took.
    """
    manifest = json.loads((GOLDENS / "manifest.json").read_text())
    stem = f"{name}__{input_name}"
    golden = torch.load(GOLDENS / "goldens" / f"{stem}.pt", weights_only=False)
    stats = json.loads((GOLDENS / "goldens" / f"{stem}.stats.json").read_text())
    unstable = next(
        run["unstable_rows"]
        for run in manifest["runs"]
        if (run["design"], run["input"]) == (name, input_name)
    )
    if golden_input is None:
        golden_input = torch.load(GOLDENS / "inputs" / f"{input_name}.pt", weights_only=False)
    started = time.time()
    checked = Audit() if audit else None
    with checked.active() if checked else nullcontext():
        reads, info = replay(name, golden_input, device, checked)
    seconds = time.time() - started
    variants = recorded_variants(GOLDENS / "goldens" / f"{stem}.unstable.pt")
    result = compare_run(golden, reads, unstable, variants, refill=name in TREES, audit=checked)
    result.sorted_rows = 0 if checked is None else checked.rows
    for note in digest_notes(info, stats):
        result.mismatched_rows += 1
        result.notes.append(note)
    return result, seconds


def main() -> None:
    """Replay every requested design over every requested input and print the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", nargs="*", default=list(DESIGNS))
    parser.add_argument("--inputs", nargs="*", default=list(INPUTS))
    parser.add_argument("--no-audit", action="store_true", help="leave the picks unchecked")
    parser.add_argument("--notes", type=Path, help="write every run's comparison here, one json a run")
    args = parser.parse_args()
    manifest = json.loads((GOLDENS / "manifest.json").read_text())
    for name in args.designs:
        for line in manifest_disagreements(name, manifest):
            print(f"{name}: settings differ from the manifest - {line}", flush=True)
    failed = 0
    print(
        f"{'design':26} {'input':24} {'status':14} {'rows':>7} {'unstable':>8} {'refilled':>8} {'exact':>6} "
        f"{'mismatch':>8} {'inexact':>7} {'sorted':>9} {'seconds':>8}"
    )
    if args.notes:
        args.notes.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name in args.designs:
            for input_name in args.inputs:
                result, seconds = run_one(name, input_name, audit=not args.no_audit)
                failed += result.status == "mismatch"
                print(
                    f"{name:26} {input_name:24} {result.status:14} {result.rows:7d} {result.unstable_rows:8d} "
                    f"{result.refilled_rows:8d} {result.exact_rows:6d} {result.mismatched_rows + result.broken_reads:8d} "
                    f"{result.inexact_picks:7d} {result.sorted_rows:9d} {seconds:8.1f}",
                    flush=True,
                )
                for note in result.notes[:12]:
                    print(f"    {note}", flush=True)
                if args.notes:
                    summary = {**vars(result), "status": result.status, "seconds": seconds}
                    (args.notes / f"{name}__{input_name}.json").write_text(json.dumps(summary, indent=1))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
