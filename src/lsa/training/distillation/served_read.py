"""Score an exported map's served read on the held-out documents of its teacher cache, a shard of them a GPU.

    lsa distill served-read configs/training/distillation/glm53_flash/long_16k_1m_heavy_x_dense_on_native_served.yaml \
        --shards 8
    lsa distill served-read <config> --documents documents.json run.output=<directory>
    lsa distill served-read <config> --tables

The config is a distillation run's. `model.space.init` names the exported map, `data.evaluate` the held-out
documents - every one of each set, up to its `limit` - and `run.report.served` the designs and the rows read a
document, spread over every row it samples, since a held-out document has no training row.
`--documents` narrows them to a list, by name or as the manifest entries a run writes.

A design may read a map of its own, `serving.projections`, and the key and the heads' queries in place of `x`,
`indexer.input: key_and_query` - the frozen design's input, and the maps distilled on it, re-exported or trained
here. Those are not in the cache, so they are made from the cached `x` with the layer's own weights, as the native picks are:
the checkpoint loaded with only the sparse layers' attention, and GLM's key latent `kv_a_layernorm(kv_a(x))` at
every position and each head's query folded through its key map at the sampled rows - the original implementation's extraction of
its probes. The serving projection reads them as a server does, each of the design's `ranks` tensor-parallel
ranks summing its own heads and reading its own list - one list of every head under `query_heads: all` - and every
rank's read is scored against the teacher, whose mass is summed over all the heads, and the ranks averaged. Beside it `<design>_union` scores what the ranks read
between them at a position - its `ids_read` the distinct keys, its mass what any rank holds. Such a design's exact
read is `exact_<design>`, at its budget in its own space, with its union too.

Each (document, layer) is one record, `records/<document>_layer<index>.json` in the run directory, holding the
mean over the rows of every read of the same rows: each design's served read, the exact top-k at the run's
`top_k` (`exact`) and at each design's other budget (`exact_k<top_k>`), and GLM's own picks from the
`_native.pt` sidecars (`native`). A record already written is skipped, so a shard that died is rerun as is.
With `run.report.served.per_row` a record also keeps each row's position and metrics, and the tables split the
document's tail rows from the rest.

The shard that finds every record written writes the tables beside them - `report.md` and `summary.json`:
every metric by read and length, mass and recall by layer, and each design's mass less native's by layer and
length, with the cells more than 0.02 under it flagged. `--tables` writes them again from the records, locally.
A cell is the mean over the documents of a length, and a mean across lengths or layers weighs each alike.
"""

import argparse
import itertools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa import env, io, slurm
from lsa.constants import Attention
from lsa.sglang.index import weights
from lsa.sglang.index.layer_index import KeyAndQuery, KeyAndQueryProjection, Projection
from lsa.sglang.index.selector import read_sampled
from lsa.sglang.settings.design import (
    ATTENTION_INPUT,
    KEY_AND_QUERY,
    PAGE_SIZE,
    QUERY_HEADS_ALL,
    DesignSettings,
)
from lsa.training import require_extra
from lsa.training.distillation.checkpoint import resolve_layers
from lsa.training.distillation.constants import Cache, Native, Results, Served
from lsa.training.distillation.data.cache import Document, layer_path, listing, set_root
from lsa.training.distillation.data.native import key_and_query, load_attention
from lsa.training.distillation.data.pool import stream
from lsa.training.distillation.methods import MethodSet, expand_methods, read_methods
from lsa.training.distillation.metrics import agreement, exact_read, probe_rows, served_row_metrics
from lsa.training.distillation.probe_report import length_label, mean, probe_tables
from lsa.training.distillation.reporting import served_designs, spread

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig, MethodConfig

# What a job asks for when the command line says nothing
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": Served.GPUS,
    "cpus_per_task": Served.CPUS,
    "mem_gb": Served.MEM_GB,
    "timeout_min": Served.HOURS * 60,
}

# Metric -> what its tables are headed, and how a value is printed
METRICS: dict[str, tuple[str, str]] = {
    "mass": ("share of the teacher's attention captured, per head", "{:.3f}"),
    "mass_top": ("share of the teacher's cached long-range mass captured", "{:.3f}"),
    "recall": ("share of the teacher's cached keys read", "{:.3f}"),
    "recall64": ("share of the teacher's top 64 keys read", "{:.3f}"),
    "recall16": ("share of the teacher's top 16 keys read", "{:.3f}"),
    "teacher_ids_read": ("teacher keys read", "{:.0f}"),
    "ids_read": ("key ids read, the window and the sinks included", "{:.0f}"),
    "far_share": ("share of the long-range keys read at least 8192 back", "{:.3f}"),
    "far_mass": ("share of the teacher's cached mass at least 8192 back captured", "{:.3f}"),
    "native_overlap": ("share of GLM's own long-range picks read", "{:.3f}"),
    "native_far_overlap": ("share of GLM's own picks at least 8192 back read", "{:.3f}"),
    "target_held": ("share of a probe's long-range target keys read", "{:.3f}"),
    "target_any": ("share of probes with a long-range target that read at least one", "{:.3f}"),
    "line_held": ("share of the needle's sentence read", "{:.3f}"),
    "agreement": ("share of the served design's long-range keys a method also reads", "{:.3f}"),
    "scored_vectors": ("vectors a method scored a query against", "{:.0f}"),
    "scored_dots": ("dot products a method took a query - a vector a query head", "{:.0f}"),
}


def held_out(experiment: "DistillConfig") -> list[dict[str, Any]]:
    """Return the documents the evaluation sets hold, each once, with where its files are.

    Args:
        experiment: The experiment.

    Returns:
        Manifest entries, each with the `root` and the `collection` its layer files sit under.
    """
    documents: dict[str, dict[str, Any]] = {}
    for document_set in experiment.data.evaluate:
        root = str(set_root(experiment.data.root, document_set).resolve())
        for entry in itertools.islice(listing(experiment.data.root, document_set), document_set.limit):
            documents.setdefault(entry["name"], {**entry, "root": root, "collection": document_set.source})
    return list(documents.values())


def document_path(entry: dict[str, Any], layer: int) -> Path:
    """Return the layer file of a document `held_out` returned."""
    return layer_path(entry["root"], entry["collection"], entry["split"], entry["name"], layer)


def record_path(directory: Path, document: str, layer: int) -> Path:
    """Return where the record of one (document, layer) is written."""
    return directory / Results.RECORDS / f"{document}_layer{layer}.json"


def deal(documents: Sequence[dict[str, Any]], shards: int) -> list[list[dict[str, Any]]]:
    """Deal documents into shards of about equal total length, the longest first to the lightest shard.

    Args:
        documents: Entries with a `name` and a `length`.
        shards: Shards to deal into.

    Returns:
        Each shard's documents.
    """
    dealt: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for entry in sorted(documents, key=lambda entry: (-entry["length"], entry["name"])):
        lightest = loads.index(min(loads))
        dealt[lightest].append(entry)
        loads[lightest] += entry["length"]
    return dealt


@dataclass(frozen=True)
class Space:
    """What a design's index keys and index queries are read with."""

    # The projection file
    projections: str
    # The indexer input it projects - `x`, or the key and the heads' queries
    input: str = ATTENTION_INPUT
    # Tensor-parallel ranks the heads split over under the key and query, each summing and selecting its own
    ranks: int = 1


def design_spaces(experiment: "DistillConfig", designs: dict[str, DesignSettings]) -> dict[str, Space]:
    """Return the space each design reads in - the run's exported map on `x` unless it names its own.

    Args:
        experiment: The experiment, whose `model.space.init` is the exported map.
        designs: The served designs.

    Returns:
        Each design's space, by name.
    """
    ranks = {design.name: design.ranks for design in experiment.run.report.served.designs}
    spaces = {}
    for name, settings in designs.items():
        per_rank = settings.indexer.input == KEY_AND_QUERY and settings.indexer.query_heads != QUERY_HEADS_ALL
        spaces[name] = Space(
            settings.serving.projections or str(experiment.model.space.init),
            settings.indexer.input,
            ranks[name] if per_rank else 1,
        )
    return spaces


def exact_budgets(
    experiment: "DistillConfig", designs: dict[str, DesignSettings], spaces: dict[str, Space]
) -> dict[str, tuple[Space, int]]:
    """Return the exact reads scored beside the served ones - at the run's budget and at each design's.

    Args:
        experiment: The experiment.
        designs: The served designs.
        spaces: Each design's space.

    Returns:
        The space each exact read reads in and the long-range keys it takes, by name.
    """
    run = Space(str(experiment.model.space.init), experiment.model.indexer.input)
    budgets = {Served.EXACT: (run, experiment.model.indexer.top_k)}
    for name, settings in designs.items():
        read = (spaces[name], settings.indexer.top_k)
        if read in budgets.values():
            continue
        if spaces[name] != run:
            budgets[Served.EXACT_OF.format(design=name)] = read
        else:
            budgets[Served.EXACT_AT.format(top_k=settings.indexer.top_k)] = read
    return budgets


def load_map(
    path: Path, layers: list[int], device: str, indexer_input: str = ATTENTION_INPUT
) -> dict[int, Projection | KeyAndQueryProjection]:
    """Read an exported map as a server reads it, one projection a layer.

    Args:
        path: The projection file `lsa distill export` wrote.
        layers: The layers to score.
        device: Where the maps land.
        indexer_input: The input the map projects.

    Returns:
        Each layer's projection.

    Raises:
        ValueError: The file holds no map for a layer to score.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    input_dim, index_dim = next(iter(blob.values())).shape
    held = weights.read(path, input_dim, index_dim, {int(str(name).partition(".")[0]) for name in blob})
    missing = sorted(set(layers) - set(held))
    if missing:
        raise ValueError(f"{path} holds no map for layers {missing} - it holds {sorted(held)}")
    if indexer_input == KEY_AND_QUERY:
        return {layer: KeyAndQueryProjection(*held[layer], device) for layer in layers}
    return {layer: Projection(*held[layer], device) for layer in layers}


def project(projection: Projection, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Read every position's index query and index key off `x`, a prefill chunk at a time as a server does.

    Args:
        projection: The layer's maps.
        x: The attention's input [T, model_dim] bf16, on the maps' device.

    Returns:
        The index queries [T, index_dim] fp32 unit rows,
        and the index keys [T, index_dim] bf16 unit rows.
    """
    chunks = [projection(x[start : start + PAGE_SIZE]) for start in range(0, x.shape[0], PAGE_SIZE)]
    return torch.cat([queries for queries, _ in chunks]), torch.cat([keys for _, keys in chunks])


def project_key_and_query(
    projection: KeyAndQueryProjection, inputs: KeyAndQuery, ranks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the key and query input as the server does - each rank's heads summed, every key latent mapped.

    Args:
        projection: The layer's maps.
        inputs: The key latents [T, key_dim] and the sampled rows' heads' queries [R, heads, key_dim], bf16.
        ranks: Tensor-parallel ranks, each holding a contiguous run of the heads.

    Returns:
        The index queries [R, ranks, index_dim] fp32 unit rows, one a rank, and the index keys [T, index_dim]
        bf16 unit rows.
    """
    rows, heads, key_dim = inputs.query.shape
    per_rank = inputs.query.view(rows, ranks, heads // ranks, key_dim)
    index_queries = torch.stack([projection.index_queries(per_rank[:, rank]) for rank in range(ranks)], dim=1)
    pieces = range(0, len(inputs.key), Served.KEY_PIECE)
    index_keys = torch.cat(
        [projection.index_keys(inputs.key[start : start + Served.KEY_PIECE]) for start in pieces]
    )
    return index_queries, index_keys


def union(read: torch.Tensor) -> torch.Tensor:
    """Merge each row's lists into one of the distinct keys they hold.

    Args:
        read: Key ids [R, ranks, W] int, -1 padded.

    Returns:
        Key ids [R, ranks * W] int, each key once, -1 padded.
    """
    merged = read.flatten(1).sort(dim=1).values
    repeated = torch.zeros_like(merged, dtype=torch.bool)
    repeated[:, 1:] = merged[:, 1:] == merged[:, :-1]
    return merged.masked_fill(repeated, -1)


def ranked_row_metrics(
    document: Document, rows: torch.Tensor, read: torch.Tensor, name: str
) -> dict[str, dict[str, torch.Tensor]]:
    """Score a read one number a row - and when it is one list a rank, each row's ranks averaged, and their union.

    Under the key and query each rank's heads attend over its own list, while the teacher's mass is summed over
    every head, so a rank's list is scored against the whole teacher and the ranks averaged. The union is what
    the ranks read between them at a position - its `ids_read` the distinct keys, its mass what any rank holds.

    Args:
        document: The cached document, on the GPU.
        rows: Query rows [R] int64.
        read: Key ids [R, W], or [R, ranks, W] one list a rank, int, -1 padded.
        name: The read's name.

    Returns:
        Each metric [R] fp32 by read - `name`, and `<name>_union` for more than one rank.
    """
    if read.ndim == 2 or read.shape[1] == 1:
        return {name: served_row_metrics(document, rows, read.flatten(1) if read.ndim == 3 else read)}
    ranks = read.shape[1]
    per_rank = served_row_metrics(document, rows.repeat_interleave(ranks), read.flatten(0, 1))
    return {
        name: {metric: values.view(len(rows), ranks).nanmean(1) for metric, values in per_rank.items()},
        Served.UNION.format(read=name): served_row_metrics(document, rows, union(read)),
    }


def key_and_query_of(document: Document, rows: torch.Tensor, attention: Any, projections: str) -> KeyAndQuery:
    """Return the key and every head's query at some rows of a document, as a served read projects them.

    Args:
        document: The cached document, on the device the attention is on.
        rows: Its query rows [R] int64.
        attention: The layer's `self_attn`, which reads them off `x`, or None.
        projections: The maps that read them, for the message.

    Returns:
        The key at every position and each head's query at the rows - kept by a cache extracted under the key and
        query, or read off `x` by the layer's own modules.

    Raises:
        ValueError: The document holds `x` and no attention was loaded to read them off it.
    """
    if document.x is None and document.key is not None and document.head_queries is not None:
        return KeyAndQuery(document.key, document.head_queries[rows])
    # A model with key-value heads keeps each group's queries summed already
    # => one head a row, which the maps' sum over heads leaves as it is
    if document.x is None and document.key is not None and document.summed_query is not None:
        return KeyAndQuery(document.key, document.summed_query[rows][:, None])
    if attention is None:
        raise ValueError(f"{projections} reads the key and query, and no attention was loaded")
    return key_and_query(attention, document.key_inputs(), document.query_positions[rows].long())


def agreement_with(
    method: "MethodConfig",
    read: torch.Tensor,
    served: dict[str, torch.Tensor],
    document: Document,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Measure how much of the long-range read of the served design a method names it holds.

    Args:
        method: The method, which names the design.
        read: The method's read [R, W] int32, on the document's device.
        served: The served reads made so far, by design.
        document: The cached document.
        positions: The rows' positions [R] int64.

    Returns:
        The share of the design's long-range keys the method holds [R] fp32.

    Raises:
        ValueError: The design was not served as one list a row - on a host with no GPU, or a read a rank.
    """
    other = served.get(method.agrees_with or "")
    if other is None or other.ndim != 2:
        raise ValueError(
            f"method {method.name!r} agrees with {method.agrees_with!r}, which has no served read of one list a row"
        )
    return agreement(read, other, positions, document.sink_size, document.window_size)


def score_layer(
    document: Document,
    projections: dict[Space, Projection | KeyAndQueryProjection],
    designs: dict[str, DesignSettings],
    spaces: dict[str, Space],
    budgets: dict[str, tuple[Space, int]],
    most_rows: int,
    attention: Any = None,
    per_row: bool = False,
    method_set: MethodSet | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Score every read of one (document, layer) on the same rows, spread over every row it samples.

    Args:
        document: The cached document with its native selection, on the GPU.
        projections: The layer's maps by space, on the same device.
        designs: The served designs.
        spaces: Each design's space.
        budgets: The exact reads, each with its space.
        most_rows: Query rows to score at most.
        attention: The layer's `self_attn`, which a space on the key and query reads its input with.
        per_row: Also keep each row's position and its `Served.PER_ROW` metrics, under `per_row`.

    Returns:
        The record - the mean of every metric over the rows by read, and the seconds each served read took.

    Raises:
        ValueError: A space reads the key and query with no attention to make it with, or reads `x` off a
            document that holds the key and query.
    """
    x = document.key_inputs()
    rows = spread(torch.arange(len(document.query_positions), device=document.device), most_rows)
    positions = document.query_positions[rows]
    inputs = None
    indexed: dict[Space, tuple[torch.Tensor, torch.Tensor]] = {}
    for space, projection in projections.items():
        if isinstance(projection, KeyAndQueryProjection):
            inputs = inputs or key_and_query_of(document, rows, attention, space.projections)
            indexed[space] = project_key_and_query(projection, inputs, space.ranks)
        elif document.x is None:
            raise ValueError(
                f"{space.projections} reads x, and {document.name} holds the key and query in its place"
            )
        else:
            every_query, index_keys = project(projection, x)
            indexed[space] = (every_query[positions], index_keys)
            del every_query
    means: dict[str, dict[str, float]] = {}
    seconds: dict[str, float] = {}
    kept: dict[str, dict[str, list[float]]] = {}
    probed: dict[str, dict[str, list[float]]] = {}
    served: dict[str, torch.Tensor] = {}

    def scored(read: torch.Tensor, name: str, extra: dict[str, torch.Tensor] | None = None) -> None:
        """Record the mean, per-row and probe metrics of a read, and of its union for more than one rank.

        Args:
            read: The read lists.
            name: The read's name.
            extra: Metrics to add to the read named `name`.
        """
        for read_name, values in ranked_row_metrics(document, rows, read, name).items():
            if read_name == name:
                values = {**values, **(extra or {})}
            means[read_name] = {metric: float(row.nanmean()) for metric, row in values.items()}
            if per_row:
                kept[read_name] = {metric: values[metric].tolist() for metric in Served.PER_ROW}
            if document.target_key_ids is not None:
                probed[read_name] = {
                    metric: values[metric].tolist() for metric in Served.PROBE_ROW_METRICS if metric in values
                }

    for name, settings in designs.items():
        index_queries, index_keys = indexed[spaces[name]]
        began = time.monotonic()
        read = read_sampled(settings, index_queries, index_keys, positions.to(torch.int32))
        seconds[name] = time.monotonic() - began
        # One list a row, the form a method's agreement reads
        served[name] = read[:, 0] if read.ndim == 3 and read.shape[1] == 1 else read
        scored(read, name)
    for name, (space, top_k) in budgets.items():
        index_queries, index_keys = indexed[space]
        ranks = index_queries.shape[1] if index_queries.ndim == 3 else 1
        read = exact_read(
            index_queries.flatten(0, -2),
            index_keys,
            positions.repeat_interleave(ranks),
            top_k,
            document.sink_size,
            document.window_size,
        )
        scored(read.view(len(rows), ranks, -1).squeeze(1), name)
    if document.native_key_ids is not None:
        scored(document.native_key_ids[rows], Served.NATIVE)
    if method_set is not None and method_set.methods:
        began = time.monotonic()
        method_reads = read_methods(
            method_set,
            {name: indexed[spaces[name]] for name in method_set.designs},
            document,
            rows,
            lambda: inputs or key_and_query_of(document, rows, attention, "a PISA method"),
            None if attention is None else float(attention.scaling),
        )
        method_seconds = (time.monotonic() - began) / len(method_reads)
        for variant in expand_methods(method_set.methods):
            seconds[variant.name] = method_seconds
            method_read = method_reads[variant.name]
            read = method_read.selected_key_ids.to(document.device)
            costs = {
                "scored_vectors": method_read.scored_vectors.float().to(document.device),
                "scored_dots": method_read.scored_dots.float().to(document.device),
            }
            if variant.method.agrees_with is not None:
                costs["agreement"] = agreement_with(variant.method, read, served, document, positions)
            scored(read, variant.name, costs)
    record: dict[str, Any] = {
        "document": document.name,
        "task": task,
        "layer": document.layer,
        "length": document.key_count,
        "key_count": document.key_count,
        "rows": len(rows),
        "means": means,
        "seconds": seconds,
    }
    if document.target_key_ids is not None and document.query_kind is not None:
        record["probe"] = {
            "query_kinds": list(document.meta.get(Cache.QUERY_KIND_NAMES, Cache.QUERY_KINDS)),
            "rows": {
                name: values.tolist()
                for name, values in probe_rows(
                    positions,
                    document.query_kind[rows],
                    document.target_key_ids[rows],
                    document.sink_size,
                    document.window_size,
                ).items()
            },
            "reads": probed,
        }
    if per_row:
        record["per_row"] = {
            "positions": positions.tolist(),
            # A row at or past this one is a tail row - the last positions of the document
            "tail_from": document.train_query_count,
            "row_ids": rows.tolist(),
            "reads": kept,
        }
    return record


def score(
    experiment: "DistillConfig",
    shard: int,
    shards: int,
    documents: list[dict[str, Any]],
    force: bool = False,
    device: str = "cuda:0",
) -> Path:
    """Score one shard's documents at every layer of the run, a record each, and the tables once all are written.

    Args:
        experiment: The experiment, whose `model.space.init` is the exported map.
        shard: This shard.
        shards: Shards the documents are dealt into.
        documents: Every document of the run, as `held_out` returns them.
        force: Rescore a record already written.
        device: Where the documents and the maps land.

    Returns:
        The directory the records and the tables are written into.
    """
    require_extra("omegaconf")
    out = experiment.out
    layers = resolve_layers(experiment)
    designs = served_designs(experiment, device)
    all_designs = experiment.run.report.served.settings(experiment.model.indexer)
    spaces = design_spaces(experiment, all_designs)
    method_set = MethodSet(
        experiment.run.report.served.methods,
        all_designs,
        {name: space.projections for name, space in spaces.items()},
    )
    budgets = exact_budgets(experiment, designs, spaces)
    maps = {
        space: load_map(Path(space.projections), layers, device, space.input)
        for space in dict.fromkeys([*spaces.values(), *(space for space, _ in budgets.values())])
    }
    latent, ships_selector = model_kind(experiment.model.name)
    # Only GLM's latent is read off `x` by the layer's modules - a model with key-value heads keeps its key and query
    # => its attention is never loaded
    attentions = (
        load_attention(experiment.model.name, layers, device)
        if latent and (any(space.input == KEY_AND_QUERY for space in maps) or needs_attention(method_set))
        else {}
    )
    todo = [
        (entry, layer)
        for entry in deal(documents, shards)[shard]
        for layer in layers
        if force or not record_path(out, entry["name"], layer).exists()
    ]
    print(
        f"{experiment.run.name}: shard {shard} of {shards} scores {len(todo)} (document, layer)", flush=True
    )
    paths = [document_path(entry, layer) for entry, layer in todo]
    with torch.inference_mode():
        for (entry, layer), waiting in zip(
            todo, stream(paths, experiment.model.indexer, native=ships_selector), strict=True
        ):
            began = time.monotonic()
            record = score_layer(
                waiting.to(device),
                {space: by_layer[layer] for space, by_layer in maps.items()},
                designs,
                spaces,
                budgets,
                experiment.run.report.served.rows,
                attentions.get(layer),
                experiment.run.report.served.per_row,
                method_set,
                entry.get("source"),
            )
            # A prompt holds fewer keys than the length it was cut at
            # => report by that length, one column a cell
            record["length"] = entry["length"]
            io.write_json(record_path(out, entry["name"], layer), record)
            mass = " ".join(f"{name} {means['mass']:.3f}" for name, means in record["means"].items())
            print(
                f"{entry['name']} layer {layer} ({record['rows']} rows, {time.monotonic() - began:.0f} s): {mass}",
                flush=True,
            )
    if all(record_path(out, entry["name"], layer).exists() for entry in documents for layer in layers):
        write_tables(out)
    return out


def model_kind(model: str) -> tuple[bool, bool]:
    """Return whether a model indexes GLM's key latent, and whether it ships a sparse selector of its own.

    Args:
        model: The model card, or a checkpoint path, which is taken to be GLM's.

    Returns:
        Whether it is an MLA model, and whether its cache keeps the native selection.
    """
    from lsa.evaluation.config import ModelConfig

    asset = ModelConfig(name=model).asset
    if asset is None:
        return True, True
    return asset.family == Native.LATENT_FAMILY, asset.attention == Attention.BLOCK_SPARSE


def needs_attention(method_set: MethodSet) -> bool:
    """Return whether a method reads GLM's attention at the layer's own softmax scale, which the layer's modules give."""
    return any(
        method.kind == Served.METHOD_PISA
        and method.space == Served.SPACE_ATTENTION
        and method.logit_scale is None
        for method in method_set.methods
    )


def cells(records: Sequence[dict[str, Any]], metric: str) -> dict[tuple[str, int, int], float]:
    """Average one metric over the documents of each (read, layer, length).

    Args:
        records: The records.
        metric: The metric.

    Returns:
        The mean by read, layer and length.
    """
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for record in records:
        for read, means in record["means"].items():
            # A record written before a metric existed has none
            grouped.setdefault((read, record["layer"], record["length"]), []).append(
                means.get(metric, math.nan)
            )
    return {key: mean(values) for key, values in grouped.items()}


def by_length(
    cell: dict[tuple[str, int, int], float],
    reads: Sequence[str],
    layers: Sequence[int],
    lengths: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Average cells over layers - one row a read, one column a length, and `all` weighing each length alike.

    Args:
        cell: Means by read, layer and length.
        reads: The rows.
        layers: The layers averaged over.
        lengths: The columns.

    Returns:
        The value by read and column.
    """
    rows = {}
    for read in reads:
        row = {
            length_label(length): mean(cell.get((read, layer, length), math.nan) for layer in layers)
            for length in lengths
        }
        row["all"] = mean(row.values())
        rows[read] = row
    return rows


def by_layer(
    cell: dict[tuple[str, int, int], float],
    reads: Sequence[str],
    layers: Sequence[int],
    lengths: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Average cells over lengths - one row a layer, one column a read.

    Args:
        cell: Means by read, layer and length.
        reads: The columns.
        layers: The rows.
        lengths: The lengths averaged over.

    Returns:
        The value by layer and read.
    """
    return {
        str(layer): {
            read: mean(cell.get((read, layer, length), math.nan) for length in lengths) for read in reads
        }
        for layer in layers
    }


def by_region(
    records: Sequence[dict[str, Any]], reads: Sequence[str], metric: str
) -> dict[str, dict[str, float]]:
    """Average one metric over the tail rows and over the rest, a record's mean first, then over records.

    Args:
        records: Records that kept their rows.
        reads: The rows of the table.
        metric: A metric the records kept a row at a time.

    Returns:
        The value by read and region - `tail`, `rest` and `all`.
    """
    out = {}
    for read in reads:
        regions: dict[str, list[float]] = {"tail": [], "rest": [], "all": []}
        for record in records:
            kept = record["per_row"]
            values = kept["reads"].get(read, {}).get(metric)
            if values is None:
                continue
            pairs = list(zip(kept["row_ids"], values, strict=True))
            tail = [value for row, value in pairs if row >= kept["tail_from"]]
            rest = [value for row, value in pairs if row < kept["tail_from"]]
            for region, chosen in (("tail", tail), ("rest", rest), ("all", values)):
                regions[region].append(mean(chosen))
        out[read] = {region: mean(values) for region, values in regions.items()}
    return out


def markdown(rows: dict[str, dict[str, float]], corner: str, form: str) -> list[str]:
    """Render rows of numbers as a markdown table, a missing value as `-`.

    Args:
        rows: The value by row and column, every row with the same columns.
        corner: The first column's heading.
        form: How a value is printed.

    Returns:
        The table's lines.
    """
    columns = list(next(iter(rows.values())))
    lines = [f"| {corner} | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for name, row in rows.items():
        values = ["-" if math.isnan(row[column]) else form.format(row[column]) for column in columns]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return lines


def tables(directory: Path) -> tuple[list[str], dict[str, Any]]:
    """Build the tables of every record in a run directory.

    Args:
        directory: The run directory, its records under `records/`.

    Returns:
        The report's lines and the numbers behind them.
    """
    records = [io.read_json(path) for path in sorted((directory / Results.RECORDS).glob("*.json"))]
    reads = list(dict.fromkeys(read for record in records for read in record["means"]))
    designs = list(dict.fromkeys(design for record in records for design in record["seconds"]))
    layers = sorted({record["layer"] for record in records})
    lengths = sorted({record["length"] for record in records})
    documents = {length: len({r["document"] for r in records if r["length"] == length}) for length in lengths}
    lines = [
        "# Served read",
        "",
        f"{len(records)} records, {sum(documents.values())} documents - "
        + ", ".join(f"{count} at {length_label(length)}" for length, count in documents.items())
        + f" - layers {', '.join(map(str, layers))}.",
    ]
    summary: dict[str, Any] = {
        "documents": {length_label(k): v for k, v in documents.items()},
        "layers": layers,
    }
    cell = {metric: cells(records, metric) for metric in METRICS}
    for metric, (heading, form) in METRICS.items():
        rows = by_length(cell[metric], reads, layers, lengths)
        # A metric no read has - a probe's, off a probe cache - is not a table
        if all(math.isnan(value) for row in rows.values() for value in row.values()):
            continue
        summary[f"{metric}_by_length"] = rows
        lines += ["", f"## {metric} by length - {heading}", "", *markdown(rows, "read", form)]
    for metric in ("mass", "recall"):
        rows = by_layer(cell[metric], reads, layers, lengths)
        summary[f"{metric}_by_layer"] = rows
        lines += ["", f"## {metric} by layer", "", *markdown(rows, "layer", METRICS[metric][1])]
    kept = [record for record in records if "per_row" in record]
    if kept:
        for metric in Served.PER_ROW:
            rows = by_region(kept, reads, metric)
            summary[f"{metric}_by_region"] = rows
            lines += [
                "",
                f"## {metric} by region - the document's tail rows against the rows drawn over the rest of it",
                "",
                *markdown(rows, "read", METRICS[metric][1]),
            ]
    flagged = []
    for design in designs:
        gaps = {
            str(layer): {
                length_label(length): cell["mass"].get((design, layer, length), math.nan)
                - cell["mass"].get((Served.NATIVE, layer, length), math.nan)
                for length in lengths
            }
            for layer in layers
        }
        summary[f"{design}_mass_minus_native"] = gaps
        lines += [
            "",
            f"## {design} mass minus native, by layer and length",
            "",
            *markdown(gaps, "layer", "{:+.3f}"),
        ]
        flagged += [
            f"{design} layer {layer} at {length}: {gap:+.3f}"
            for layer, row in gaps.items()
            for length, gap in row.items()
            if gap < -Served.BELOW_NATIVE
        ]
    summary["below_native"] = flagged
    lines += ["", f"## More than {Served.BELOW_NATIVE} of mass under native", ""]
    lines += [f"- {line}" for line in flagged] or ["None."]
    probe_lines, summary["probe"] = probe_tables(records)
    return lines + probe_lines, summary


def write_tables(directory: Path) -> list[str]:
    """Write the tables of a run directory's records beside them.

    Args:
        directory: The run directory.

    Returns:
        The report's lines.
    """
    lines, summary = tables(directory)
    (directory / Results.TABLES).write_text("\n".join(lines) + "\n")
    io.write_json(directory / Results.SUMMARY, summary)
    return lines


def main() -> None:
    """Parse the command line and submit each run's shards, or write its tables from the records."""
    parser = argparse.ArgumentParser(
        description="Score an exported map's served read on held-out documents, a shard of them a GPU"
    )
    parser.add_argument(
        "config", help="path to the training yaml, whose model.space.init is the exported map"
    )
    parser.add_argument("--shards", type=int, default=1, help="jobs the documents are dealt into, a GPU each")
    parser.add_argument(
        "--documents", type=Path, help="a json list of the documents to score, by name or as manifest entries"
    )
    parser.add_argument(
        "--tables", action="store_true", help="write the tables from the records, without a job"
    )
    parser.add_argument("--force", action="store_true", help="rescore records already written")
    slurm.add_flags(parser)
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")

    from lsa.training.distillation.config import load

    for experiment in load(args.config, args.overrides):
        experiment.run.output = str(Path(experiment.run.output).resolve())
        if args.tables:
            print("\n".join(write_tables(experiment.out)), flush=True)
            continue
        if not experiment.model.space.init:
            raise ValueError(f"{experiment.run.name}: model.space.init must name the exported map to score")
        experiment.model.space.init = str(Path(experiment.model.space.init).resolve())
        for design in experiment.run.report.served.designs:
            if design.serving.get("projections"):
                design.serving["projections"] = str(Path(design.serving["projections"]).resolve())
        documents = held_out(experiment)
        if args.documents:
            wanted = [
                item if isinstance(item, str) else item["name"] for item in io.read_json(args.documents)
            ]
            unknown = sorted(set(wanted) - {entry["name"] for entry in documents})
            if unknown:
                raise ValueError(f"{args.documents} names documents no evaluation set holds - {unknown}")
            documents = [entry for entry in documents if entry["name"] in wanted]
        print(
            f"{experiment.run.name}: {len(documents)} documents, designs "
            f"{[design.name for design in experiment.run.report.served.designs]}, map {experiment.model.space.init}, "
            f"{args.shards} shards into {experiment.out}",
            flush=True,
        )
        if args.dry_run:
            continue
        io.write_json(experiment.out / Results.DOCUMENTS, documents)
        executor = slurm.executor(
            Served.JOB_NAME, env.distill(), args.folder, **{**PARAMETERS, **slurm.overrides(args)}
        )
        shards = range(args.shards)
        jobs = executor.map_array(
            score,
            [experiment] * args.shards,
            list(shards),
            [args.shards] * args.shards,
            [documents] * args.shards,
            [args.force] * args.shards,
        )
        for job, shard in zip(jobs, shards, strict=True):
            print(f"submitted {job.job_id}  {experiment.run.name} shard {shard}", flush=True)
