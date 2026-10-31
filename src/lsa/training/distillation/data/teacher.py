"""Extract the targets - the teacher's own attention, one document and one layer at a time.

    lsa distill targets configs/data/targets/glm53_flash/long_16k_1m_heavy_key_and_query.yaml
    lsa distill targets configs/data/targets/glm53_flash/long_16k_1m.yaml --workers 2
    lsa distill targets configs/data/targets/glm53_flash/long_16k_1m.yaml --stage 1 'run.stages=[16,30]'

A checkpoint no card holds runs a [span of its stack](stage.py) per job with `stages`, each on one card.
Every span is submitted unless `--stage` names one, and `--workers` submits that many jobs per span - each
claims the documents it reads, so any number of them work one cache.

The model is run once per document with its own attention, the layer inputs are captured on the way past,
and each captured layer gives two things: `x`, what the layer's attention was handed - the residual streams
mixed and normed, the input the index keys and index queries are projected from - and the attention the
teacher actually paid. Nothing numeric is reimplemented - the projections, the QK norm, the rotation and the
scale are the model's own, reached through the same [adapter](../../../evaluation/adapter/__init__.py) the accuracy
harness drives, so a checkpoint that harness supports is a checkpoint this extracts from.

The target is defined over the keys a read may select - past the sinks and outside the window - because that
is the set a selection is judged on. The cluster block is the window, so every such key sits in a closed
cluster block and the block adds nothing to the rule.

`run.input: key_and_query` keeps the key and query the layer's own modules read off `x` in its place - the
[native](native.py) reading a run on that input makes of a cache of `x`, so the run reads the same tensors off a
file a fifth the size on GLM.

`data.mass` writes the teacher's dense attention mass in place of the layer files - see [teacher_mass](teacher_mass.py) for what it keeps
and the files it writes.

`queries.targets: ruler` takes the query positions off a RULER row in place of drawing them, and keeps beside each its
kind and the keys it is meant to find - see [teacher_plan](teacher_plan.py) and the field table in [cache](cache.py).
"""

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lsa import env, io, slurm
from lsa.sglang.settings.design import KEY_AND_QUERY
from lsa.training import require_extra
from lsa.training.distillation.constants import Cache, Stage, Teacher
from lsa.training.distillation.data.cache import layer_path, write
from lsa.training.distillation.data.native import KeyAndQueryInput, key_and_query_input
from lsa.training.distillation.data.stage import (
    Boundary,
    BoundaryWriter,
    boundary_path,
    boundary_ready,
    remove_boundary,
    stage_spans,
)
from lsa.training.distillation.data.teacher_attention import native_attention, teacher_attention
from lsa.training.distillation.data.teacher_capture import (
    capture_layer_inputs,
    capture_span_inputs,
    hook_native_selection,
    input_device,
)
from lsa.training.distillation.data.teacher_config import TeacherConfig, load
from lsa.training.distillation.data.teacher_mass import MassRequest, dump_layer, mass_path
from lsa.training.distillation.data.teacher_plan import (
    Planned,
    Probes,
    document_queries,
    extracted_now,
    flatten_plan,
    plan,
    resolve_selection,
    window_length,
    write_collection_manifests,
)


def locate_claim(root: Path, collection: str, split: str, name: str) -> io.Claim:
    """Return the claim that covers one document.

    Args:
        root: The cache root.
        collection: The dataset's directory.
        split: `train` or `eval`.
        name: The document's name.

    Returns:
        The claim, which may or may not be held.
    """
    return io.Claim(Path(root) / collection / split / f"{name}{Teacher.CLAIM_SUFFIX}", Teacher.CLAIM_STALE)


def _said(document: Planned) -> str:
    """Return how one planned document is named in a log line.

    Args:
        document: The document.

    Returns:
        `<collection>/<split>/<name>`.
    """
    return f"{document.documents.collection}/{document.selected.split}/{document.selected.name}"


def extract(
    run: TeacherConfig, force: bool = False, device: str = "cuda:0", stage: int | None = None
) -> Path:
    """Build one extraction's cache, taking whatever of it no other process is holding.

    A document already cached is skipped and a document another process claimed is left to it, so any
    number of these may work one cache and a rerun costs only what is missing. This returns when every
    document is written rather than when its own walk ends, so the last process standing finishes what a
    killed one was holding.

    With `stage` this process loads one span of the stack the extraction's `stages` cut it into, and writes
    the layers inside it. A span after the first takes a document only once the span before it has
    published the streams it reads, and deletes them once its own layers are written.

    Args:
        run: The extraction.
        force: Re-extract documents that are already written.
        device: Where the weights land, as a transformers device map - a device string for one card, or
            `auto` to spread a checkpoint no single card holds across every visible one.
        stage: Which span of `run.run.stages` to run, or None for the whole stack.

    Returns:
        The cache root.
    """
    require_extra("omegaconf", "transformers", "pyarrow")
    from lsa.evaluation import adapter

    root = Path(run.run.output)
    resolved = resolve_selection(run) if run.data.selection is not None else None
    model_adapter = adapter.adapter_for(run.model)
    every_layer = run.run.layers or model_adapter.full_attention_layers(
        model_adapter.text_config(run.model), run.model
    )
    span = resolve_span(run, every_layer, stage)
    layers = every_layer if span is None else [index for index in every_layer if span[0] <= index < span[1]]

    model, tokenizer = model_adapter.load(run.model, device, attention="sdpa", span=span)
    if run.run.prefill:
        model_adapter.prepare_long_prefill(model)
    text_model = model_adapter.text_model(model)
    inputs = input_device(model) if span is None else next(text_model.layers[span[0]].parameters()).device
    rotary = model_adapter.rotary_layout(model, run.model, layers)

    planned = plan(run, tokenizer, resolved)
    documents = extracted_now(run, flatten_plan(planned))
    running = run.run.name if span is None else f"{run.run.name} [{span[0]}, {span[1]})"
    print(f"{running}: layers {layers}, rotary {rotary}, {len(documents)} documents", flush=True)

    # One throwaway wrapper per layer, built for its projections and its rotation and never called forward
    readers = {
        index: model_adapter.attention_class(text_model.layers[index].self_attn, run.model.indexer, rotary)
        for index in layers
    }
    # What a layer file keeps in place of `x`, when it keeps the key and query
    reading = key_and_query_input(run.model.name, rotary) if run.run.input == KEY_AND_QUERY else None
    write_collection_manifests(root, run, planned, every_layer, rotary)
    pending_forced = {_said(document) for document in documents} if force else set()
    dumping = run.data.mass is not None

    # Sweep until nothing is left rather than walking once, so the last process standing finishes
    # whatever a killed one was holding
    while True:
        left = [
            one
            for one in documents
            if _said(one) in pending_forced or not _cached(root, one, layers, dumping)
        ]
        if not left:
            break
        worked = False
        for document in left:
            selected = document.selected
            collection = document.documents.collection
            where = (root, collection, selected.split, selected.name)
            if (
                span is not None
                and span[0] > 0
                and not boundary_ready(boundary_path(root, span[0], *where[1:]))
            ):
                continue
            claimed = selected.name if span is None else f"{selected.name}{Stage.CLAIM_TAG}{span[0]}"
            claim = locate_claim(root, collection, selected.split, claimed)
            if not claim.take():
                continue
            try:
                if _said(document) not in pending_forced and _cached(root, document, layers, dumping):
                    continue
                started = time.monotonic()
                ids, positions, probes = document_queries(
                    run, document.documents, selected, window_length(run, selected), document.index
                )
                ids = ids.to(inputs)
                native, native_hooks = (
                    hook_native_selection(model_adapter, text_model, layers, positions)
                    if run.data.queries.native
                    else (None, [])
                )
                try:
                    if span is None:
                        captured = capture_layer_inputs(text_model, layers, ids, run.run.prefill)
                    else:
                        captured = _run_span(
                            model_adapter, text_model, span, every_layer, layers, ids, run, where
                        )
                finally:
                    for hook in native_hooks:
                        hook.remove()
                forwarded = time.monotonic()

                for index in layers:
                    hidden, position_embeddings = captured[index]
                    if run.data.mass is not None:
                        dump_layer(
                            readers[index],
                            hidden,
                            position_embeddings,
                            positions,
                            mass_path(*where, index),
                            run.model.name,
                            selected.name,
                            run.model.indexer,
                            MassRequest(run.data.mass.top, tuple(run.data.mass.head_tops)),
                        )
                        continue
                    picked = native.of(index, ids.shape[1]) if native is not None else None
                    blob = layer_blob(
                        readers[index],
                        hidden,
                        position_embeddings,
                        positions,
                        run,
                        selected.name,
                        picked,
                        reading,
                        probes,
                    )
                    write(layer_path(*where, index), blob)
                pending_forced.discard(_said(document))
                del captured
                if span is not None and span[0] > 0:
                    remove_boundary(boundary_path(root, span[0], *where[1:]))
                torch.cuda.empty_cache()
                worked = True
                finished = time.monotonic()
                elapsed = finished - started
                # The forward is paid once a document, the targets once a query row
                split = f" - forward {forwarded - started:.0f} s, targets {finished - forwarded:.0f} s"
                # A span is one card, whose peak says how much room its longest document leaves
                peak = ""
                if span is not None and inputs.type == "cuda":
                    peak = f", card peak {torch.cuda.max_memory_allocated(inputs) / 2**30:.1f} GiB"
                    torch.cuda.reset_peak_memory_stats(inputs)
                print(
                    f"{running}: cached {_said(document)} at {ids.shape[1]} tokens in {elapsed:.0f} s{split}{peak}",
                    flush=True,
                )
            finally:
                claim.release()
        if not worked:
            # Everything left is held by something alive, or waits on the span before - wait for it
            print(
                f"{running}: {len(left)} documents with other processes or the span before - waiting",
                flush=True,
            )
            time.sleep(Teacher.CLAIM_POLL)
    print(f"{running}: every document cached", flush=True)
    return root


def resolve_span(run: TeacherConfig, layers: list[int], stage: int | None) -> tuple[int, int] | None:
    """Return the span of the stack one process runs, or None for the whole of it.

    Args:
        run: The extraction.
        layers: Every layer the extraction captures.
        stage: Which span, or None.

    Returns:
        Decoder layers `[start, stop)`, or None.

    Raises:
        ValueError: A stage is asked for and the extraction names no stages, or no span has that number.
    """
    if stage is None:
        return None
    if not run.run.stages:
        raise ValueError(f"{run.run.name} names no stages to cut its stack at, so there is no stage {stage}")
    spans = stage_spans(run.run.stages, layers)
    if not 0 <= stage < len(spans):
        raise ValueError(
            f"stages {run.run.stages} make {len(spans)} spans, numbered from 0 - got stage {stage}"
        )
    return spans[stage]


def _run_span(
    model_adapter: Any,
    text_model: Any,
    span: tuple[int, int],
    every_layer: list[int],
    layers: list[int],
    ids: torch.Tensor,
    run: TeacherConfig,
    where: tuple[Path, str, str, str],
) -> dict[int, tuple[Any, Any]]:
    """Run one span over a document between the cuts either side of it, publishing the cut after it.

    Args:
        model_adapter: The adapter that runs the stack a span at a time.
        text_model: The decoder stack, loaded for this span alone.
        span: Decoder layers `[start, stop)`.
        every_layer: Every layer the extraction captures, whose last says where the last span ends.
        layers: The captured layers inside the span.
        ids: Token ids [1, T] int64, on the span's device.
        run: The extraction.
        where: The cache root, collection, split and document name.

    Returns:
        What `capture_span_inputs` returns.

    Raises:
        ValueError: The input cut's streams are not this document's length.
    """
    root, names = where[0], where[1:]
    source = Boundary(boundary_path(root, span[0], *names)) if span[0] > 0 else None
    if source is not None and source.length != ids.shape[1]:
        raise ValueError(
            f"the streams at layer {span[0]} for {names[-1]} are {source.length} tokens against its {ids.shape[1]}"
        )
    sink = BoundaryWriter(boundary_path(root, span[1], *names)) if span[1] <= max(every_layer) else None
    try:
        captured = capture_span_inputs(
            model_adapter, text_model, span, layers, ids, run.run.prefill, source, sink
        )
    except BaseException:
        if sink is not None:
            sink.discard()
        raise
    if sink is not None:
        sink.publish()
    return captured


def _cached(root: Path, document: Planned, layers: list[int], mass: bool = False) -> bool:
    """Return whether every layer of one document is already written.

    Args:
        root: The cache root.
        document: The document.
        layers: The layers an extraction covers.
        mass: Whether the files are a dense mass dump's rather than layer files.

    Returns:
        True when nothing is left to do for it.
    """
    where = (root, document.documents.collection, document.selected.split, document.selected.name)
    path = mass_path if mass else layer_path
    return all(path(*where, index).exists() for index in layers)


@torch.no_grad()
def layer_blob(
    reader: Any,
    hidden: torch.Tensor,
    position_embeddings: Any,
    positions: torch.Tensor,
    run: TeacherConfig,
    document: str,
    picked: torch.Tensor | None = None,
    reading: KeyAndQueryInput | None = None,
    probes: Probes | None = None,
) -> dict[str, Any]:
    """Build one cached (document, layer).

    Args:
        reader: The layer's sparse-attention wrapper, used for its projections and its rotation.
        hidden: The layer input [1, T, model_dim], on its layer's device or on the host - cached whole as `x`.
        position_embeddings: The model's own `(cos, sin)` tables, or None for a model that does not rotate.
        positions: Absolute query positions [Q] int64.
        run: The extraction.
        document: Document name, recorded in the meta block.
        picked: The checkpoint's own sparse selection at the sampled positions [Q, W] int32, kept with the
            attention the served model spends on it, or None to keep neither.
        reading: How the key and query are read off `x` to be kept in its place - `run.input: key_and_query`, a
            fifth of `x` on GLM - or None to keep `x`.
        probes: The queries a prepared prompt named for itself, whose kinds and targets are kept beside the rest,
            or None for queries that have none.

    Returns:
        The tensors and the meta block, ready to write.
    """
    # A model spread over several cards reads each layer on that layer's own device
    device = next(reader.attention.parameters()).device
    positions = positions.to(device)
    view = reader.teacher_view(hidden, positions, position_embeddings)
    targets = targets_blob(reader, view.q, view.k, positions, run, document, picked)
    if probes is not None:
        targets = {
            **targets,
            **probes.fields(),
            Cache.META: {**targets[Cache.META], Cache.QUERY_KIND_NAMES: list(Cache.QUERY_KINDS)},
        }
    if run.data.queries.attention:
        targets = {
            **targets,
            **attention_fields(reader, view.q, view.k),
            Cache.META: {**targets[Cache.META], Cache.ATTENTION_SCALE: float(reader.attention.scaling)},
        }
    if reading is None:
        return {Cache.X: hidden[0].to(getattr(torch, Cache.X_DTYPE)).contiguous().cpu(), **targets}
    x = hidden[0].to(device=device, dtype=getattr(torch, Cache.X_DTYPE))
    fields = reading.fields(reader.attention, x, positions)
    return {**fields, **targets, Cache.META: {**targets[Cache.META], Cache.INPUT: KEY_AND_QUERY}}


def attention_fields(reader: Any, q: torch.Tensor, k: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the model's own rotated keys and the sampled rows' queries, a position's heads together, on the host.

    Args:
        reader: The layer's sparse-attention wrapper, for the message.
        q: Query heads at the sampled positions [query_head_count, Q, head_dim], rotated.
        k: Key heads at every position [kv_head_count, T, head_dim], rotated.

    Returns:
        The keys [T, kv_head_count, head_dim] bf16 and the queries [Q, query_head_count, head_dim] bf16.

    Raises:
        ValueError: The query heads do not group evenly over the key-value heads.
    """
    if q.shape[0] % k.shape[0] != 0:
        raise ValueError(
            f"layer {reader.attention.layer_idx} has {q.shape[0]} query heads over {k.shape[0]} key-value heads - "
            "queries.attention needs them to group evenly"
        )
    return {
        Cache.ATTENTION_KEYS: k.transpose(0, 1).to(torch.bfloat16).contiguous().cpu(),
        Cache.ATTENTION_QUERIES: q.transpose(0, 1).to(torch.bfloat16).contiguous().cpu(),
    }


@torch.no_grad()
def targets_blob(
    reader: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    run: TeacherConfig,
    document: str,
    picked: torch.Tensor | None = None,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Build what one (document, layer) holds besides `x` - the teacher's attention at the sampled positions, and the meta.

    Args:
        reader: The layer's sparse-attention wrapper, read for the layer's scale and index.
        q: Query vectors at the sampled positions [Hq, Q, dh], on the layer's device.
        k: Key vectors of every position, one set per KV head [Hkv, T, dh], on the layer's device.
        positions: Absolute query positions [Q] int64, on the layer's device.
        run: The extraction.
        document: Document name, recorded in the meta block.
        picked: The checkpoint's own sparse selection at the sampled positions [Q, W] int32, or None.
        block_size: The block a QSA indexer pools keys in, whose target is QSA's, or None for LSA's.

    Returns:
        The targets and the meta block.
    """
    top_key_ids, top_key_mass, peak_key_id = teacher_attention(
        q, k, reader.attention.scaling, positions, run.model.indexer, run.data.queries.top, block_size
    )
    pooled = {} if block_size is None else {Cache.BLOCK_SIZE: block_size}
    native = {}
    if picked is not None:
        native = {
            Cache.NATIVE: picked,
            Cache.NATIVE_MASS: native_attention(q, k, reader.attention.scaling, picked).cpu(),
        }
    return {
        **native,
        "query_positions": positions.int().cpu(),
        "top_key_ids": top_key_ids.cpu(),
        "top_key_mass": top_key_mass.cpu(),
        "peak_key_id": peak_key_id.cpu(),
        Cache.META: {
            "document": document,
            "layer": int(reader.attention.layer_idx),
            Cache.SINK_SIZE: run.model.indexer.sink_size,
            Cache.WINDOW_SIZE: run.model.indexer.window_size,
            Cache.CLUSTER_BLOCK_SIZE: run.model.indexer.window_size,
            Cache.LENGTH: int(k.shape[1]),
            Cache.TRAIN_QUERIES: len(positions) - run.data.queries.tail,
            "top": run.data.queries.top,
            "model": run.model.name,
            **pooled,
        },
    }


# What a job asks for when the command line says nothing
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": Teacher.GPUS,
    "cpus_per_task": Teacher.CPUS,
    "mem_gb": Teacher.MEM_GB,
    "timeout_min": Teacher.HOURS * 60,
}


def stages_submitted(run: TeacherConfig, stage: int | None) -> list[int | None]:
    """Return the spans one extraction's jobs run - the one asked for, every span its stages cut, or the stack.

    Args:
        run: The extraction.
        stage: The one span the command line names, or None for all of them.

    Returns:
        One entry per span to submit, None for the whole stack in one job.
    """
    if stage is not None:
        return [stage]
    if run.run.stages:
        return list(range(len(run.run.stages) + 1))
    return [None]


def main() -> None:
    """Parse the command line and submit the jobs that build every cache the file describes."""
    parser = argparse.ArgumentParser(
        description="Extract the targets a distillation trains on, as slurm jobs"
    )
    parser.add_argument("config", help="path to the extraction yaml")
    slurm.add_flags(parser)
    parser.add_argument("--device", default="cuda:0", help="torch device the weights land on, or auto")
    parser.add_argument("--force", action="store_true", help="re-extract a document that is already cached")
    parser.add_argument(
        "--stage", type=int, default=None, help="submit one span of the stack the config's stages cut"
    )
    parser.add_argument("--workers", type=int, default=1, help="jobs per span, each claiming its documents")
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    extractions = load(args.config, args.overrides)
    for extraction in extractions:
        if extraction.data.selection is not None:
            selection = f"balanced selection in {extraction.run.collection}, lengths {extraction.data.selection.lengths}"
        elif extraction.data.tokenized.documents:
            selection = f"documents selected in {extraction.data.tokenized.documents}"
        else:
            selection = (
                f"{extraction.data.tokenized.tokens_per_document} tokens, {extraction.data.tokenized.train} training and "
                f"{extraction.data.tokenized.evaluate} evaluation documents per source"
            )
        print(
            f"{extraction.run.name}: {extraction.model.name} from {selection}, into {extraction.run.output}, "
            f"spans {stages_submitted(extraction, args.stage)}, {args.workers} workers each",
            flush=True,
        )
    if args.dry_run:
        return

    # A compute node does not start where the submission did
    for extraction in extractions:
        extraction.run.output = str(Path(extraction.run.output).resolve())
        if extraction.data.tokenized.documents:
            extraction.data.tokenized.documents = str(Path(extraction.data.tokenized.documents).resolve())
        if extraction.data.tokenized.path:
            extraction.data.tokenized.path = str(Path(extraction.data.tokenized.path).resolve())
    tasks = [
        (extraction, stage)
        for extraction in extractions
        for stage in stages_submitted(extraction, args.stage)
        for _ in range(args.workers)
    ]
    executor = slurm.executor(
        Teacher.JOB_NAME, env.TEACHER, args.folder, **{**PARAMETERS, **slurm.overrides(args)}
    )
    jobs = executor.map_array(
        extract,
        [extraction for extraction, _ in tasks],
        [args.force] * len(tasks),
        [args.device] * len(tasks),
        [stage for _, stage in tasks],
    )
    for job, (extraction, stage) in zip(jobs, tasks, strict=True):
        span = "the whole stack" if stage is None else f"span {stage}"
        print(f"submitted {job.job_id}  {extraction.run.name} {span}", flush=True)
