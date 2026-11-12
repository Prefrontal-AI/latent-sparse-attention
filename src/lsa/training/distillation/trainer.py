"""Train learned index projections."""

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.io import write_json
from lsa.sglang.settings.design import KEY_AND_QUERY, QSAConfig
from lsa.training import distilled_maps, require_extra
from lsa.training.distillation.constants import Cache, Divergence, Pool, Results, Space, Target, Term
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.data.native import KeyAndQueryReader, key_and_query_input, load_attention
from lsa.training.distillation.data.online import OnlineTeacher, teacher_devices
from lsa.training.distillation.data.pool import (
    Entry,
    Interrupted,
    Item,
    Rolling,
    catalog,
    device_budget,
    drawn_documents,
    host_budget,
    interleave,
    items_of,
    layer_steps,
    remaining,
    require_native,
    shared_order,
)
from lsa.training.distillation.data.teacher_nodes import Staging
from lsa.training.distillation.losses import (
    calibration_grid,
    calibration_losses,
    choose_tau,
    step_terms,
    target_entropy,
    teacher_entropy,
    weighted,
)
from lsa.training.distillation.model import Indexer, IndexMap, QSAIndexer, init_rows
from lsa.training.distillation.tracking import Tracker

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig


from lsa.training.distillation.checkpoint import (
    Preemption,
    cached_rotary,
    load_maps,
    meta,
    provenance,
    read_resume,
    resolve_layers,
    resumed_maps,
    run_directory,
    save_maps,
    save_resume,
    shard_label,
)
from lsa.training.distillation.reporting import load_scoring, report, scored_entries


def train(
    experiment: "DistillConfig", force: bool = False, device: str = "cuda:0", shard: list[int] | None = None
) -> Path:
    """Train one run's maps and write its reports, or one shard's share of them.

    The signature puts `force` before `device` so `submit` can map it over runs without a wrapper.

    A run whose directory holds a resume state continues from it - see [checkpoint](checkpoint.py) - and a job told
    it is about to be stopped writes one at the step it has reached and is requeued.

    In a job of several nodes taught online, every node but the first serves the teacher instead - see
    [teacher_nodes](data/teacher_nodes.py).

    Args:
        experiment: The experiment.
        force: Retrain a run whose maps are already written, and start over one that would resume.
        device: Torch device the targets and the maps land on.
        shard: The layers this process trains, of the run's, or None for all of them - see
            [shard](shard.py).

    Returns:
        The directory the run, or the shard, wrote into.

    Raises:
        ValueError: The shard is empty, or names a layer the run does not train.
    """
    require_extra("omegaconf")
    out = run_directory(experiment, shard)
    staging = None
    if experiment.data.teacher is not None:
        staging = Staging.of_job(out, experiment.data.teacher.staging_gb)
    if staging is not None and staging.serving:
        return serve(experiment, staging, shard)
    if staging is not None:
        staging.open()
    try:
        return train_here(experiment, force, device, shard, staging)
    finally:
        # The serving nodes stop once the trainer takes nothing more, however it ended
        if staging is not None:
            staging.finish()


def train_here(
    experiment: "DistillConfig", force: bool, device: str, shard: list[int] | None, staging: Staging | None
) -> Path:
    """Train on this node - the body of `train`, with the teacher's other nodes, if any, behind `staging`.

    Args:
        experiment: The experiment.
        force: Retrain a run whose maps are already written, and start over one that would resume.
        device: Torch device the targets and the maps land on.
        shard: The layers this process trains, or None for all of them.
        staging: Where the teacher's copies on other nodes leave what they read, or None for one node.

    Returns:
        The directory the run, or the shard, wrote into.

    Raises:
        ValueError: The shard is empty, or names a layer the run does not train.
    """
    out = run_directory(experiment, shard)
    if (out / Results.MAPS).exists() and not force:
        print(f"{experiment.run.name}: complete - nothing to run", flush=True)
        return out
    if force:
        (out / Results.RESUME).unlink(missing_ok=True)
    resumed = read_resume(out / Results.RESUME, experiment, shard)

    teacher = None
    if experiment.data.teacher is not None:
        teacher = OnlineTeacher(
            experiment.data.teacher,
            experiment.model.indexer,
            experiment.model.name,
            staging,
            experiment.model.qsa,
        )
    recorded = provenance(experiment, teacher)
    layers = resolve_layers(experiment)
    if shard is not None and not shard:
        raise ValueError(f"{experiment.run.name}: a shard trains at least one layer")
    foreign = sorted(set(shard or []) - set(layers))
    if foreign:
        raise ValueError(
            f"{experiment.run.name}: the shard trains layers {foreign} the run does not - it trains {layers}"
        )
    trained_layers = [layer for layer in layers if shard is None or layer in shard]
    torch.manual_seed(experiment.run.optim.seed)
    # The projection reads bf16 `x` against fp32 weights, and tf32 keeps the weights' ten mantissa bits
    torch.set_float32_matmul_precision("high")

    if teacher is not None:
        teacher.start(teacher_devices(device), trained_layers)
        entries = teacher.catalog(experiment.data.train)
        # The held-out sets and the init's sample are read before step one - queued at once, every copy reads
        scored = [entry for one in experiment.data.evaluate for entry in scored_entries(one, teacher)]
        sample = init_entries(entries, experiment.run.optim.seed) if resumed is None else []
        teacher.prefetch([*scored, *sample], trained_layers)
    else:
        entries = catalog(experiment)
    scoring = load_scoring(experiment, trained_layers, teacher)
    if experiment.run.objective.target in Target.PICKED:
        require_native(entries, trained_layers)
    print(
        f"{experiment.run.name}: {len(entries)} training documents, layers {trained_layers} of {layers}, "
        f"scored on {', '.join(scoring)}",
        flush=True,
    )

    reader = input_reader(experiment, trained_layers, device) if teacher is None else None
    if resumed is None:
        maps = build_maps(experiment, entries, trained_layers, device, reader, teacher)
    else:
        maps = resumed_maps(resumed, device, experiment.model.qsa)
    optimiser = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for item in maps.values()
                    for name, parameter in item.named_parameters()
                    if name != "log_tau"
                ],
                "lr": experiment.run.optim.lr,
            },
            {"params": [m.log_tau for m in maps.values()], "lr": experiment.run.optim.tau_lr},
        ],
        weight_decay=0.0,
    )
    parameters = [parameter for item in maps.values() for parameter in item.parameters()]

    tracker = Tracker(
        experiment,
        out,
        label=None if shard is None else shard_label(shard),
        resume_id=None if resumed is None else resumed["wandb"],
    )
    write_json(out / Results.META, meta(experiment, device, maps, recorded))
    if resumed is None:
        begun, trained = 0, 0
        untrained = report(experiment, maps, scoring, 0, reader)
        write_json(out / Results.REPORT.format(step=0), untrained)
        tracker.report(0, untrained)
        save_maps(out / Results.SNAPSHOT.format(step=0), maps, recorded, 0)
    else:
        begun, trained = resumed["step"], resumed["trained"]
        optimiser.load_state_dict(resumed["optimiser"])
        restore_rng(resumed["rng"])
        print(f"{experiment.run.name}: resumed at step {begun} from {out / Results.RESUME}", flush=True)

    # A run of no steps is the init and the export, and has no document to draw
    pools = (
        rolling_pools(
            experiment,
            layers,
            trained_layers,
            entries,
            device,
            reader,
            teacher,
            None if resumed is None else resumed["pools"],
        )
        if experiment.run.optim.steps
        else {}
    )
    del resumed

    def save_state(step: int) -> None:
        """Write everything the run continues from, at a step every pool has caught up to."""
        state = {
            "run": asdict(experiment),
            "shard": shard,
            "step": step,
            "trained": trained,
            "maps": {layer: item.state_dict() for layer, item in maps.items()},
            "optimiser": optimiser.state_dict(),
            "rng": rng_state(),
            "pools": {layer: pool.state() for layer, pool in pools.items()},
            "wandb": tracker.run_id,
        }
        save_resume(out / Results.RESUME, state)

    preemption = Preemption()
    pace_began, pace_steps = time.monotonic(), 0
    step = begun
    try:
        for step in range(begun + 1, experiment.run.optim.steps + 1):
            if preemption.requested.is_set():
                raise Interrupted(f"stopped before step {step}")
            layer = layers[step % len(layers)]
            if layer in pools:
                drawn = pools[layer].next(step, preemption.requested)
                index_map = maps[drawn.layer]
                rows = drawn.rows.to(device)
                terms = step_terms(index_map, drawn.document, rows, experiment.run.objective)
                step_loss = weighted(terms)
                optimiser.zero_grad()
                step_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, experiment.run.optim.grad_clip)
                optimiser.step()
                trained += 1
                pace_steps += 1
                printed = trained % experiment.run.report.log_every == 0
                sent = trained % experiment.run.wandb.log_every == 0
                if printed or sent:
                    logged = logged_terms(experiment, terms, drawn.document, rows)
                if printed:
                    parts = " ".join(f"{name} {value:.4f}" for name, value in logged.items())
                    pace = pace_steps / (time.monotonic() - pace_began)
                    pace_began, pace_steps = time.monotonic(), 0
                    peak = ""
                    if torch.device(device).type == "cuda":
                        peak = f", card peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
                    print(
                        f"{experiment.run.name}: step {step} layer {drawn.layer} loss {float(step_loss.detach()):.4f} "
                        f"{parts} tau {index_map.tau:.3f} - {pace:.2f} steps/s{peak}",
                        flush=True,
                    )
                    print(f"{experiment.run.name}: pool {pools[drawn.layer].describe()}", flush=True)
                    if teacher is not None:
                        print(f"{experiment.run.name}: teacher {teacher.describe()}", flush=True)
                if sent:
                    tracker.step(
                        step, drawn.layer, float(step_loss.detach()), index_map.tau, float(grad_norm), logged
                    )
            if step % experiment.run.report.every == 0 or step == experiment.run.optim.steps:
                measured = report(experiment, maps, scoring, step, reader)
                write_json(out / Results.REPORT.format(step=step), measured)
                tracker.report(step, measured)
                save_maps(out / Results.SNAPSHOT.format(step=step), maps, recorded, step)
                save_state(step)
    except Interrupted:
        # The step that was stopped never started, so the run stands at the one before it
        save_state(step - 1)
        print(f"{experiment.run.name}: preempted - resume state written at step {step - 1}", flush=True)
        close(pools, teacher, tracker)
        preemption.hand_over()
    finally:
        preemption.close()

    close(pools, teacher, tracker)
    save_maps(out / Results.MAPS, maps, recorded, experiment.run.optim.steps)
    print(f"{experiment.run.name}: wrote {out}", flush=True)
    return out


def serve(experiment: "DistillConfig", staging: Staging, shard: list[int] | None = None) -> Path:
    """Serve the teacher to the trainer on another node - a copy on every card here, reading its requests.

    Args:
        experiment: The experiment.
        staging: This node's view of the job's staging.
        shard: The layers the run trains, or None for all of them.

    Returns:
        The run directory, which this node writes nothing into but the staged documents.

    Raises:
        ValueError: The run reads a cache, which no other node has anything to serve of.
    """
    out = run_directory(experiment, shard)
    if experiment.data.teacher is None:
        raise ValueError(f"{experiment.run.name} reads a cache - only the online teacher serves across nodes")
    if (out / Results.MAPS).exists():
        print(f"{experiment.run.name}: complete - nothing to serve", flush=True)
        return out
    teacher = OnlineTeacher(
        experiment.data.teacher,
        experiment.model.indexer,
        experiment.model.name,
        staging,
        experiment.model.qsa,
    )
    layers = [layer for layer in resolve_layers(experiment) if shard is None or layer in shard]
    teacher.start(teacher_devices(None), layers)
    print(f"{experiment.run.name}: node {staging.node} of {staging.nodes} serving the teacher", flush=True)
    try:
        teacher.serve()
    finally:
        print(f"{experiment.run.name}: teacher {teacher.describe()}", flush=True)
        teacher.close()
    return out


def requeued(
    experiment: "DistillConfig", force: bool = False, device: str = "cuda:0", shard: list[int] | None = None
) -> Any:
    """Return the job a preempted `train` is requeued as - the same run, resuming rather than forced over.

    submitit calls this on SIGUSR2, once the trainer has written its resume state and handed the signal on.

    Args:
        experiment: The experiment.
        force: Whether the job was submitted to start over, which its requeue must not do again.
        device: The training device.
        shard: The layers the job trains.

    Returns:
        The submission to requeue.
    """
    from submitit.helpers import DelayedSubmission

    return DelayedSubmission(train, experiment, False, device, shard)


# submitit requeues a job whose function carries a `checkpoint`
train.checkpoint = requeued  # type: ignore[attr-defined]


def close(pools: dict[int, Rolling], teacher: OnlineTeacher | None, tracker: Tracker) -> None:
    """Stop the pools and the teacher, and finish the wandb run.

    Args:
        pools: Each trained layer's pool.
        teacher: The online teacher, or None.
        tracker: The run's tracker.
    """
    for pool in pools.values():
        pool.close()
    if teacher is not None:
        print(f"teacher: {teacher.describe()}", flush=True)
        teacher.close()
    tracker.close()


def rng_state() -> dict[str, Any]:
    """Return the global generators' states, the host's and every card's."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Put the global generators back as `rng_state` found them.

    Args:
        state: What `rng_state` returned.
    """
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def logged_terms(
    experiment: "DistillConfig", terms: dict[str, torch.Tensor], document: Document, rows: torch.Tensor
) -> dict[str, float]:
    """Return a step's terms as numbers, with the per-key divergence above the target's own entropy.

    The forward divergence is a cross-entropy, whose floor is the target's entropy, and that entropy moves
    with a document's length and with the depth the target is cut to. The KL is comparable across both.

    Args:
        experiment: The experiment, for the divergence and the depth.
        terms: The step's terms, as `step_terms` returns them.
        document: The step's document.
        rows: The step's query rows [R] int64.

    Returns:
        Every term by name, plus `kl` under the forward divergence.
    """
    logged = {name: float(value.detach()) for name, value in terms.items()}
    if experiment.run.objective.divergence == Divergence.FORWARD:
        objective = experiment.run.objective
        entropy = target_entropy(document, rows, objective.target_depth, objective.target, objective.tail)
        logged[Results.KL] = logged[Term.KEY] - float(entropy)
    return logged


def input_reader(experiment: "DistillConfig", layers: list[int], device: str) -> KeyAndQueryReader | None:
    """Return what reads the run's input off a document's `x`, which `x` itself needs nothing of.

    Under the key and query it is the layers' own attention, loaded onto the training device.

    Args:
        experiment: The experiment, for the input and the model.
        layers: The layers this process trains.
        device: The training device.

    Returns:
        The key and query reader, or None for `x`.
    """
    if experiment.model.indexer.input != KEY_AND_QUERY:
        return None
    return KeyAndQueryReader(
        load_attention(experiment.model.name, layers, device),
        experiment.run.objective.closed_block,
        key_and_query_input(experiment.model.name, cached_rotary(experiment)),
    )


def rolling_pools(
    experiment: "DistillConfig",
    layers: list[int],
    trained_layers: list[int],
    entries: list[Entry],
    device: str,
    reader: Callable[[Document], Document] | None = None,
    teacher: OnlineTeacher | None = None,
    resumed: dict[int, dict[str, Any]] | None = None,
) -> dict[int, Rolling]:
    """Start one pool per trained layer, each reading its own layer's share of the run's draw.

    The memory a process has is shared between its layers equally. Fed by the online teacher, the layers share one
    arrival order and the teacher is handed every document they draw, in that order.

    A resumed run takes each layer's plan from its resume state rather than working it out again - the items' sizes
    and the card's budget the plan was made under are what they were then - and trains what is left of it.

    Args:
        experiment: The experiment, for the step count, the draw rule, the batch, the window and the seed.
        layers: Every layer of the run, which steps rotate over.
        trained_layers: The layers this process trains.
        entries: The training documents.
        device: The training device.
        reader: Reads the run's input off a document once it is on the device, or None to train on `x`.
        teacher: The online teacher the documents come from, or None to read them off the cache.
        resumed: Each trained layer's pool state at the step the run resumes from, or None to start at step 0.

    Returns:
        The pool of each trained layer, already reading.
    """
    optim = experiment.run.optim
    drawn = drawn_documents(entries, optim.steps, optim.seed, optim.draw)
    dealt = layer_steps(layers, optim.steps)
    host_bytes = host_budget(experiment.data.pool_gb, len(trained_layers))
    # Emptying the cache empties every card's, which the online teacher's copies are reading on
    if torch.device(device).type == "cuda" and teacher is None:
        torch.cuda.empty_cache()
    device_bytes = device_budget(device, experiment.data.resident_gb, len(trained_layers))
    if torch.device(device).type != "cuda":
        device_bytes = host_bytes
    order = shared_order(drawn, optim.seed) if teacher is not None else None
    if resumed is None:
        size_of = teacher.size_of if teacher is not None else None
        items = {
            layer: items_of(entries, drawn, layer, dealt[layer], optim.seed, order, size_of)
            for layer in trained_layers
        }
        sequences = {
            layer: interleave(items[layer], device_bytes, experiment.data.window, optim.seed + layer)
            for layer in trained_layers
        }
        budgets = dict.fromkeys(trained_layers, device_bytes)
    else:
        items, sequences, budgets = {}, {}, {}
        for layer in trained_layers:
            saved = resumed[layer]
            items[layer], sequences[layer] = remaining(saved["items"], saved["sequence"], saved["cursor"])
            budgets[layer] = saved["device_bytes"]
    load = None
    if teacher is not None and order is not None:
        teacher.submit(*teacher_requests(entries, order, items))
        load = teacher.take_item

    pools = {}
    for layer in trained_layers:
        sequence = sequences[layer]
        pools[layer] = Rolling(
            layer,
            items[layer],
            sequence,
            experiment.model.indexer,
            device,
            host_bytes,
            budgets[layer],
            optim.batch,
            optim.seed,
            native=experiment.run.objective.target in Target.PICKED,
            reader=reader,
            load=load,
        )
        if resumed is not None:
            pools[layer].generator.set_state(resumed[layer]["generator"])
        print(
            f"{experiment.run.name}: layer {layer} trains {len(sequence)} steps on {len(items[layer])} documents, "
            f"{sum(item.size for item in items[layer]) / 1e9:.0f} GB to read, pool {host_bytes / 1e9:.0f} GB of host "
            f"memory and {budgets[layer] / 1e9:.0f} GB of {device}, window {experiment.data.window}",
            flush=True,
        )
    return pools


def teacher_requests(
    entries: list[Entry], order: list[int], items: dict[int, list[Item]]
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], list[int]]]:
    """Return what the teacher reads for the pools - every document some layer draws, in the order they share.

    Args:
        entries: The training documents.
        order: The shared arrival order, as indices into `entries`.
        items: Each trained layer's items.

    Returns:
        order: Each document as `(collection, name)`, in arrival order.
        wanted: The layers each document is read for.
    """
    wanted: dict[tuple[str, str], list[int]] = {}
    for layer, layer_items in items.items():
        for item in layer_items:
            wanted.setdefault((item.entry.where[1], item.entry.name), []).append(layer)
    keys = [(entries[index].where[1], entries[index].name) for index in order]
    return [key for key in keys if key in wanted], wanted


def init_entries(
    entries: list[Entry], seed: int, layer: int | None = None, count: int = Pool.INIT_DOCUMENTS
) -> list[Entry]:
    """Return the documents a layer's map is started and its temperature calibrated from.

    They are drawn in proportion to their tokens, as a step draws, without replacement - for each layer apart off the
    cache, and once for every layer off the online teacher, which reads a document for all of them at once.

    Args:
        entries: The training documents.
        seed: The run's seed.
        layer: The layer, or None for one draw every layer shares.
        count: How many to draw.

    Returns:
        The drawn documents.
    """
    generator = torch.Generator().manual_seed(seed + 2000 + (layer or 0))
    weights = torch.tensor([entry.tokens for entry in entries], dtype=torch.float32)
    picked = torch.multinomial(weights, min(count, len(entries)), replacement=False, generator=generator)
    return [entries[int(index)] for index in picked]


def init_paths(entries: list[Entry], layer: int, seed: int, count: int = Pool.INIT_DOCUMENTS) -> list[Path]:
    """Return the files of the documents a layer's map is started and its temperature calibrated from.

    Args:
        entries: The training documents.
        layer: The layer.
        seed: The run's seed.
        count: How many to draw.

    Returns:
        The layer's file of each drawn document.
    """
    return [entry.path(layer) for entry in init_entries(entries, seed, layer, count)]


def build_maps(
    experiment: "DistillConfig",
    entries: list[Entry],
    layers: list[int],
    device: str,
    reader: KeyAndQueryReader | None = None,
    teacher: OnlineTeacher | None = None,
) -> dict[int, Indexer]:
    """Build one map per layer and give each the temperature it should start from.

    A map starts at the leading PCA directions of its own layer's `x`, taken from a sample of documents, or at
    a checkpoint's when one is named. The temperature is whatever the config pins, else the checkpoint's when
    there is one, else the grid value the untrained map is least wrong at - calibrating a warm-started map
    would throw away the temperature it was trained under.

    Under the key and query the map acts in the layer's latent bases, and starts at the PCA of its key latents
    carried into the key basis - the original implementation's init.

    A document is up to 8.7 GB, so init holds one at a time: it reads each, keeps its sampled rows and lets
    it go, and reads the sample again to calibrate once the map exists.

    A QSA indexer starts at random, seeded by the run, with the teacher's own rotary frequencies, and the sample
    calibrates its temperature alone.

    Args:
        experiment: The experiment.
        entries: The training documents.
        layers: The layers to build.
        device: Where the maps land.
        reader: Reads the key and query off a document's `x`, or None to train on `x`.
        teacher: The online teacher, which reads the sample for every layer at once, or None to read the cache.

    Returns:
        One map, or QSA indexer, per layer.
    """
    indexer = experiment.model.indexer
    seed = experiment.run.optim.seed
    objective = experiment.run.objective
    native = objective.target in Target.PICKED
    if teacher is not None:
        drawn = init_entries(entries, seed)
        held = teacher.read_now(drawn, layers)

        def sample(layer: int) -> Iterator[Document]:
            """Yield the layer's sample, as the teacher read it."""
            yield from held[layer]

    else:
        paths = {layer: init_paths(entries, layer, seed) for layer in layers}

        def sample(layer: int) -> Iterator[Document]:
            """Yield the layer's sample a document at a time, read off the cache under the run's input."""
            for path in paths[layer]:
                yield read_input(Document(path, indexer, Cache.HOST, native=native), reader, device)

    maps: dict[int, Indexer] = {}
    for layer in layers:
        if experiment.model.qsa is not None and teacher is not None:
            maps[layer] = qsa_indexer(experiment.model.qsa, teacher, next(sample(layer)).model_dim).to(device)
            continue
        sampled = []
        count = min(Pool.INIT_DOCUMENTS, len(entries))
        for document in sample(layer):
            # The rows are copied, so the document's input is freed with the document
            sampled.append(init_rows(document.key_inputs(), Space.INIT_ROWS // count).clone())
            del document
        rows = torch.cat(sampled).to(device)
        bases = (
            reader.bases(layer)
            if reader is not None
            else teacher.bases(layer)
            if teacher is not None
            else None
        )
        maps[layer] = IndexMap(rows.shape[1], indexer.index_dim, init=rows, bases=bases).to(device)
    if experiment.model.space.init:
        path = Path(experiment.model.space.init)
        blob = torch.load(path, map_location=device, weights_only=False)
        if experiment.model.qsa is None:
            distilled_maps.decode(blob, path).checked_for(experiment.model.name, list(maps))
        load_maps(blob, maps)
        print(f"{experiment.run.name}: warm started from {experiment.model.space.init}", flush=True)

    for layer, item in maps.items():
        if experiment.model.space.tau is not None:
            item.set_tau(experiment.model.space.tau)
            print(f"{experiment.run.name}: layer {layer} tau {item.tau:.3f} as configured", flush=True)
        elif experiment.model.space.init:
            print(f"{experiment.run.name}: layer {layer} tau {item.tau:.3f} from the checkpoint", flush=True)
        else:
            grid = calibration_grid(item)
            cross_entropy = torch.zeros_like(grid)
            entropies = []
            for document in sample(layer):
                cross_entropy += calibration_losses(
                    item,
                    document,
                    grid,
                    depth=objective.target_depth,
                    target=objective.target,
                    tail=objective.tail,
                )
                entropies.append(
                    teacher_entropy(
                        [document],
                        device=device,
                        target=objective.target,
                        depth=objective.target_depth,
                        tail=objective.tail,
                    )
                )
                del document
            chosen = choose_tau(item, grid, cross_entropy)
            print(
                f"{experiment.run.name}: layer {layer} calibrated tau {chosen:.3f}, "
                f"teacher entropy {sum(entropies) / len(entropies):.3f}",
                flush=True,
            )
    return maps


def qsa_indexer(config: QSAConfig, teacher: OnlineTeacher, model_dim: int) -> QSAIndexer:
    """Build one layer's QSA indexer at its random start, rotating at the teacher's own frequencies.

    Args:
        config: The indexer's shape.
        teacher: The online teacher, whose copies are loaded.
        model_dim: Dims of `x`.

    Returns:
        The indexer, on the host.

    Raises:
        ValueError: No copy of the teacher has been loaded to read the frequencies off.
    """
    if teacher.rotary_frequencies is None:
        raise ValueError("a QSA indexer rotates at the teacher's frequencies, and no copy of it is loaded")
    inverse_frequencies, attention_scaling = teacher.rotary_frequencies
    return QSAIndexer(model_dim, config, inverse_frequencies, attention_scaling)


def read_input(document: Document, reader: KeyAndQueryReader | None, device: str) -> Document:
    """Return a document as the run reads it - as read off the disk, or its key and query read on the device.

    Args:
        document: A document as read off the disk.
        reader: Reads the key and query off its `x`, or None for `x` itself.
        device: Where the reader's attention is.

    Returns:
        The document, where it was for `x` and on the device under the key and query.
    """
    if reader is None:
        return document
    return reader(document.to(device))
