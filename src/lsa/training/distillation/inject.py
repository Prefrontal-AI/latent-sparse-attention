"""Put a distilled index space inside the checkpoint it was distilled from, as a CPU job.

    lsa distill inject --model glm53_flash --maps results/training/glm53_flash/long_16k_1m_heavy_x/maps.pt \
        --out $LSA_ROOT/models/glm53_flash_x

A run writes its maps beside its reports, keyed by layer, and that file means nothing without the note
saying which checkpoint it belongs to. This writes the pair as one thing: a checkpoint directory that
carries its own index space, so the space travels with the weights instead of alongside them.

What comes out is the source checkpoint with two tensors added per indexed layer -
`model.language_model.layers.<i>.self_attn.W_Q_idx` and `W_K_idx`, both `[model_dim, index_dim]` - a
`lsa_index` block in `config.json` recording what they are and where they came from, and the flat
`projections.pt` [the plugin](../../sglang/index/weights.py) reads. One command, and either consumer is served.

The weights are copied, not linked. An injected checkpoint goes under the shared root, and a link into the
snapshot it came from would point into whoever ran this's own HuggingFace cache - unreadable to everyone else
and gone when that cache is cleared. The projections go in a shard of their own, and `config.json` and the
weight index are the only files rewritten.

Nothing is transformed. `W_K_idx` is `key_projection` and `W_Q_idx` is `query_projection`, as `lsa distill export` writes them, because
a reader normalises after projecting exactly as training did.

A map belongs to one checkpoint and this refuses every way of pairing it with another: a recorded teacher
that is not this model, a layer the checkpoint does not attend on, a shape that disagrees between the two
projections of one layer.

A run that trained Qwen's own sparse attention (`model.qsa`) is injected as Qwen's server loads it instead: each
layer's `indexer.index_qk_proj`, `indexer.q_layernorm` and `indexer.k_layernorm` beside its attention, at the
checkpoint's dtype, and the `indexer_*` fields in `text_config` that make the server build the indexer and its sparse
backend. No projection file is written, since no plugin reads one.
"""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lsa import assets, env, slurm
from lsa.io import file_sha256, read_json, write_json
from lsa.sglang.settings.design import ATTENTION_INPUT
from lsa.training import distilled_maps
from lsa.training.constants import Snapshot
from lsa.training.distillation.constants import Inject, Results

# Where a projection sits in the checkpoint's own naming, beside the projections it is fitted to
QUERY: str = "model.language_model.layers.{layer}.self_attn.W_Q_idx"
KEY: str = "model.language_model.layers.{layer}.self_attn.W_K_idx"

# Where a QSA indexer's weights sit, as Qwen's server loads them - the decoder layer's `indexer`
QSA_INDEXER: str = "model.language_model.layers.{layer}.indexer.{name}"
QSA_WEIGHTS: tuple[str, ...] = ("index_qk_proj", "q_layernorm", "k_layernorm")

# One layer's weights, and the block they sit in - Qwen3.5 and GLM-5.3-Flash both name layers this way
LAYER: re.Pattern[str] = re.compile(r"model\.language_model\.layers\.(\d+)\.([^.]+)")

# The block a quadratic layer's attention sits in, and the one a linear layer's does. Qwen3.5 separates
# them by name; GLM-5.3-Flash puts Kimi delta attention under `self_attn` too, which is why a layer is only
# refused when it has the linear block and not the quadratic one
QUADRATIC: str = "self_attn"
LINEAR: str = "linear_attn"


@dataclass(frozen=True)
class Injected:
    """What was written, for the caller that wants to say so."""

    out: Path
    layers: list[int]
    index_dim: int
    model_dim: int
    model: str
    tensors: int


def checkpoint_directory(model: str) -> Path:
    """Return the checkpoint a name or a path points at.

    Args:
        model: A model card name, or a path to a checkpoint directory.

    Returns:
        The checkpoint directory.

    Raises:
        FileNotFoundError: Neither a model card nor a directory on disk.
    """
    direct = Path(model)
    if direct.is_dir():
        return direct
    try:
        return Path(assets.model(model).checkpoint)
    except ValueError as unknown:
        raise FileNotFoundError(
            f"{model} is neither a checkpoint directory nor a registered model card"
        ) from unknown


def layer_blocks(index: dict[str, Any]) -> dict[int, set[str]]:
    """Return the block names each layer of a checkpoint carries.

    Args:
        index: A parsed `model.safetensors.index.json`.

    Returns:
        The first path element under each layer - `self_attn`, `linear_attn`, `mlp` - by layer index.
    """
    found: dict[int, set[str]] = {}
    for name in index["weight_map"]:
        match = LAYER.match(name)
        if match:
            found.setdefault(int(match.group(1)), set()).add(match.group(2))
    return found


def refuse_wrong_layers(path: Path, model: str, layers: list[int], blocks: dict[int, set[str]]) -> None:
    """Refuse a map whose layers are not attention layers of this checkpoint.

    Which layers a model attends on is the adapter's answer and the extraction manifest recorded it, so this
    does not re-derive it from weight names - two families spell their linear layers differently and a rule
    that reads one right reads the other wrong. What it checks is what a checkpoint can say for itself: the
    layer exists, and it is not one whose only attention block is a linear one.

    Args:
        path: The artifact, for the message.
        model: The checkpoint, for the message.
        layers: Layers the artifact holds maps for.
        blocks: Block names by layer, as `layer_blocks` returns them.

    Raises:
        ValueError: A layer is not in the checkpoint, or is a linear-attention layer.
    """
    absent = sorted(layer for layer in layers if layer not in blocks)
    if absent:
        raise ValueError(
            f"{path} holds maps for layers {absent}, which {model} does not have - "
            f"it has {len(blocks)} layers"
        )
    linear = sorted(layer for layer in layers if LINEAR in blocks[layer] and QUADRATIC not in blocks[layer])
    if linear:
        raise ValueError(
            f"{path} holds maps for layers {linear}, which {model} runs linear attention on - "
            "an index applies to the quadratic layers"
        )


def gather(
    maps: list[Path], model: str, blocks: dict[int, set[str]]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Read every run's maps and return one query and key pair per layer.

    Several files are one artifact when a rung was distilled in halves, so this unions them and refuses an
    overlap rather than letting the last file win.

    Args:
        maps: The runs' `maps.pt` files, or directories holding one.
        model: The checkpoint the maps must have been distilled for.
        blocks: Block names by layer, as `layer_blocks` returns them.

    Returns:
        `(query_projection, key_projection)` by layer, each `[model_dim, index_dim]` fp32 on the cpu.

    Raises:
        ValueError: Two files cover the same layer, a map is for another model, a layer is not one the
            checkpoint attends on, or the shapes disagree.
    """
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for where in maps:
        path = where / Results.MAPS if where.is_dir() else where
        artifact = distilled_maps.load(path)
        artifact.checked_for(model, artifact.layers)
        refuse_wrong_layers(artifact.path, model, artifact.layers, blocks)
        both = sorted(set(artifact.layers) & set(out))
        if both:
            raise ValueError(f"{artifact.path} covers layers {both} that another file already covered")
        for layer, projection in artifact.projections.items():
            out[layer] = (
                projection.query_projection.detach().cpu().float(),
                projection.key_projection.detach().cpu().float(),
            )
    if not out:
        raise ValueError("no maps were given")
    shapes = {tuple(query.shape) for query, _ in out.values()}
    if len(shapes) != 1:
        raise ValueError(f"the maps disagree on shape - {sorted(shapes)}")
    return out


def trains_qsa(maps: list[Path]) -> bool:
    """Return whether the runs trained QSA indexers rather than LSA maps.

    Args:
        maps: The runs' `maps.pt` files, or directories holding one.

    Returns:
        True when the first run's maps carry a QSA indexer's weights.
    """
    where = maps[0]
    path = where / Results.MAPS if where.is_dir() else where
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return QSA_WEIGHTS[0] in next(iter(blob["maps"].values()))


def gather_qsa(
    maps: list[Path], model: str, blocks: dict[int, set[str]], dtype: torch.dtype
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read every run's QSA indexers and return their weights under the server's names, and its config fields.

    Args:
        maps: The runs' `maps.pt` files, or directories holding one.
        model: The checkpoint the indexers must have been distilled for.
        blocks: Block names by layer, as `layer_blocks` returns them.
        dtype: The checkpoint's dtype, which the server holds the indexer at.

    Returns:
        weights: Each layer's indexer weights by checkpoint name, at `dtype`.
        fields: The `indexer_*` fields of `text_config`.

    Raises:
        ValueError: A file is for another model, two files cover one layer or disagree on the indexer, a layer is not
            one the checkpoint attends on, or the indexer reads something other than `x`.
    """
    weights: dict[str, torch.Tensor] = {}
    configs = []
    covered: set[int] = set()
    for where in maps:
        path = where / Results.MAPS if where.is_dir() else where
        blob = torch.load(path, map_location="cpu", weights_only=False)
        distilled_for = blob["teacher"]["model"]
        if distilled_for != model:
            raise ValueError(
                f"{path} was distilled for {distilled_for!r} and this injects into {model!r} - an indexer is only "
                "valid for the model whose states trained it"
            )
        layers = sorted(int(layer) for layer in blob["maps"])
        refuse_wrong_layers(path, model, layers, blocks)
        both = sorted(set(layers) & covered)
        if both:
            raise ValueError(f"{path} covers layers {both} that another file already covered")
        covered.update(layers)
        configs.append(blob["run"]["model"]["qsa"])
        for layer, state in blob["maps"].items():
            for name in QSA_WEIGHTS:
                weights[QSA_INDEXER.format(layer=int(layer), name=f"{name}.weight")] = state[name].to(dtype)
    if not weights:
        raise ValueError("no indexers were given")
    if any(config != configs[0] for config in configs):
        raise ValueError(f"the runs trained different indexers - {configs}")
    config = configs[0]
    if config.get("input", ATTENTION_INPUT) != ATTENTION_INPUT:
        raise ValueError(
            f"the runs trained a QSA indexer on {config['input']!r}, and Qwen's server projects `x` alone - "
            "serving it needs the query and key projections apart"
        )
    fields = {
        "indexer_n_heads": config["query_head_count"],
        "indexer_kv_heads": 1,
        "indexer_head_dim": config["head_dim"],
        "indexer_budget": config["top_k"],
        "indexer_compress_ratio": config["block_size"],
    }
    return weights, fields


def copy_unchanged(source: Path, into: Path, rewritten: set[str]) -> None:
    """Copy every file of the source that this does not rewrite, as a file of its own rather than a link.

    Args:
        source: The checkpoint being injected into.
        into: The output directory.
        rewritten: Names this writes itself, which are skipped here.
    """
    for item in sorted(source.iterdir()):
        if item.name in rewritten or item.is_dir():
            continue
        target = into / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copyfile(item.resolve(), target)


def inject(model: str, maps: list[Path], out: Path, force: bool = False) -> Injected:
    """Write a checkpoint that carries its own distilled index space.

    Args:
        model: A model card name or a checkpoint directory.
        maps: The runs' `maps.pt` files, or directories holding one.
        out: Where the injected checkpoint goes.
        force: Replace an output directory that already exists.

    Returns:
        What was written.

    Raises:
        FileExistsError: The output exists and `force` was not given.
        FileNotFoundError: The source has no weight index.
        ValueError: A map does not belong to this checkpoint.
    """
    source = checkpoint_directory(model)
    published = source / Snapshot.INDEX
    if not published.is_file():
        raise FileNotFoundError(f"{source} has no {Snapshot.INDEX} - a sharded checkpoint is what this reads")
    index = read_json(published)
    if trains_qsa(maps):
        return inject_qsa(source, index, model, maps, out, force)

    pairs = gather(maps, model, layer_blocks(index))
    model_dim, index_dim = (int(dimension) for dimension in next(iter(pairs.values()))[0].shape)

    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists - pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    flat: dict[str, torch.Tensor] = {}
    for layer, (query, key) in sorted(pairs.items()):
        flat[QUERY.format(layer=layer)] = query
        flat[KEY.format(layer=layer)] = key

    from safetensors.torch import save_file

    save_file(flat, out / Inject.SHARD, metadata={"format": "pt"})

    weight_map = dict(index["weight_map"])
    weight_map.update({name: Inject.SHARD for name in flat})
    total = int(index.get("metadata", {}).get("total_size", 0))
    grown = total + sum(tensor.numel() * tensor.element_size() for tensor in flat.values())
    write_json(out / Snapshot.INDEX, {"metadata": {"total_size": grown}, "weight_map": weight_map})

    published_config = read_json(source / Snapshot.CONFIG)
    published_config[Inject.BLOCK] = {
        "model_dim": model_dim,
        "index_dim": index_dim,
        "layers": sorted(pairs),
        "teacher": model,
        # The checkpoints the space was read from, so the injected model names the runs that trained it
        "maps": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                where.resolve() / Results.MAPS if where.is_dir() else where.resolve() for where in maps
            )
        ],
        "query": QUERY,
        "key": KEY,
        "projections": Inject.PROJECTIONS,
    }
    write_json(out / Snapshot.CONFIG, published_config)

    from lsa.training.distillation.export import KEY as PLUGIN_KEY
    from lsa.training.distillation.export import QUERY as PLUGIN_QUERY

    torch.save(
        {
            name.format(layer=layer): tensor
            for layer, (query, key) in pairs.items()
            for name, tensor in ((PLUGIN_QUERY, query), (PLUGIN_KEY, key))
        },
        out / Inject.PROJECTIONS,
    )

    copy_unchanged(source, out, {Snapshot.INDEX, Snapshot.CONFIG, Inject.SHARD, Inject.PROJECTIONS})
    return Injected(
        out=out,
        layers=sorted(pairs),
        index_dim=index_dim,
        model_dim=model_dim,
        model=model,
        tensors=len(flat),
    )


def inject_qsa(
    source: Path, index: dict[str, Any], model: str, maps: list[Path], out: Path, force: bool
) -> Injected:
    """Write a checkpoint that carries the QSA indexers a run trained, as Qwen's server loads them.

    Args:
        source: The checkpoint directory.
        index: Its parsed weight index.
        model: The model card name or the checkpoint directory, for the check against the runs.
        maps: The runs' `maps.pt` files, or directories holding one.
        out: Where the injected checkpoint goes.
        force: Replace an output directory that already exists.

    Returns:
        What was written.

    Raises:
        FileExistsError: The output exists and `force` was not given.
    """
    published_config = read_json(source / Snapshot.CONFIG)
    text_config = published_config.get("text_config", published_config)
    dtype = getattr(torch, text_config.get("dtype") or text_config.get("torch_dtype") or "bfloat16")
    weights, fields = gather_qsa(maps, model, layer_blocks(index), dtype)

    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists - pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    from safetensors.torch import save_file

    save_file(weights, out / Inject.SHARD, metadata={"format": "pt"})
    weight_map = dict(index["weight_map"])
    weight_map.update({name: Inject.SHARD for name in weights})
    total = int(index.get("metadata", {}).get("total_size", 0))
    grown = total + sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
    write_json(out / Snapshot.INDEX, {"metadata": {"total_size": grown}, "weight_map": weight_map})

    text_config.update(fields)
    layers = sorted({int(match.group(1)) for name in weights if (match := LAYER.match(name))})
    published_config[Inject.BLOCK] = {
        "kind": "qsa",
        "layers": layers,
        "teacher": model,
        "indexer": fields,
        "maps": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                where.resolve() / Results.MAPS if where.is_dir() else where.resolve() for where in maps
            )
        ],
        "weights": QSA_INDEXER,
    }
    write_json(out / Snapshot.CONFIG, published_config)
    copy_unchanged(source, out, {Snapshot.INDEX, Snapshot.CONFIG, Inject.SHARD})
    head_dim = fields["indexer_head_dim"]
    return Injected(
        out=out,
        layers=layers,
        index_dim=head_dim,
        model_dim=int(next(iter(weights.values())).shape[1]),
        model=model,
        tensors=len(weights),
    )


def inject_and_report(model: str, maps: list[Path], out: Path, force: bool) -> Path:
    """Inject one distilled space into its checkpoint, and print what the checkpoint now carries.

    Args:
        model: Registered asset name, or a checkpoint directory.
        maps: The runs' maps checkpoints, or directories holding them.
        out: Where the injected checkpoint goes.
        force: Replace an output directory that exists.

    Returns:
        The injected checkpoint's directory.
    """
    written = inject(model, maps, out, force)
    print(
        f"{written.out}: {written.model}, layers {written.layers}, "
        f"{written.model_dim} -> {written.index_dim}, {written.tensors} tensors",
        flush=True,
    )
    return written.out


def main() -> None:
    """Submit the injection of one distilled space into the checkpoint it came from."""
    parser = argparse.ArgumentParser(
        description="Put a distilled index space inside its checkpoint, as a CPU job"
    )
    parser.add_argument("--model", required=True, help="model card name, or a checkpoint directory")
    parser.add_argument(
        "--maps",
        required=True,
        nargs="+",
        type=Path,
        help="the runs' maps.pt, or directories holding them - several when a rung distilled in halves",
    )
    parser.add_argument("--out", required=True, type=Path, help="where the injected checkpoint goes")
    parser.add_argument("--force", action="store_true", help="replace an output directory that exists")
    slurm.add_flags(parser)
    args = parser.parse_args()

    # A checkpoint directory is a path to resolve, and an asset name is not
    model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    maps = [path.resolve() for path in args.maps]
    slurm.submit(
        args,
        "lsa-inject",
        env.OFFLINE_HF,
        slurm.CPU,
        inject_and_report,
        model,
        maps,
        args.out.resolve(),
        args.force,
    )
