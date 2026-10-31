"""Can LSA run this checkpoint - answered from its config, before a GPU is booked.

    lsa probe Qwen/Qwen3.5-4B-Base

Reads `config.json` and nothing else, so it needs no weights, no GPU and no evaluation extra beyond
transformers. It runs as a CPU job, from the hub cache the job's environment names, and the report is in
the job's log.

It is a screen, not the authority. The rotary layout is measured from the model itself in
[adapter.rotary](rotary.py) when `attach` runs, and that measurement is what an experiment uses.
What this catches is the checkpoint that was never going to work - the one that rotates every head dim, and
so has no position-free sub-term for LSA to cluster.
"""

import argparse
import os
from typing import Any

from lsa import env, slurm
from lsa.config import text_config
from lsa.evaluation import require_extra
from lsa.evaluation.constants import Adapter, Layer

# Config fields that name a rotary fraction of the head, in the order they are looked for
_ROTARY_FRACTIONS: tuple[str, ...] = ("partial_rotary_factor", "rotary_pct", "rotary_percentage")

# Config fields that name a rotary dim outright, as the MLA checkpoints do
_ROTARY_DIMS: tuple[str, ...] = ("qk_rope_head_dim",)


def head_dim_of(config: Any) -> int:
    """Return the dims of one attention head.

    Args:
        config: The decoder's config.

    Returns:
        The head dim, from `head_dim` or from the hidden size split across the heads.

    Raises:
        ValueError: The config names neither.
    """
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    hidden_size = getattr(config, "hidden_size", None)
    head_count = getattr(config, "num_attention_heads", None)
    if not hidden_size or not head_count:
        raise ValueError("config names neither head_dim nor hidden_size and num_attention_heads")
    return int(hidden_size) // int(head_count)


def rotary_dim_of(config: Any, head_dim: int) -> int:
    """Return how many head dims the config says are rotated.

    Args:
        config: The decoder's config.
        head_dim: The dims of one head.

    Returns:
        Rotated dims. A config that names no fraction and no dim rotates the whole head.
    """
    for field in _ROTARY_DIMS:
        dim = getattr(config, field, None)
        if dim is not None:
            return int(dim)
    for field in _ROTARY_FRACTIONS:
        fraction = getattr(config, field, None)
        if fraction is not None:
            return int(head_dim * float(fraction))
    return head_dim


def layer_stack(config: Any, wanted: tuple[str, ...] = Layer.DEFAULT_TYPES) -> tuple[int, int, list[str]]:
    """Describe the layer stack and how much of it the indexer would replace.

    Args:
        config: The decoder's config.
        wanted: The layer types to count as quadratic attention.

    Returns:
        layer_count: Layers in the stack.
        matched: Layers of a wanted type, which is all of them when the model declares no types.
        declared: The layer type names the stack uses, empty when it declares none.
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_count = int(getattr(config, "num_hidden_layers", 0))
        return layer_count, layer_count, []
    matched = sum(1 for kind in layer_types if kind in wanted)
    return len(layer_types), matched, sorted(set(layer_types))


def is_mla(config: Any) -> bool:
    """Say whether attention reads a compressed latent rather than per-head keys and values.

    Args:
        config: The decoder's config.

    Returns:
        True for an MLA-style checkpoint.
    """
    return bool(getattr(config, "kv_lora_rank", None))


def checkpoint_config(name: str) -> Any:
    """Read a checkpoint's config, saying so plainly when transformers does not know the architecture.

    Args:
        name: Checkpoint name or path.

    Returns:
        The config.

    Raises:
        ValueError: The installed transformers cannot read this checkpoint's config.
    """
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(name)
    except ValueError as error:
        raise ValueError(
            f"the installed transformers cannot read {name} - {error.args[0].splitlines()[0] if error.args else error}"
        ) from error


def mla_report(name: str, decoder: Any, adapter_name: str, lines: list[str]) -> str:
    """Finish the report for a checkpoint whose attention projects through a compressed latent.

    Args:
        name: Checkpoint name or path.
        decoder: The decoder's config.
        adapter_name: The adapter that would drive it.
        lines: The report so far, appended to.

    Returns:
        The report, ready to print.
    """
    nope = int(getattr(decoder, "qk_nope_head_dim", 0))
    rope = int(getattr(decoder, "qk_rope_head_dim", 0))
    kv_head_count = int(getattr(decoder, "num_key_value_heads", 0) or 0)
    lines += [
        f"  kv latent        {getattr(decoder, 'kv_lora_rank', '?')} dims, shared across heads",
        f"  head dims        {nope} position-free + {rope} rotary, expanded to "
        f"{getattr(decoder, 'num_attention_heads', '?')} heads",
    ]
    if kv_head_count and nope:
        lines.append(f"  position-free    {kv_head_count * nope} dims = {kv_head_count} kv heads x {nope}")

    if adapter_name == Adapter.GENERIC:
        lines.append(
            "\nAttention projects through a compressed latent, so the projections are not `q_proj`, `k_proj` "
            "and `v_proj`.\nThis needs an adapter of its own overriding `project`. Whether it also needs "
            "`read` depends on what the model caches: one that expands the latent into per-head keys and "
            "values before caching them reads like any other model, and one that caches the latent itself "
            "owns the mapping from a selected id back to what it reads."
        )
        return "\n".join(lines)

    lines += [
        "",
        f"Supported - the {adapter_name} adapter unpacks the latent. This runs it:",
        "",
        "model:",
        f"  name: {name}",
    ]
    return "\n".join(lines)


def report(name: str) -> str:
    """Describe what a checkpoint would need, and whether LSA can index it at all.

    Args:
        name: Checkpoint name or path.

    Returns:
        The report, ready to print.
    """
    require_extra("transformers")

    from lsa.evaluation.adapter import ADAPTERS

    config = checkpoint_config(name)
    decoder = text_config(config)
    model_type = str(getattr(config, "model_type", ""))
    adapter_name = model_type if model_type in ADAPTERS else Adapter.GENERIC
    wanted = ADAPTERS[adapter_name].layer_types
    layer_count, matched, declared = layer_stack(decoder, wanted)

    lines = [
        f"{name}",
        f"  model_type       {model_type or 'unnamed'}",
        f"  adapter          {adapter_name}{'' if adapter_name != Adapter.GENERIC else ' - the generic HF decoder'}",
        f"  layers           {layer_count}, {matched} of them a swap target for that adapter",
        f"  layer types      {', '.join(declared) if declared else 'undeclared - the stack is uniform'}",
    ]

    if is_mla(decoder):
        return mla_report(name, decoder, adapter_name, lines)

    head_dim = head_dim_of(decoder)
    rotary_dim = rotary_dim_of(decoder, head_dim)
    kv_head_count = getattr(decoder, "num_key_value_heads", "?")
    lines += [
        f"  heads            {getattr(decoder, 'num_attention_heads', '?')} query, "
        f"{kv_head_count} key-value, {head_dim} dims each",
        f"  rotary           {rotary_dim} of {head_dim} head dims",
    ]

    if rotary_dim >= head_dim:
        lines.append(
            "\nEvery head dim is rotated, so this model has no position-free sub-term and the "
            "training-free index space does not exist for it.\n"
            "LSA needs partial rotary or none at all."
        )
        return "\n".join(lines)

    if matched == 0:
        lines.append(
            f"\nNo layer is called {' or '.join(wanted)}, so nothing would be swapped.\n"
            "Name the type that is this model's quadratic attention over per-token keys and values:\n\n"
            "model:\n"
            f"  name: {name}\n"
            f"  layer_types: [one of {', '.join(declared)}]\n\n"
            "A layer that pools or compresses its keys before attending is not that, and needs an adapter "
            "of its own that overrides `read`."
        )
        return "\n".join(lines)

    position_free_dim = int(kv_head_count) * (head_dim - rotary_dim) if kv_head_count != "?" else "?"
    lines += [
        f"  position-free    {position_free_dim} dims = {kv_head_count} kv heads x {head_dim - rotary_dim}",
        "",
        "Supported. Which end the rotation sits at is measured from the model at attach, so the config",
        "does not have to say - this runs it:",
        "",
        "model:",
        f"  name: {name}",
    ]
    if adapter_name == Adapter.GENERIC and model_type:
        lines.append(f"  # {model_type} has no adapter of its own, so the generic HF decoder path runs it")
    return "\n".join(lines)


def screen(name: str) -> str:
    """Print the report on one checkpoint, or why it could not be read.

    Args:
        name: Checkpoint name or path.

    Returns:
        What was printed.
    """
    try:
        answer = report(name)
    except ValueError as error:
        # A checkpoint this transformers cannot read is an answer, not a crash
        answer = str(error)
    print(answer, flush=True)
    return answer


def main() -> None:
    """Parse the command line and submit the job that screens the checkpoint it names."""
    parser = argparse.ArgumentParser(description="screen a checkpoint before running an experiment on it")
    parser.add_argument("name", help="checkpoint name or path")
    slurm.add_flags(parser)
    args = parser.parse_args()
    # A checkpoint directory is a path to resolve, and a hub id is not
    name = os.path.abspath(args.name) if os.path.isdir(args.name) else args.name
    slurm.submit(args, "lsa-probe", env.OFFLINE_HF, slurm.CPU, screen, name)
