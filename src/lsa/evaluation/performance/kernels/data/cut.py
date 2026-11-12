"""Truncated GLM-5.3-Flash or Qwen3.5 for single-GPU engine tests - dummy weights or the real first layers.

The full checkpoint needs more than one device, and the layer harness drives one. So the harness is pointed
at the first few layers instead: the config is rewritten and only the shards those layers name are copied,
which is enough to measure layer 3 and the linear layer after it and nothing more.

    lsa kernels cut --out $LSA_ROOT/models/glm53_flash_real5
    lsa kernels cut --snapshot $HF_HOME/hub/models--Qwen--Qwen3.8-27b/snapshots/<revision> --out $LSA_ROOT/models/qwen38_27b_real5

The defaults are the five layers and real weights every kernel config reads; `--layers` and `--no-weights` change them.

It reads the snapshot's config and copies its files, so it needs neither a GPU nor the container, and runs as
a CPU job.
"""

import argparse
import glob
import json
import os
import shutil

from lsa import env, io, slurm
from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.family import model_family

MODEL = "zai-org/GLM-5.3-Flash"
# Five layers make layer 3 the sparse one, which every kernel config reads
LAYERS = 5
SIDE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
)


def default_snapshot() -> str:
    """Return the first GLM-5.3-Flash snapshot in the hub cache this process reads.

    The cache is `env.hf_home()`'s - `HF_HOME`, then the site's `hf_home`, then the hub's default.

    Raises:
        FileNotFoundError: The cache holds no snapshot of the model.
    """
    cache = env.hf_home() / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots"
    snapshots = sorted(glob.glob(os.path.join(glob.escape(str(cache)), "*/")))
    if not snapshots:
        raise FileNotFoundError(
            f"no snapshot of {MODEL} under {cache} - fetch it with `hf download {MODEL}`, "
            "or name a checkpoint with --snapshot"
        )
    return snapshots[0]


def truncate(config, layers):
    """Rewrite a model config to keep its first decoder layers.

    Args:
        config: The checkpoint's `config.json`, edited in place.
        layers: Decoder layers kept.

    Returns:
        The edited config.
    """
    text = config["text_config"]
    text["num_hidden_layers"] = layers
    # The multi-token prediction head needs its own weights and is outside the engine tests
    # => drop it
    if model_family(config["text_config"]["model_type"]) == Family.QWEN3_5:
        text["layer_types"] = text["layer_types"][:layers]
        text["mtp_num_hidden_layers"] = 0
        return config
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[key] = text[key][:layers]
    kda = text["linear_attn_config"]["kda_layers"]
    text["linear_attn_config"]["kda_layers"] = [i for i in kda if i < layers]
    text["num_nextn_predict_layers"] = 0
    return config


def layer_of(name):
    """Return the decoder layer a weight name belongs to, or -1 when it names none."""
    if ".layers." not in name:
        return -1
    return int(name.split(".layers.")[1].split(".")[0])


def kept(name, layers):
    """Return whether a weight belongs to the first `layers` decoder layers or to no layer."""
    # Qwen's prediction head lives under `mtp.layers.*`, which would otherwise pass as a decoder layer
    return not name.startswith("mtp.") and layer_of(name) < layers


def link(source, target) -> None:
    """Copy a file to `target` unless it is already there."""
    # A copy rather than a symlink: a cut goes under the shared root, and a link would point into whoever cut
    # it's own HuggingFace cache
    if not os.path.exists(target):
        shutil.copyfile(os.path.realpath(source), target)


def link_weights(snapshot, out, layers):
    """Write the weight index of the kept layers and copy the shards it names.

    Args:
        snapshot: The checkpoint to cut.
        out: Where the cut goes.
        layers: Decoder layers kept.

    Returns:
        The number of shards copied.
    """
    # Keep the shards the kept layers need - the loader reads only files named in the index
    index = io.read_json(f"{snapshot}/model.safetensors.index.json")
    weights = {name: shard for name, shard in index["weight_map"].items() if kept(name, layers)}
    index["weight_map"] = weights
    with open(f"{out}/model.safetensors.index.json", "w") as target:
        json.dump(index, target, indent=2)
    for shard in sorted(set(weights.values())):
        link(f"{snapshot}/{shard}", f"{out}/{shard}")
    return len(set(weights.values()))


def cut(snapshot: str, out: str, layers: int, weights: bool) -> str:
    """Write a checkpoint of a snapshot's first layers - its config rewritten, its files copied.

    Args:
        snapshot: The checkpoint to cut.
        out: Where the cut goes.
        layers: Decoder layers kept.
        weights: Link the real shards of the kept layers, rather than leaving the weights to a dummy load.

    Returns:
        The cut's directory.
    """
    config = truncate(io.read_json(f"{snapshot}/config.json"), layers)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/config.json", "w") as target:
        json.dump(config, target, indent=2)
    for name in SIDE_FILES:
        if os.path.exists(f"{snapshot}/{name}"):
            link(f"{snapshot}/{name}", f"{out}/{name}")
    shards = link_weights(snapshot, out, layers) if weights else 0
    types = config["text_config"]["layer_types"]
    attention = [
        i
        for i, kind in enumerate(types)
        if kind in Family.ATTENTION_LAYERS[model_family(config["text_config"]["model_type"])]
    ]
    print(f"{out}: {layers} layers, attention {attention}, shards {shards}", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of `lsa kernels cut`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=LAYERS, help="decoder layers kept")
    parser.add_argument(
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the real shards of the kept layers, rather than leaving the weights to a dummy load",
    )
    parser.add_argument(
        "--snapshot", help="the checkpoint to cut - GLM-5.3-Flash from the hub cache by default"
    )
    parser.add_argument("--out", required=True)
    slurm.add_flags(parser)
    return parser


def main() -> None:
    """Parse the command line and submit the job that cuts the checkpoint it names."""
    args = build_parser().parse_args()
    snapshot = os.path.abspath(args.snapshot or default_snapshot())
    out = os.path.abspath(args.out)
    slurm.submit(args, "lsa-cut", env.offline_hf(), slurm.CPU, cut, snapshot, out, args.layers, args.weights)
