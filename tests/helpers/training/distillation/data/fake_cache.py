"""The x cache at test size, written to a tmp directory.

Structurally identical to the real one - `x` for every position, sinks, a window, training rows then a tail -
at shapes that project and score on a CPU in under a second.
The teacher ranks keys by a hidden low-rank map of `x`, so a pair of projections can learn what it likes.
"""

import json
from pathlib import Path

import torch

from lsa.sglang.settings.design import KEY_AND_QUERY, IndexerConfig
from lsa.training.data.targets import write_manifest
from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data.cache import layer_path, write
from lsa.training.distillation.data.native import LatentKeyAndQuery

LENGTH, WINDOW, SINK, BUDGET = 1024, 32, 4, 64
MODEL_DIM, INDEX_DIM, LATENT, HEADS, TOP = 24, 8, 4, 4, 16
TRAIN_QUERIES, TAIL_QUERIES = 8, 4


def config(**overrides):
    values = dict(window_size=WINDOW, sink_size=SINK, top_k=BUDGET, index_dim=INDEX_DIM)
    return IndexerConfig(**{**values, **overrides})


def document(seed=0, model_dim=MODEL_DIM, length=LENGTH):
    """One cached (document, layer) as a blob, with a teacher that likes the keys a hidden map of x lines up with."""
    generator = torch.Generator().manual_seed(seed)
    first = SINK + 2 * WINDOW
    positions = torch.cat(
        [
            torch.arange(first, length - TAIL_QUERIES)[
                torch.randperm(length - TAIL_QUERIES - first, generator=generator)[:TRAIN_QUERIES]
            ]
            .sort()
            .values,
            torch.arange(length - TAIL_QUERIES, length),
        ]
    )
    x = torch.randn(length, model_dim, generator=generator).bfloat16()
    hidden = torch.randn(model_dim, LATENT, generator=torch.Generator().manual_seed(1234))
    latent = torch.nn.functional.normalize(x.float() @ hidden, dim=-1)
    scores = latent[positions] @ latent.t()
    keys = torch.arange(length)
    readable = (keys[None] >= SINK) & (positions[:, None] - keys[None] >= WINDOW)
    best = scores.masked_fill(~readable, float("-inf")).topk(TOP, dim=-1)
    top_key_mass = HEADS * torch.softmax(best.values * 8, dim=-1)
    return {
        "x": x,
        "query_positions": positions.int(),
        "top_key_ids": best.indices.int(),
        "top_key_mass": top_key_mass.half(),
        "peak_key_id": best.indices[:, :HEADS].int(),
        "meta": {
            "document": "source0",
            "layer": 0,
            "sink_size": SINK,
            "window_size": WINDOW,
            "cluster_block_size": WINDOW,
            "length": length,
            "train_queries": TRAIN_QUERIES,
            "top": TOP,
            "model": MODEL,
        },
    }


# What a real extraction records about itself, which is what a later stage reads a map's teacher off
MODEL = "glm53_flash"


@torch.no_grad()
def keep_key_and_query(path, attention):
    """Replace a layer file's `x` with the key and query `attention` reads off it, as an extraction under that input writes."""
    blob = torch.load(path, mmap=True, weights_only=False)
    x = blob[Cache.X]
    positions = blob["query_positions"].long()
    kept = {name: value for name, value in blob.items() if name not in (Cache.X, Cache.META)}
    kept.update(LatentKeyAndQuery().fields(attention, x, positions))
    kept[Cache.META] = {**blob[Cache.META], Cache.INPUT: KEY_AND_QUERY}
    write(path, kept)
    return kept


def collection(
    root, name="tokens", documents=("source0", "source1"), layers=(0,), seed=0, split="train", model=MODEL
):
    """Write one side of a collection, with the manifest a run resolves its selection against."""
    for offset, document_name in enumerate(documents):
        for layer in layers:
            blob = document(seed + offset + 17 * layer)
            blob["meta"]["document"] = document_name
            blob["meta"]["layer"] = layer
            write(layer_path(root, name, split, document_name, layer), blob)
    manifest(root, name, {split: documents}, layers=layers, model=model)
    return root


def manifest(root, name, by_split, layers=(0,), model=MODEL, length=LENGTH):
    """Write or extend a collection's document index.

    `by_split` maps a side to the document names written into it, and names already recorded under another
    side are kept, so a caller can write the two sides in two calls.

    The extraction block - the model and the layers - is what the real manifest carries and what a saved map
    records as its teacher.
    """
    path = Path(root) / name / "manifest.json"
    entries = json.loads(path.read_text())["documents"] if path.exists() else []
    kept = [entry for entry in entries if entry["split"] not in by_split]
    for split, names in by_split.items():
        kept += [
            {
                "name": document_name,
                "source": document_name.rstrip("0123456789") or document_name,
                "document": f"doc-{document_name}",
                "window": 0,
                "split": split,
                "length": length,
            }
            for document_name in names
        ]
    payload = {
        "collection": name,
        "documents": kept,
        "run": {"model": {"name": model}},
        "layers": list(layers),
    }
    write_manifest(Path(root) / name, payload)
