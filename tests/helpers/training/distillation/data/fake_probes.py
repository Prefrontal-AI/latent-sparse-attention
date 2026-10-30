"""A probe cache at test size - the key latents and the heads' queries, and a needle every probe is meant to find.

One needle of three keys sits far back, with its sentence around it.
Every probe query points at it, so the exact read under the identity map holds it, and a selector that groups keys badly
may not.
Rows are of three kinds - an answer row with a target, a question row, and a control with none.
"""

import torch
import torch.nn.functional as F

from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data.cache import layer_path, write
from tests.helpers.training.distillation.data import fake_cache

LENGTH, WINDOW, SINK = 4096, 64, 4
KEY_DIM, HEADS, TOP = 16, 4, 8
NEEDLE = [1000, 1001, 1002]
LINE = [997, 998, 999, 1000, 1001, 1002, 1003]
# An answer row, three question rows and a control, in position order
POSITIONS = [3000, 4090, 4091, 4092, 4093]
KINDS = [1, 0, 0, 0, 2]
TASKS = ("needle", "cwe")


def blob(seed=0, layer=0, name="needle0", length=LENGTH, rows=POSITIONS):
    """One probe (document, layer) as a blob, in the storage a cache extracted under `key_and_query` writes"""
    generator = torch.Generator().manual_seed(seed)
    key = F.normalize(torch.randn(length, KEY_DIM, generator=generator), dim=-1)
    direction = F.normalize(torch.randn(KEY_DIM, generator=generator), dim=0)
    key[NEEDLE] = direction
    count = len(rows)
    positions = torch.tensor(rows)
    heads = direction / HEADS + 0.02 * torch.randn(count, HEADS, KEY_DIM, generator=generator)
    readable = (torch.arange(length)[None] >= SINK) & (
        positions[:, None] - torch.arange(length)[None] >= WINDOW
    )
    scores = (F.normalize(heads.sum(1), dim=-1) @ key.t()).masked_fill(~readable, float("-inf"))
    best = scores.topk(TOP, dim=-1)
    target_key_ids = torch.full((count, len(NEEDLE)), -1, dtype=torch.int32)
    line_key_ids = torch.full((count, len(LINE)), -1, dtype=torch.int32)
    for row, kind in enumerate(KINDS):
        if kind != 2:
            target_key_ids[row] = torch.tensor(NEEDLE, dtype=torch.int32)
            line_key_ids[row] = torch.tensor(LINE, dtype=torch.int32)
    native = torch.full((count, 16), -1, dtype=torch.int32)
    native[:, : len(NEEDLE)] = torch.tensor(NEEDLE, dtype=torch.int32)
    return {
        "key": key.bfloat16(),
        "summed_query": heads.sum(1).float(),
        "head_queries": heads.bfloat16(),
        "query_positions": positions.int(),
        "top_key_ids": best.indices.int(),
        "top_key_mass": (HEADS * torch.softmax(best.values * 8, dim=-1)).half(),
        "peak_key_id": best.indices[:, :HEADS].int(),
        Cache.NATIVE: native,
        Cache.QUERY_KIND: torch.tensor(KINDS, dtype=torch.int8),
        Cache.TARGET_KEY_IDS: target_key_ids,
        Cache.LINE_KEY_IDS: line_key_ids,
        "meta": {
            "document": name,
            "layer": layer,
            "sink_size": SINK,
            "window_size": WINDOW,
            "cluster_block_size": WINDOW,
            "length": length,
            "train_queries": 0,
            "top": TOP,
            "model": fake_cache.MODEL,
            Cache.QUERY_KIND_NAMES: list(Cache.QUERY_KINDS),
        },
    }


KV_HEADS, GROUPED_HEADS, HEAD_DIM, ATTENTION_SCALE = 2, 4, 8, 4.0
# The needle's whole leaf block of 16 points the same way, since a block mean thins out a needle of three
NEEDLE_BLOCK = list(range(992, 1008))


def grouped_blob(seed=0, layer=0, name="needle0"):
    """One probe (document, layer) of a model with key-value heads, extracted under `queries.attention` and without a
    native selection - each key-value head's needle block points its own way, and its query heads point there too
    """
    generator = torch.Generator().manual_seed(seed + 1)
    grouped = {key: value for key, value in blob(seed, layer, name).items() if key != Cache.NATIVE}
    keys = F.normalize(torch.randn(LENGTH, KV_HEADS, HEAD_DIM, generator=generator), dim=-1)
    direction = F.normalize(torch.randn(KV_HEADS, HEAD_DIM, generator=generator), dim=-1)
    keys[NEEDLE_BLOCK] = direction
    queries = direction.repeat_interleave(GROUPED_HEADS // KV_HEADS, dim=0).expand(len(POSITIONS), -1, -1)
    grouped[Cache.ATTENTION_KEYS] = keys.bfloat16()
    grouped[Cache.ATTENTION_QUERIES] = queries.bfloat16().contiguous()
    grouped["meta"] = {**grouped["meta"], Cache.ATTENTION_SCALE: ATTENTION_SCALE}
    return grouped


def collection(root, documents=("needle0", "needle1", "cwe0", "cwe1"), layers=(0,), make=blob):
    """Write a probe collection on the eval side, with the manifest a run resolves its documents against"""
    for offset, name in enumerate(documents):
        for layer in layers:
            write(layer_path(root, "ruler", "eval", name, layer), make(offset + 17 * layer, layer, name))
    fake_cache.manifest(root, "ruler", {"eval": documents}, layers=layers, length=LENGTH)
    return root


def indexer(**overrides):
    """The run's indexer at the probe cache's rule, on the key and query"""
    return fake_cache.config(
        window_size=WINDOW, sink_size=SINK, index_dim=KEY_DIM, input="key_and_query", **overrides
    )
