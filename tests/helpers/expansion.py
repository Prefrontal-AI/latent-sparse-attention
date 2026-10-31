"""A small bucket cache with permuted pages, queries into it, and listings of it - the expansion tests' fixtures."""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from lsa.core import Buckets, build_buckets, read
from lsa.core.kernels.scan_select import dense_keys
from lsa.core.kernels.triton.keys import EMPTY

DEVICE = "cuda"
POOL, BUCKET, CAP, BLOCK, SINK, WINDOW = 4, 8, 16, 64, 4, 64
PER_BLOCK = BLOCK // BUCKET


class Cache(NamedTuple):
    # The reference's buckets, centroids rounded to bf16 as the scan reads them
    buckets: Buckets
    # Centroids by logical bucket [N, 64] bf16, for the scan
    centroids: torch.Tensor
    # Members and sizes by cache row, pages permuted
    bucket_members: torch.Tensor
    bucket_sizes: torch.Tensor
    # Three requests' pages [3, blocks] and each page's first position [blocks]
    pages: torch.Tensor
    block_positions: torch.Tensor
    key_count: int


def make_cache(blocks, seed, repeated=False):
    generator = torch.Generator().manual_seed(seed)
    key_count = blocks * BLOCK + 37
    keys = F.normalize(torch.randn(key_count, 64, generator=generator), dim=-1)
    if repeated:
        # A repeated text repeats its buckets, so scores tie and the bucket breaks them
        keys = keys[: 8 * BLOCK].repeat(blocks // 8 + 1, 1)[:key_count]
    found = build_buckets(keys, POOL, BUCKET, CAP, BLOCK)
    rounded = found.bucket_centroids.bfloat16()
    buckets = found._replace(bucket_centroids=rounded.float())
    page_of_block = torch.randperm(blocks, generator=generator)
    logical = torch.arange(blocks * PER_BLOCK)
    cache_row = page_of_block[logical // PER_BLOCK] * PER_BLOCK + logical % PER_BLOCK
    members = torch.empty_like(found.bucket_members)
    members[cache_row] = found.bucket_members
    sizes = torch.empty_like(found.bucket_sizes)
    sizes[cache_row] = found.bucket_sizes
    block_positions = torch.empty(blocks, dtype=torch.int32)
    block_positions[page_of_block] = torch.arange(blocks, dtype=torch.int32) * BLOCK
    return Cache(
        buckets,
        rounded.to(DEVICE),
        members.to(DEVICE),
        sizes.to(DEVICE),
        page_of_block.int()[None].expand(3, -1).contiguous().to(DEVICE),
        block_positions.to(DEVICE),
        key_count,
    )


def queries_at(cache, count, seed, norm=40.0, first=0):
    """Index queries [R, 64] bf16 and positions [R] int32, the last ones late where the window cuts into buckets"""
    generator = torch.Generator().manual_seed(seed)
    queries = (F.normalize(torch.randn(count, 64, generator=generator), dim=-1) * norm).bfloat16()
    positions = torch.randint(first, cache.key_count, (count,), generator=generator, dtype=torch.int32)
    if first == 0:
        positions[:3] = torch.tensor([5, BLOCK + 3, 2 * BLOCK + 40], dtype=torch.int32)
    positions[3:6] = cache.key_count - 1 - torch.arange(3, dtype=torch.int32)
    return queries.to(DEVICE), positions.to(DEVICE)


def physical(cache, chosen):
    """Cache rows of logical chosen [R, top_p] int32, -1 kept"""
    page = cache.pages[0].long()[(chosen.clamp_min(0) // PER_BLOCK).long()]
    return torch.where(chosen >= 0, page * PER_BLOCK + chosen.clamp_min(0) % PER_BLOCK, -1).int()


def visible(positions):
    return (positions // BLOCK * PER_BLOCK).int()


def listing_of(cache, queries, positions, top_p, seed):
    """A listing as the scan makes one - the best 3 * top_p buckets, in no order, packed, the empty key after"""
    keys = dense_keys(queries, cache.centroids, visible(positions))
    best = keys.topk(3 * top_p, dim=1).values
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    best = best.gather(1, torch.rand(best.shape, device=DEVICE, generator=generator).argsort(1))
    best = best.gather(1, (best == EMPTY).int().argsort(dim=1, stable=True))
    return best, (best != EMPTY).sum(1).int(), keys.topk(top_p, dim=1).values


def oracle(cache, queries, positions, top_p, top_k, early_budget=None):
    """The reference's read of the queries [R, top_k + WINDOW + SINK] int32, -1 padded, on the device"""
    selected, _ = read(
        F.normalize(queries.float().cpu(), dim=-1),
        positions.cpu(),
        cache.buckets,
        top_p=top_p,
        top_k=top_k,
        sink_size=SINK,
        window_size=WINDOW,
        early_budget=early_budget,
    )
    return selected.to(DEVICE)
