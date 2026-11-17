import argparse
import logging
import time

import mlx.core as mx

from fusion_lsa.core.reference import build_buckets, read

logger = logging.getLogger(__name__)

# Defaults follow the upstream served flat config: block 512, pool 4, bucket 8, cap 16,
# top_k 2048, top_p 512, window 512, sink 4
DEFAULTS = dict(
    total=131072,
    index_dim=64,
    block_size=512,
    pool_size=4,
    bucket_size=8,
    bucket_cap=16,
    top_k=2048,
    top_p=512,
    window=512,
    sink=4,
    decode_queries=64,
)


def bench(total, index_dim, block_size, pool_size, bucket_size, bucket_cap, top_k, top_p, window, sink, decode_queries):
    logger.info(
        "bench start: %d keys, index_dim %d, top_p %d, top_k %d", total, index_dim, top_p, top_k
    )
    gen = mx.random.key(0)
    keys = mx.random.normal((total, index_dim), key=gen).astype(mx.float32)
    keys = keys / mx.linalg.norm(keys, axis=1, keepdims=True)
    t0 = time.perf_counter()
    buckets = build_buckets(keys, pool_size, bucket_size, bucket_cap, block_size)
    mx.eval(buckets.bucket_centroids, buckets.bucket_members, buckets.bucket_sizes)
    build_s = time.perf_counter() - t0
    bucket_count = buckets.bucket_centroids.shape[0]
    logger.info(
        "build_buckets: %.3fs (%d buckets, %d blocks)",
        build_s,
        bucket_count,
        bucket_count // (block_size // bucket_size),
    )

    queries = mx.random.normal((decode_queries, index_dim), key=gen).astype(mx.float32)
    queries = queries / mx.linalg.norm(queries, axis=1, keepdims=True)
    positions = mx.arange(decode_queries) + (total - decode_queries)

    # Warmup: compile the read graph and prime the allocator before timing
    warm_selected, warm_lengths = read(
        queries[:4], positions[:4], buckets, top_p, top_k, sink, window
    )
    mx.eval(warm_selected, warm_lengths)

    t0 = time.perf_counter()
    selected, lengths = read(queries, positions, buckets, top_p, top_k, sink, window)
    mx.eval(selected, lengths)
    decode_s = time.perf_counter() - t0
    per_query_us = decode_s / decode_queries * 1e6
    print(f"decode selection: {decode_s:.3f}s for {decode_queries} queries = {per_query_us:.0f} us/query")
    print(f"decode budget: {1e6 / per_query_us:.0f} selections/s single-stream equivalent")

    chunk = block_size
    chunk_queries = mx.random.normal((chunk, index_dim), key=gen).astype(mx.float32)
    chunk_queries = chunk_queries / mx.linalg.norm(chunk_queries, axis=1, keepdims=True)
    chunk_positions = mx.arange(chunk) + (total - decode_queries)
    t0 = time.perf_counter()
    selected_c, lengths_c = read(chunk_queries, chunk_positions, buckets, top_p, top_k, sink, window)
    mx.eval(selected_c, lengths_c)
    chunk_s = time.perf_counter() - t0
    print(
        f"prefill-block selection: {chunk_s:.3f}s for {chunk} queries in one {block_size}-key block "
        f"({chunk_s / chunk * 1e6:.0f} us/query)"
    )
    print(f"per-128k-context selection: {chunk_s * (total / block_size):.2f}s projected")

    t0 = time.perf_counter()
    scores = chunk_queries @ buckets.bucket_centroids.T
    picks = mx.argsort(-scores, axis=1)[:, :top_p]
    mx.eval(scores, picks)
    batched_s = time.perf_counter() - t0
    print(f"batched scoring+top-p (no expansion): {batched_s * 1000:.1f}ms for {chunk} queries")
    return {
        "build_s": build_s,
        "decode_s": decode_s,
        "per_query_us": per_query_us,
        "chunk_s": chunk_s,
        "batched_s": batched_s,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="fusion-lsa selection logic cost bench")
    parser.add_argument("--index-dim", type=int, default=DEFAULTS["index_dim"])
    parser.add_argument("--total", type=int, default=DEFAULTS["total"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--top-p", type=int, default=DEFAULTS["top_p"])
    parser.add_argument("--decode-queries", type=int, default=DEFAULTS["decode_queries"])
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(index_dim=args.index_dim, total=args.total, top_k=args.top_k, top_p=args.top_p, decode_queries=args.decode_queries)
    logger.info("config: %s", cfg)
    bench(**cfg)


if __name__ == "__main__":
    main()
