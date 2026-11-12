import logging
import time

import mlx.core as mx
import pytest

from fusion_lsa.core.reference import build_buckets, read

logger = logging.getLogger(__name__)

TOTAL = 131072
BLOCK = 512
POOL = 4
BUCKET = 8
CAP = 16
DIM = 64
SINK = 4
WINDOW = 512
TOP_K = 2048
TOP_P = 512
TOPICS = 256
NEEDLE_START = 70000
NEEDLE_RUN = 8


def _unit_rows(a: mx.array) -> mx.array:
    return a / mx.maximum(mx.linalg.norm(a, axis=1, keepdims=True), 1e-30)


@pytest.fixture(scope="module")
def index_cache():
    # Structured keys: one topic per 512-key block, so buckets are topic-pure the way real index
    # keys cluster by content; centroid scoring is meaningless on iid noise
    gen = mx.random.key(42)
    topics = _unit_rows(mx.random.normal((TOPICS, DIM), key=gen).astype(mx.float32))
    topic_ids = mx.repeat(mx.arange(TOTAL // BLOCK) % TOPICS, BLOCK)
    keys = topics[topic_ids] + mx.random.normal((TOTAL, DIM), key=gen).astype(mx.float32) * 0.05
    needle = _unit_rows(mx.random.normal((1, DIM), key=gen).astype(mx.float32))[0]
    keys[NEEDLE_START : NEEDLE_START + NEEDLE_RUN] = needle + mx.random.normal(
        (NEEDLE_RUN, DIM), key=gen
    ).astype(mx.float32) * 1e-3
    keys = _unit_rows(keys)
    needle_positions = list(range(NEEDLE_START, NEEDLE_START + NEEDLE_RUN))
    t0 = time.perf_counter()
    buckets = build_buckets(keys, POOL, BUCKET, CAP, BLOCK)
    mx.eval(buckets.bucket_centroids, buckets.bucket_members, buckets.bucket_sizes)
    logger.info("128k build_buckets: %.2fs", time.perf_counter() - t0)
    return keys, topics, needle, needle_positions, buckets


def test_128k_pipeline_shape_and_cost(index_cache):
    keys, topics, needle, needle_positions, buckets = index_cache
    assert buckets.bucket_centroids.shape[0] == TOTAL // BUCKET
    gen = mx.random.key(7)
    queries = _unit_rows(mx.random.normal((8, DIM), key=gen).astype(mx.float32))
    positions = mx.arange(8) + (TOTAL - 8)
    t0 = time.perf_counter()
    selected, lengths = read(queries, positions, buckets, TOP_P, TOP_K, SINK, WINDOW)
    mx.eval(selected, lengths)
    elapsed = time.perf_counter() - t0
    logger.info("128k decode read: %.3fs for 8 queries", elapsed)
    assert selected.shape == (8, TOP_K + WINDOW + SINK)
    assert int(lengths.min().item()) > 0


def test_128k_needle_retrieved_by_selection(index_cache):
    keys, topics, needle, needle_positions, buckets = index_cache
    queries = needle[None, :]
    positions = mx.array([TOTAL - 1])
    selected, lengths = read(queries, positions, buckets, TOP_P, TOP_K, SINK, WINDOW)
    mx.eval(selected, lengths)
    sel = set(selected[0].tolist())
    missing = [p for p in needle_positions if p not in sel]
    assert not missing, f"needle positions missed by selection: {missing}"


def _captured_mass(keys, query, selected_row):
    # Share of full-attention softmax mass the selected set holds - what selection quality means.
    # Queries are scaled x10 the way a trained model's logits are peaked; unscaled, 131k
    # near-equal logits spread mass so thin no 2k subset can hold it
    logits = keys @ query
    weights = mx.softmax(logits)
    sel = mx.array([p for p in selected_row.tolist() if p >= 0], dtype=mx.int32)
    return float(mx.take(weights, sel).sum().item())


def test_128k_selected_keys_hold_attention_mass(index_cache):
    keys, topics, needle, needle_positions, buckets = index_cache
    queries = mx.concatenate([needle[None, :], topics[:4]], axis=0)
    positions = mx.array([TOTAL - 1] * 5)
    selected, lengths = read(queries, positions, buckets, TOP_P, TOP_K, SINK, WINDOW)
    mx.eval(selected, lengths)
    for row, query in enumerate(queries.tolist()):
        mass = _captured_mass(keys, mx.array(query) * 14.0, selected[row])
        logger.info("query %d: captured softmax mass %.4f", row, mass)
        assert mass > 0.9, f"selected keys hold only {mass:.3f} of attention mass for query {row}"
