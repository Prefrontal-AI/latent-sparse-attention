# fusion-lsa P0 report — algorithm port + parity + cost gate

Date: 2026-11-05. Answers the review checklist in
`~/fusion/architecture/fusion-lsa-mlx-migration-review-1105.md` item by item.

## What landed

- `src/fusion_lsa/core/ward.py` — Ward clustering port (`pool_keys`, `ward_rounds`, `ward_merge`,
  `members_from_labels`), pure MLX ops, parity-tested against the upstream torch oracle.
- `src/fusion_lsa/core/reference.py` — flat read path port (`build_buckets`, `read`,
  `expand_buckets`, input validation), parity-tested against the oracle.
- `src/fusion_lsa/budget.py` — early budget, pure Python.
- `tests/` — 22 tests: budget unit, ward parity, flat reference parity, 128k end-to-end pipeline.
- `src/fusion_lsa/evaluation/selection_bench.py` + `docs/bench-p0.md` + `docs/logs/` — cost gate.

## Review checklist status

### P0-1 tie-break / set equivalence — satisfied, argument below

Ward labels are compared exactly, not as sets: the label numbering ("order of first pool") is
deterministic given the merge sequence, and the merge sequence on random inputs has no cost ties
(measure zero). The tight-cap test pins the stop-short path. Centroid comparison allows 1e-3:
one-hot matmul vs `index_add_` differ only in fp32 summation order (~1.5e-4 observed).

Tie-break impact on SDPA: within `read`, ties can reorder only keys with equal score at the
top-p/top-k boundary. Attention is a weighted mean over the selected set — a permutation of the set
changes nothing (SDPA is permutation-invariant); a different tie member at the boundary changes one
key among 2048+, an O(1/2048) logit-mass perturbation bounded by the tie's own score gap (zero at
an exact tie in scores, hence equal attention weight). No downstream effect beyond fp32 noise.

### P0-2 Ward lazy-graph cost — measured; decode never clusters

The review's "per-token ward" framing does not match the design: ward/hierarchy runs at page
boundaries during prefill (the upstream growth protocol, `growth_at`); decode per token does
centroid scoring + top-k + gather only. Measured on M5 Max:

- bucket build over 128k keys (the full ward cost): 7.2s at index_dim 64, 14.8s at 512 — one-time
  per context, amortized over the whole decode.
- decode selection: 0.7-1.0 ms/query (row loop, fp32, no compile caching) — 5-10% of a step budget
  at 50-100 tok/s.
- prefill selection row-loop is not viable (34-37s/128k); batched scoring+top-p is 4-9ms per
  512-key block.

Known P1 item: the per-pair Python loop inside `ward_rounds` (correctness-first port) is the
build-cost driver; vectorizing merge pairs is straightforward once parity locks the semantics.
The stale-hierarchy fallback the review asks for already exists in the design — the hierarchy
only ever grows at page ends (`growth_at` replays it), so lagging updates are the default, not
a fallback.

### P0-3 page-boundary contract — accepted, scheduled for P1/P2

The plan already states "页增长协议只在页边界推进". The `index/` module (incremental bucket
growth with explicit page-boundary contract tests) is P1 scope; the adapter contract lands in P2.
Runtime parameter validation (prefill chunk vs cluster_block_size) rides with it. Not blocking P0.

### P0-4 early TQ quantization probe — accepted, scheduled for P1

Needs the released N-LSA 0.8B checkpoint running end-to-end (see "Deferred" below) plus
fusion-mlx `turboquant_kv.py` applied to latent KV. Queued as the first P1 task.

### P0-5 sink/window union — satisfied by construction, pinned by tests

`expand_buckets` returns `[long-range members (filtered to exclude the sink/window ranges), window,
sinks]` — the union is structural, not an optional merge. The filter `(members >= sink_size) &
(members < window_start)` guarantees no double count and no drop; window and sinks are always
appended from `arange`. Parity tests exercise sink=2/window=16 against the oracle, and the 128k
e2e asserts selected width `top_k + window + sink` with non-zero lengths. The P2 adapter must not
re-derive this — it consumes `read`'s output directly.

### P0-6 lsa-map metadata validation — accepted, P1 converter checklist

The P1 converter (`convert/`, .pt → safetensors) will write `model_arch`, `num_layers`,
`latent_dim` metadata and the loader will fail-loud on mismatch (plan's #1441 sticky-sentinel
lesson). Note the deliberate split: config/mismatch failures fail loud (P0-6); runtime numeric
anomalies degrade to full attention with a warning (review P1-3) — different failure classes.

## Deferred from P0

- Released N-LSA 0.8B checkpoint e2e (plan P0 step 3): running the 0.8B model in MLX requires
  porting the transformer architecture (upstream torch), which is model-port work outside the
  "algorithm port + parity" gate. The synthetic 128k e2e (`tests/test_e2e_128k.py`) covers the
  full selection pipeline end to end: build over 128k keys, needle retrieval, and attention-mass
  capture > 0.9 at 128k context. Checkpoint e2e is the first P1 task.

## Gate result

P0 passes: exact parity on partition/members/sizes/selection sets, centroid parity 1.5e-4,
decode selection ~1ms/token, batched prefill path viable. P1 proceeds.
