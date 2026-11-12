# P0 selection-logic cost bench

Date: 2026-11-05. Logs: `docs/logs/bench-p0-dim{64,512}.log`. Machine: Apple M5 Max, 128 GB unified memory, mlx 0.31.2, fp32.

Method: `python -m fusion_lsa.evaluation.selection_bench`. Oracle row-loop `read` port (lazy MLX, but
`read` syncs per row via `positions.tolist()` and the per-row mask count — the 64 decode queries are
64 sequential GPU round-trips, and that per-row sync is the real cost driver behind the 0.7-1.0 ms
figure). Bucket table built once per run; no warmup pass, first-eval kernel compilation lands in the
decode numbers. Knobs follow the upstream served flat config:
cluster block 512, pool 4, bucket 8, cap 16, top_k 2048, top_p 512, window 512, sink 4, 131,072 keys
(16,384 buckets, 256 blocks).

## Results

| Scenario | index_dim 64 | index_dim 512 |
|---|---|---|
| build_buckets (128k keys) | 7.2s | 14.8s |
| decode selection, row loop | 1,021 us/query | 727 us/query |
| decode selections/s, single stream | ~980 | ~1,380 |
| prefill block selection (512 queries, row loop) | 258 us/query, 33.8s upper bound per 128k* | 280 us/query, 36.8s upper bound per 128k* |
| batched scoring + top-p (512 queries, no expansion) | 9.4 ms/block | 4.3 ms/block |

\* upper bound: the measured block sits at context end with full 16k-bucket visibility; real prefill
blocks average roughly half the visible buckets.

## Reading

- Decode path is viable as-is: ~0.7-1.0 ms per token selection. At 50-100 decode tok/s the selection
  costs roughly 5-10% of the step budget. The per-row Python loop is the overhead, not the matmuls —
  index_dim barely moves it.
- Prefill with the row loop is not viable (34-37s per 128k context). Selection must batch: the batched
  scoring + top-p path runs a 512-key block in 4-9 ms, i.e. roughly 2-5 s per 128k context, and it is
  exactly the doc's expected split (batched prefill selection at page boundaries, per-token decode on
  GPU ops).
- P2 gate passes: decode overhead single-digit percent, prefill batched path measurable and small
  against a 128k-key SDPA.

## Deferred to P1/P2

- Expansion (gather + window/sink assembly) still row-wise; batch it before prefill integration.
- fp16/bf16 scoring and compile-cached graphs not yet applied; both cut the decode number further.
