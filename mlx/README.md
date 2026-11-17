# MLX / Apple Silicon Port

This directory contains an MLX port of the latent-sparse-attention selection
logic from this repository (the plain-torch reference), developed for the
Fusion monorepo — a local-first, fully offline Apple Silicon inference
stack. It keeps the algorithm and parity discipline of the torch reference
and replaces the tensor backend with MLX/Metal.

## What is ported

- `src/fusion_lsa/core` — selection logic: index space, Ward bucketing,
  top-p bucket scoring, top-k expansion beside sink tokens and the sliding
  window. Parity-tested against the torch reference as an oracle.
- `src/fusion_lsa/quant` — TurboQuant-style KV packing (k8v4: 8-bit keys /
  4-bit values, ~24.7 kB/token vs ~64 kB/token fp16).
- `src/fusion_lsa/budget` — memory budget helper for long-context serving.
- `src/fusion_lsa/index` — page-boundary incremental index.
- `src/fusion_lsa/evaluation` — selection-quality benchmark.
- `tests/` — 42 tests, including torch-vs-MLX parity suites
  (`test_reference_parity`, `test_ward_parity`, `test_tie_parity`,
  `test_index_space_parity`). Run standalone: `pytest tests/`.
- `docs/` — P0 parity report and selection benchmark notes.

The serving-engine integration (attention patch, TurboQuant coexistence,
prefix-cache restore, memory-budget gate) lives in the Fusion monorepo's
inference engine and is intentionally out of scope here.

## Results on Qwen3.8-27B (M5 Max 128 GB, MLX)

- RULER single-needle accuracy (needle at end of context): 4k / 32k / 128k
  all hit; 5/5 on completed runs.
- Packed KV footprint ~24.7 kB/token (2.6x below fp16).
- Same-session decode A/B at 32k: ~1.5 tok/s with LSA+k8v4 vs ~6.8 tok/s
  full attention — the port targets long-context feasibility (128k+), not
  short-context decode speed.
- 1M-token prefill: memory flat at the packed rate through 986k tokens.

See `docs/p0-report.md` for the parity methodology and numbers.

## Dependencies

`mlx>=0.29`, `numpy`; the test extra adds `pytest` and `torch` (the parity
oracle). `src/fusion_lsa/convert` additionally needs `safetensors`
(and `torch` for upstream-map conversion).
