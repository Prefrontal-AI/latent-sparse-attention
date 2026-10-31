"""Vendored CUDA sm_90a sparse attention kernels from upstream sparse-attention campaigns.

Source: upstream sparse_attn starts {fwd_r7,bwd_r1}.py
(upstream commit fabbc3ef), copied verbatim with a two-line header:

- fwd_r7.attention(q, k, v, selected, window, sink, scale) -> (o, lse): the forward rounds' round 7
  kernel, a planning kernel per 128-row query tile (the union of its rows' selected ids as a
  bitmap, dense 128-key blocks where the union covers 112 or more keys, a gather list for the
  rest) and a persistent warp-specialised kernel over the plan.
- bwd_r1.attention_backward(q, k, v, selected, o, lse, do, window, sink, scale) -> (dq, dk, dv):
  the backward's round 1 sealed kernel, key-major: an inverse of the selection by 128-key block,
  K and V of a block resident while the query tiles that read it stream past, dq by TMA fp32
  reduce-add. Its fast path needs d 64 or 128, a group of query heads dividing 64 and t up to
  51,200 (a t-by-t bitmap, t^2 / 8 bytes a key head); past it a one-thread-block-a-query
  fallback is orders of magnitude slower.

- fwd_hq16_hkv2.attention(q, k, v, selected, window, sink, scale) -> (o, lse): the
  final kernel of an upstream campaign, r7's planner and persistent kernel tuned for
  sixteen query heads over two key heads, verbatim
  but for its load_inline name. Of the eight campaign finals the one correct on LSA's shapes and
  faster than r7 there, at 4,096 tokens and under.

- fa3idx.Indexer()(q, k, top_k, window, sink, scale) -> (b, h, t, top_k) int32 ids: an
  FA3-style top-k indexer (fa3idx.py and its helpers common.py, here
  fa3idx_common.py), for ops/fa3_index. TMA streams key tiles into a ring, two wgmma warpgroups
  score them, and v8d's two-threshold selection runs on the accumulators. It took 16 query heads
  over 2 key heads and t < 65,536 only; the edits, each marked 'lsa:', let an even group pair
  heads as before and an odd group (one head, or DSA's per-head keys) give the two warpgroups two
  consecutive 64-row tiles of one head, bound t by t - window <= 65,536 (its ids are at most
  t - 1 - window), and give it this package's imports and load_inline name.

The later local campaigns' kernels for eight query heads over a 128-wide key head, which
ops/cuda_sparse reads by under v2 (sparse_read 'cuda_v2'), each file headed by its source path, md5,
author and date, and each load_inline name prefixed 'lsa_':

- fwd_hard.attention(q, k, v, selected, window, sink, scale) -> (o, lse): Claude Fable 5.1's
  final_hard.py, with its union pipeline
  fwd_hard_union (c13.py) and split pipeline fwd_hard_split (base_split.py, final_best2's). The
  union runs for any group: 16 queries of 8 heads an item over the 128-key tiles of their union,
  a plan kernel co-resident with the attention kernel and handing it items by ready flags. Its
  softmax keeps one reference max an item, tile 0's exact row max, shifted by 32 only when a
  row sum passes 2^20, and an exact fixup kernel recomputes any item whose row sum overflows
  (none for sane logits), so its rounding differs from r7's online softmax. The split needs b 1,
  G 8 and t of 65,536 or more, and runs when a statistics kernel finds the selection's query
  halves share under 22% of their keys; both pipelines launch and the one not chosen exits on
  the GPU. Edits, marked 'lsa:', load the two pipelines from this package.
- bwd_v14.attention_backward(q, k, v, selected, o, lse, do, window, sink, scale) -> (dq, dk, dv):
  Claude Opus 5.5's v14 (v14.py), key-major over 8-query tiles
  as r1, d 128 and G 8 only (else a one-block-a-query reference fallback); its query-tile bitmap
  is b * h_kv * t^2 / 8 bytes, 512 MiB at 65,536.
- bwd_hybrid.attention_backward(...) -> (dq, dk, dv): Claude Opus 5.5's dispatcher
  (final.py, frozen_v13) over v14 and a key-major hybrid
  (64-key tiles with per-query skipping for dk, dv and a query-major dq; bwd_hybrid_kernel, hyb.py,
  read as source and never imported, with bwd_hybrid_common, hyb_common.py). From 16,384 tokens a
  statistics kernel samples the selection and gates one of the two on the GPU; below, v14. It
  needs d 128, G 8 and t a multiple of 64. Edits, marked 'lsa:', read v14 and hyb.py from
  this package.

They take the operator above: selected (b, h_kv, t, top_k) int32, -1 padding, ids in any order,
and return or take lse as the natural log-sum-exp (b, h_q, t), as r7 and r1 do. fwd_hard and
both backwards keep static device buffers (a plan counter, a fixup flag list, cached tensor maps)
and so serve one stream at a time.

All build with torch.utils.cpp_extension.load_inline when imported (about a minute each, cached
under TORCH_EXTENSIONS_DIR), so ops/cuda_sparse imports them on first use only. Their operator is
gather_sparse.attention's: causal attention over the window, the sink and selected (b, h_kv, t,
top_k) int32 ids, -1 padding, every id in [sink, i - window] and none repeated. Their tensor maps
are cached on (pointer, shape), which stays right when a tensor is reallocated, since a map holds
only those.
"""
