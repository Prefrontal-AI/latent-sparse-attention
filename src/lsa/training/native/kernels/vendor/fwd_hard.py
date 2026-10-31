# Vendored from kernels/final_hard.py (md5 7337dac5ae975ce32233bbfcfee85416), by a local Claude Fable 5.1 session, 2026-10-14;
# see __init__.py. Edits are marked 'lsa:', and load_inline's name "sa_hard_stats" is renamed.
"""Sparse-attention forward, hq16/hkv2-style shapes (G = 8 query heads per KV head), d128.

attention(q, k, v, selected, window, sink, scale) -> (o, lse).

Two pipelines, chosen per call on the device (no host synchronization, nothing cached between calls):

- union (UNION_KERNEL below): 16 queries x 8 heads per item, both consumer warpgroups compute every 128-key tile of
  the item's union. Against final_push's union kernel:
  * the softmax keeps no per-tile row max: tile 0 of an item sets a reference (its exact row max, plus a 2^-32
    headroom), later tiles exponentiate against it and only shift the reference by 32 (an exact power-of-two
    rescale of o and l) when a tile's row sum exceeds 2^20; a row sum that overflows to inf flags the item and an
    exact fixup kernel recomputes it (never taken for sane logits: it needs a score 160 log2 units above tile 0's);
  * the tile loop is unrolled by two with compile-time ring stages (wgmma descriptors hoisted);
  * the plan runs as a persistent 64-thread kernel co-resident with the attention CTA on every SM (the attention
    reserves 384 x 160 registers, the plan CTA 64 x 56) and publishes per-item ready flags; the attention kernel
    starts under programmatic dependent launch and waits per item, so the plan is hidden instead of serialized.
- split (base_split.py = final_best2's g_split2.py / v17, as in final_push): used from t = 32K when the selection's
  two query halves share less than SHARED_THR of their keys (random-like selections).
"""
import importlib.util
import os

import torch
from torch.utils.cpp_extension import load_inline

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = _HERE  # lsa: the vendored copies beside this file
UNION_KERNEL = os.environ.get("SA_HARD_UNION", "fwd_hard_union.py")  # lsa: c13.py  # c13 = c10 attention + plan v2; same-job benchmark against c10


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_KDIR, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_union = _load("sa_hard_union_kernel", UNION_KERNEL)
_split = _load("sa_hard_split_kernel", "fwd_hard_split.py")  # lsa: base_split.py

_STATS_SRC = r"""
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
__global__ void __launch_bounds__(1024) stats_k(const int* __restrict__ SEL, unsigned long long* __restrict__ acc,
    int* __restrict__ gate, int T, int HKV, int TOPK, int nqb, int nsamp, float thr) {
  extern __shared__ unsigned sm[];
  __shared__ bool last;
  const int s = blockIdx.x;
  const long totalItems = (long)nqb * HKV;
  const long it = (long)(((double)s + 0.5) * (double)totalItems / (double)nsamp);
  const int hk = (int)(it % HKV), qb = (int)(it / HKV);
  const int pf = qb * 16, NP = min(16, T - pf);
  const int NW = ((pf + NP - 1) >> 5) + 1;
  unsigned* UA = sm; unsigned* UB = sm + NW;
  for (int w = threadIdx.x; w < 2 * NW; w += blockDim.x) sm[w] = 0u;
  __syncthreads();
  const int* sb = SEL + ((size_t)hk * T + pf) * TOPK;
  if ((TOPK & 3) == 0) {
    for (int e4 = threadIdx.x; e4 < NP * TOPK / 4; e4 += blockDim.x) {
      const int p = (e4 * 4) / TOPK;
      const int4 v = __ldg(reinterpret_cast<const int4*>(sb) + e4);
      const int js[4] = {v.x, v.y, v.z, v.w};
      unsigned* arr = p < 8 ? UA : UB;
#pragma unroll
      for (int q = 0; q < 4; ++q) if ((unsigned)js[q] <= (unsigned)(pf + p)) atomicOr(arr + (js[q] >> 5), 1u << (js[q] & 31));
    }
  } else {
    for (int e = threadIdx.x; e < NP * TOPK; e += blockDim.x) {
      const int p = e / TOPK, j = __ldg(sb + e);
      if (j >= 0 && j <= pf + p) atomicOr((p < 8 ? UA : UB) + (j >> 5), 1u << (j & 31));
    }
  }
  __syncthreads();
  unsigned long long u = 0, ab = 0;
  for (int w = threadIdx.x; w < NW; w += blockDim.x) { u += __popc(UA[w] | UB[w]); ab += __popc(UA[w] & UB[w]); }
  for (int off = 16; off > 0; off >>= 1) { u += __shfl_xor_sync(0xffffffff, u, off); ab += __shfl_xor_sync(0xffffffff, ab, off); }
  if ((threadIdx.x & 31) == 0) { atomicAdd(acc, u); atomicAdd(acc + 1, ab); }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(acc + 2, 1ull) == (unsigned long long)(nsamp - 1);
  __syncthreads();
  if (last && threadIdx.x == 0) {
    __threadfence();
    const unsigned long long U = atomicAdd(acc, 0ull), AB = atomicAdd(acc + 1, 0ull);
    *gate = (U > 0 && (double)AB < (double)thr * (double)U) ? 1 : 0;
  }
}
}
torch::Tensor split_gate(torch::Tensor sel, int64_t nsamp, double thr) {
  const int B = sel.size(0), HKV = sel.size(1), T = sel.size(2), TOPK = sel.size(3);
  TORCH_CHECK(B == 1, "stats: batch 1 only");
  auto acc = torch::zeros({3}, sel.options().dtype(torch::kInt64));
  auto gate = torch::empty({1}, sel.options().dtype(torch::kInt32));
  const int nqb = (T + 15) / 16;
  const int NWmax = (T >> 5) + 2;
  const int smem = 2 * NWmax * 4;
  static int set = 0;
  if (smem > set) { cudaFuncSetAttribute(stats_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = smem; }
  stats_k<<<(int)nsamp, 1024, smem, at::cuda::getCurrentCUDAStream()>>>(sel.data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(acc.data_ptr()), gate.data_ptr<int>(), T, HKV, TOPK, nqb, (int)nsamp, (float)thr);
  return gate;
}
"""

_stats = load_inline(
    name="lsa_fwd_hard_stats",
    cpp_sources="torch::Tensor split_gate(torch::Tensor sel, int64_t nsamp, double thr);",
    cuda_sources=_STATS_SRC,
    functions=["split_gate"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
)

NSAMP = 48
SHARED_THR = 0.22       # split only when fewer than 22% of an item's union keys are shared by both halves (measured:
                        # the union now beats the split on every real selection and on random/rho0.9 at 32K; only
                        # random 64K, shared ~0.16-0.2, is still faster on the split)
SPLIT_MIN_T = 65536
last_choice = None


def attention(q, k, v, selected, window=0, sink=0, scale=None):
    """o, lse for one call. For t >= SPLIT_MIN_T both kernel pipelines are launched; a statistics kernel
    on this call's selection writes a device-side gate and the pipeline not chosen exits immediately
    (no host synchronization, nothing cached)."""
    global last_choice
    b, t, hq, d = q.shape
    hkv = k.shape[2]
    scale = d ** -0.5 if scale is None else scale
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    if selected.dtype != torch.int32:
        selected = selected.to(torch.int32)
    selected = selected.contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(b, hq, t, dtype=torch.float32, device=q.device)
    args = (q, k, v, selected, o, lse, int(window), int(sink), float(scale))
    if b == 1 and hq // hkv == 8 and d == 128 and t >= SPLIT_MIN_T and SHARED_THR > 0:
        gate = _stats.split_gate(selected, NSAMP, SHARED_THR)
        _union._ext.sparse_fwd(*args, _union.THR, _union.DFAC, gate, 0)
        _split._ext.sparse_fwd(*args, _split.THR, _split.DFAC, _split.SPLIT_THR, gate, 1)
        last_choice = gate
    else:
        _union._ext.sparse_fwd(*args, _union.THR, _union.DFAC, _union._nogate(q.device), 0)
        last_choice = None
    return o, lse


def debug_choice():
    """The last call's choice (reads the device gate; synchronizes - for reporting only)."""
    if last_choice is None:
        return "union"
    return "split" if int(last_choice.item()) == 1 else "union"


attention_sorted = attention
