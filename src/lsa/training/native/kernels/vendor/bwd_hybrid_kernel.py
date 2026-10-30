# Vendored from frozen_v13/hyb.py (md5 b97d72251b399c2e1ec9fbc2d6006bd5), by a local Claude Opus 5.5 session, 2026-10-14;
# see __init__.py. Not edited: bwd_hybrid reads its KERN and HOST sources as text; this module is never imported.
"""Hybrid sparse-attention backward (H200, d128, G=8).

dK/dV: key-major over 64-key tiles, with PER-QUERY skipping. For every (kv head, 64-key tile) a list of the
queries that attend to at least one key of the tile (selected, window band or sinks) is built on the GPU; the
kernel packs 8 listed queries (x 8 heads = 64 rows) per MMA tile, gathering their Q/dO rows with one TMA per
query, and accumulates dK/dV for the tile in registers.
dQ: a separate query-major pass, one warpgroup per 8 consecutive queries over the union of their keys (gathered
64-key K/V tiles, cp.async), recomputing S and dP; no atomics.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline
from hyb_common import _gen, HEADER

KERN = r"""
#define QREC 4160
#define QSTR 5120
__device__ __forceinline__ void mbar_inval(unsigned addr) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ unsigned range_bits(int lo, int hi, int w) {
  const int a = max(lo - 32 * w, 0), b = min(hi - 32 * w, 31);
  if (a > b) return 0u;
  return (b - a == 31 ? 0xffffffffu : ((1u << (b - a + 1)) - 1u)) << a;
}
__device__ __forceinline__ void mbar_expect_tx_only(unsigned addr, unsigned bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async16(unsigned s, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_mbar_arrive(unsigned addr) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ unsigned long long dadd(unsigned long long d, unsigned off) { return d + (unsigned long long)(off >> 4); }
__device__ __forceinline__ void red_add_f2(float* p, float a, float b) { atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b)); }

template <int NW>
__device__ __forceinline__ int bscan(int v, int& total, int* red) {   // exclusive block scan, NW warps
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffff, x, o); if (lane >= o) x += y; }
  if (lane == 31) red[warp] = x;
  __syncthreads();
  int pre = 0, tot = 0;
#pragma unroll
  for (int k = 0; k < NW; ++k) { int r = red[k]; if (k < warp) pre += r; tot += r; }
  __syncthreads();
  total = tot;
  return pre + x - v;
}

// LD[(bh*T + pos)*8 + g] = (lse*log2e - log2(scale), rowsum(o*do)); zero dK/dV fp32 accumulators
__global__ void delta_k(const bf16* __restrict__ O, const bf16* __restrict__ DO, const float* __restrict__ LSE,
                        float2* __restrict__ LD, float* __restrict__ dk32, float* __restrict__ dv32,
                        long nkv8, int B, int T, int HQ, int HKV, float l2e, float lsh) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long row = idx / 16;
  const bool ok = row < (long)B * T * HQ;
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
  float s = 0.f;
  if (ok) {
    uint4 a = *reinterpret_cast<const uint4*>(O + idx * 8);
    uint4 c = *reinterpret_cast<const uint4*>(DO + idx * 8);
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pc = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float2 x = __bfloat1622float2(pa[e]), y = __bfloat1622float2(pc[e]);
      s += x.x * y.x + x.y * y.y;
    }
  }
  if (idx < nkv8) {
    float4* kz = reinterpret_cast<float4*>(dk32 + idx * 8);
    float4* vz = reinterpret_cast<float4*>(dv32 + idx * 8);
    kz[0] = z; kz[1] = z; vz[0] = z; vz[1] = z;
  }
  for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  if (ok && (idx % 16) == 0) {
    const int h = row % HQ;
    const int pos = (row / HQ) % T;
    const int b = row / ((long)HQ * T);
    const int G = HQ / HKV, hk = h / G, g = h % G;
    LD[((size_t)(b * HKV + hk) * T + pos) * G + g] = make_float2(LSE[((size_t)b * HQ + h) * T + pos] * l2e - lsh, s);
  }
}

// fused prep: LD, zero dK/dV accumulators, and QD[(bh*T + t)] = 4 KB smem image of query t's Q and dO rows
// (8 heads): [Q d0-63 | Q d64-127 | dO d0-63 | dO d64-127], each 8 rows x 128 B with the 128B swizzle.
__global__ void prep_k(const bf16* __restrict__ Q, const bf16* __restrict__ O, const bf16* __restrict__ DO,
                       const float* __restrict__ LSE, float2* __restrict__ LD, unsigned char* __restrict__ QD,
                       float* __restrict__ dk32, float* __restrict__ dv32, long nkv8, int B, int T, int HQ, int HKV,
                       float l2e, float lsh) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long row = idx / 16;
  const int cch = idx & 15;
  const bool ok = row < (long)B * T * HQ;
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
  float s = 0.f;
  const int h = row % HQ;
  const int pos = (row / HQ) % T;
  const int b = row / ((long)HQ * T);
  const int G = HQ / HKV, hk = h / G, g = h % G;
  if (ok) {
    uint4 a = *reinterpret_cast<const uint4*>(O + idx * 8);
    uint4 c = *reinterpret_cast<const uint4*>(DO + idx * 8);
    uint4 qv = *reinterpret_cast<const uint4*>(Q + idx * 8);
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pc = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float2 x = __bfloat1622float2(pa[e]), y = __bfloat1622float2(pc[e]);
      s += x.x * y.x + x.y * y.y;
    }
    unsigned char* dst = QD + ((size_t)(b * HKV + hk) * T + pos) * QREC + (cch >> 3) * 1024 + g * 128 + (((cch & 7) ^ g) << 4);
    *reinterpret_cast<uint4*>(dst) = qv;
    *reinterpret_cast<uint4*>(dst + 2048) = c;
  }
  if (idx < nkv8) {
    float4* kz = reinterpret_cast<float4*>(dk32 + idx * 8);
    float4* vz = reinterpret_cast<float4*>(dv32 + idx * 8);
    kz[0] = z; kz[1] = z; vz[0] = z; vz[1] = z;
  }
  for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  if (ok && cch == 0) {
    const float2 ldv = make_float2(LSE[((size_t)b * HQ + h) * T + pos] * l2e - lsh, s);
    LD[((size_t)(b * HKV + hk) * T + pos) * G + g] = ldv;
    *reinterpret_cast<float2*>(QD + ((size_t)(b * HKV + hk) * T + pos) * QREC + 4096 + 8 * g) = ldv;
  }
}

__global__ void cvt_k(const float* __restrict__ b, bf16* __restrict__ bo, const float* __restrict__ c, bf16* __restrict__ co,
                      long nb8, float bs, float cs) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  auto cv = [&](const float* src, bf16* dst, float sc) {
    const float4 x = reinterpret_cast<const float4*>(src + idx * 8)[0];
    const float4 y = reinterpret_cast<const float4*>(src + idx * 8)[1];
    uint4 r;
    r.x = pack_bf16(x.x * sc, x.y * sc); r.y = pack_bf16(x.z * sc, x.w * sc);
    r.z = pack_bf16(y.x * sc, y.y * sc); r.w = pack_bf16(y.z * sc, y.w * sc);
    *reinterpret_cast<uint4*>(dst + idx * 8) = r;
  };
  if (idx < nb8) { cv(b, bo, bs); cv(c, co, cs); }
}

// ---------------- index ----------------
// One CTA per (16-query chunk c, bh), one pass. Per-query key bitmaps (selected | window | sinks, causal) in smem.
// kv: chunk-local entries (tile-major, ascending query) LQ/LM at (bh*NC + c)*CAPC, CNTK[bh][tile][c] = count,
//     LPRE[(bh*NC + c)*(NT+1) + tile] = chunk-local exclusive prefix over tiles.
// dQ: union of each 8-query group, ascending keys QK + owner bytes QM at (bh*NG + g)*CAPQ, padded to 64; CNTQ.
__global__ void __launch_bounds__(512) idx_k(const int* __restrict__ sel, int* __restrict__ CNTK, int* __restrict__ LPRE,
    int* __restrict__ LQ, uint2* __restrict__ LM, int* __restrict__ CNTQ, int* __restrict__ QK, unsigned char* __restrict__ QM,
    int T, int TOPK, int NT, int NC, int NG, int CAPC, int CAPQ, int window, int sink, int SPLIT) {
  extern __shared__ unsigned bm[];
  __shared__ int red[16];
  const int c = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int i0 = c * 16;
  const int tl = (i0 + 15) >> 6;
  const int NWc = 2 * (tl + 1), NWp = NWc + 1;     // odd row stride: no bank conflicts across queries
  int* spre = reinterpret_cast<int*>(bm + 16 * NWp);
  for (int x = tid; x < 16 * NWp; x += 512) {
    const int p = x / NWp, w = x - p * NWp, i = i0 + p;
    unsigned v = 0u;
    if (window > 0) v |= range_bits(max(0, i - window + 1), i, w);
    if (sink > 0) v |= range_bits(0, min(sink, i + 1) - 1, w);
    bm[x] = v;
  }
  __syncthreads();
  const int* sb = sel + ((size_t)bh * T + i0) * TOPK;
  if ((TOPK & 3) == 0) {
    const int n4 = 4 * TOPK;
    for (int x = tid; x < n4; x += 512) {
      const int4 jv = __ldg(reinterpret_cast<const int4*>(sb) + x);
      const int p = (4 * x) / TOPK;
      const unsigned i = (unsigned)(i0 + p);
      unsigned* row = bm + p * NWp;
      if ((unsigned)jv.x <= i) atomicOr(row + (jv.x >> 5), 1u << (jv.x & 31));
      if ((unsigned)jv.y <= i) atomicOr(row + (jv.y >> 5), 1u << (jv.y & 31));
      if ((unsigned)jv.z <= i) atomicOr(row + (jv.z >> 5), 1u << (jv.z & 31));
      if ((unsigned)jv.w <= i) atomicOr(row + (jv.w >> 5), 1u << (jv.w & 31));
    }
  } else {
    for (int x = tid; x < 16 * TOPK; x += 512) {
      const int j = __ldg(sb + x);
      const int p = x / TOPK;
      if ((unsigned)j <= (unsigned)(i0 + p)) atomicOr(bm + p * NWp + (j >> 5), 1u << (j & 31));
    }
  }
  __syncthreads();
  int* lpre = LPRE + ((size_t)bh * NC + c) * (NT + 1);
  int carry = 0;
  for (int bs = 0; bs <= tl; bs += 512) {
    const int tt = bs + tid;
    int n = 0;
    if (tt <= tl) {
#pragma unroll
      for (int p = 0; p < 16; ++p) n += (bm[p * NWp + 2 * tt] | bm[p * NWp + 2 * tt + 1]) != 0u ? 1 : 0;
    }
    int tot;
    const int pos = carry + bscan<16>(n, tot, red);
    if (tt <= tl) {
      CNTK[((size_t)bh * NT + tt) * NC + c] = n;
      lpre[tt] = pos;
      spre[tt] = pos;
    }
    carry += tot;
  }
  if (tid == 0) lpre[tl + 1] = carry;
  __syncthreads();
  {
    // warp-cooperative entry writes: half-warp per tile, lane = query, so a store covers <= 2 adjacent runs
    int* lq = LQ + ((size_t)bh * NC + c) * CAPC;
    uint2* lm = LM + ((size_t)bh * NC + c) * CAPC;
    const int p = lane & 15, hf = lane >> 4;
    for (int pr = warp; 2 * pr <= tl; pr += 16) {
      const int tt = 2 * pr + hf;
      unsigned lo = 0u, hi = 0u;
      if (tt <= tl) { lo = bm[p * NWp + 2 * tt]; hi = bm[p * NWp + 2 * tt + 1]; }
      const bool f = (lo | hi) != 0u;
      const unsigned bal = __ballot_sync(0xffffffffu, f);
      if (f) {
        const unsigned mine = (bal >> (16 * hf)) & ((1u << p) - 1u);
        const int pos = spre[tt] + __popc(mine);
        lq[pos] = i0 + p;
        lm[pos] = make_uint2(lo, hi);
      }
    }
  }
  if (SPLIT) {
    // 16-query split list for the dQ pass: keys ordered [A-only | shared | B-only] (A = queries 0-7, B = 8-15),
    // ascending within each class; 16-bit owner masks; counts (nA, nS, nB) in CNTQ[3 * pair ..]; padded to 64.
    int* qk = QK + (size_t)(bh * NC + c) * CAPQ;
    unsigned short* qm = reinterpret_cast<unsigned short*>(QM) + (size_t)(bh * NC + c) * CAPQ;
    int carry[3] = {0, 0, 0};
    // pass 1: class totals (needed for the class offsets)
    int ta = 0, ts = 0, tb2 = 0;
    for (int w = tid; w < NWc; w += 512) {
      unsigned ua = 0u, ub = 0u;
#pragma unroll
      for (int p = 0; p < 8; ++p) { ua |= bm[p * NWp + w]; ub |= bm[(8 + p) * NWp + w]; }
      ta += __popc(ua & ~ub); ts += __popc(ua & ub); tb2 += __popc(ub & ~ua);
    }
    int nA, nS, nB;
    bscan<16>(ta, nA, red);
    bscan<16>(ts, nS, red);
    bscan<16>(tb2, nB, red);
    carry[1] = nA; carry[2] = nA + nS;
    for (int bs = 0; bs < NWc; bs += 512) {
      const int w = bs + tid;
      unsigned wp[16], ua = 0u, ub = 0u;
#pragma unroll
      for (int p = 0; p < 16; ++p) { wp[p] = w < NWc ? bm[p * NWp + w] : 0u; if (p < 8) ua |= wp[p]; else ub |= wp[p]; }
      const unsigned cls[3] = {ua & ~ub, ua & ub, ub & ~ua};
#pragma unroll
      for (int k3 = 0; k3 < 3; ++k3) {
        unsigned u = cls[k3];
        int tot;
        int k = carry[k3] + bscan<16>(__popc(u), tot, red);
        while (u) {
          const int bb = __ffs(u) - 1;
          u &= u - 1;
          unsigned own = 0u;
#pragma unroll
          for (int p = 0; p < 16; ++p) own |= ((wp[p] >> bb) & 1u) << p;
          qk[k] = 32 * w + bb;
          qm[k] = (unsigned short)own;
          ++k;
        }
        carry[k3] += tot;
      }
    }
    const int n = nA + nS + nB, padded = (n + 63) & ~63;
    for (int k = n + tid; k < padded; k += 512) { qk[k] = 0; qm[k] = 0; }
    if (tid == 0) { CNTQ[3 * (bh * NC + c)] = nA; CNTQ[3 * (bh * NC + c) + 1] = nS; CNTQ[3 * (bh * NC + c) + 2] = nB; }
  } else {
    // dQ unions of the two 8-query groups: packed (key << 8 | owner byte), padded to 64 with zeros
    for (int gq = 0; gq < 2; ++gq) {
      const unsigned* r = bm + 8 * gq * NWp;
      const int g = 2 * c + gq;
      int* qk = QK + (size_t)(bh * NG + g) * CAPQ;
      unsigned char* qm = QM + (size_t)(bh * NG + g) * CAPQ;
      int carry2 = 0;
      for (int bs = 0; bs < NWc; bs += 512) {
        const int w = bs + tid;
        unsigned wp[8], u = 0u;
  #pragma unroll
        for (int p = 0; p < 8; ++p) { wp[p] = w < NWc ? r[p * NWp + w] : 0u; u |= wp[p]; }
        int tot;
        int k = carry2 + bscan<16>(__popc(u), tot, red);
        while (u) {
          const int bb = __ffs(u) - 1;
          u &= u - 1;
          unsigned own = 0u;
  #pragma unroll
          for (int p = 0; p < 8; ++p) own |= ((wp[p] >> bb) & 1u) << p;
          qk[k] = 32 * w + bb;
          qm[k] = (unsigned char)own;
          ++k;
        }
        carry2 += tot;
      }
      const int padded = (carry2 + 63) & ~63;
      for (int k = carry2 + tid; k < padded; k += 512) { qk[k] = 0; qm[k] = 0; }
      if (tid == 0) CNTQ[bh * NG + g] = carry2;
    }
  }
}

// chunk-local kv entries -> per-tile lists (CNTK = exclusive offsets within tile lists after scan_k)
__global__ void __launch_bounds__(256) move_k(const int* __restrict__ CNTK, const int* __restrict__ LPRE, const int* __restrict__ LQ,
    const uint2* __restrict__ LM, const int* __restrict__ BASE, int* __restrict__ EQ, uint2* __restrict__ EM, int NT, int NC, int CAPC) {
  extern __shared__ int pre[];
  const int c = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  const int tl = (c * 16 + 15) >> 6;
  const int* lpre = LPRE + ((size_t)bh * NC + c) * (NT + 1);
  for (int x = tid; x <= tl + 1; x += 256) pre[x] = lpre[x];
  __syncthreads();
  const int n = pre[tl + 1];
  const int* lq = LQ + ((size_t)bh * NC + c) * CAPC;
  const uint2* lm = LM + ((size_t)bh * NC + c) * CAPC;
  for (int x = tid; x < n; x += 256) {
    int lo = 0, hi = tl;            // largest tt with pre[tt] <= x
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (pre[mid] <= x) lo = mid; else hi = mid - 1; }
    const int r = bh * NT + lo;
    const int dst = BASE[r] + CNTK[(size_t)r * NC + c] + (x - pre[lo]);
    EQ[dst] = lq[x];
    EM[dst] = lm[x];
  }
}

// per (tile, bh): exclusive scan of CNTK over chunks 4*tile .. NC-1, list length TOT
__global__ void __launch_bounds__(256) scan_k(int* __restrict__ CNTK, int* __restrict__ TOT, int NT, int NC) {
  __shared__ int red[8];
  const int tt = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  int* row = CNTK + ((size_t)bh * NT + tt) * NC;
  int carry = 0;
  for (int bs = 4 * tt; bs < NC; bs += 256) {
    const int x = bs + tid;
    const int v = x < NC ? row[x] : 0;
    int tot;
    const int pre = bscan<8>(v, tot, red);
    if (x < NC) row[x] = carry + pre;
    carry += tot;
  }
  if (tid == 0) TOT[bh * NT + tt] = carry;
}

// one CTA: list bases, dQ list offsets (padded to 64), band-major kv work items (r = bh*NT + tile, -, start, end)
__global__ void __launch_bounds__(1024) plan_k(const int* __restrict__ CNTK, const int* __restrict__ TOT, int* __restrict__ BASE,
    int4* __restrict__ ITEMS, int* __restrict__ NIT,
    int BH, int NT, int NC, int NG, int CPB, int nb) {
  extern __shared__ int sb[];
  __shared__ int red[32];
  const int tid = threadIdx.x, n = BH * NT;
  int carry = 0;
  for (int bs = 0; bs < n; bs += 1024) {
    const int i = bs + tid;
    const int v = i < n ? TOT[i] : 0;
    int tot;
    const int pre = bscan<32>(v, tot, red);
    if (i < n) { sb[i] = carry + pre; BASE[i] = carry + pre; }
    carry += tot;
  }
  __syncthreads();
  carry = 0;
  const int nseg = nb * n;
  for (int bs = 0; bs < nseg; bs += 1024) {
    const int x = bs + tid;
    int f = 0, a = 0, e = 0, r = 0;
    if (x < nseg) {
      const int be = x / n;
      r = x - be * n;
      const int tt = r % NT;
      const int lo = max(be * CPB, 4 * tt), c1 = (be + 1) * CPB;
      if (lo < c1 && lo < NC) {
        a = CNTK[(size_t)r * NC + lo];
        e = c1 >= NC ? TOT[r] : CNTK[(size_t)r * NC + c1];
        f = e > a;
      }
    }
    int tot;
    const int pre = bscan<32>(f, tot, red);
    if (f) ITEMS[carry + pre] = make_int4(r, 0, sb[r] + a, sb[r] + e);
    carry += tot;
  }
  if (tid == 0) { NIT[0] = carry; NIT[1] = 0; NIT[2] = 0; NIT[3] = 0; }
}

// ---------------- key-major dK/dV kernel ----------------
struct KCfg {
  static constexpr int NS = KV_NS;
  static constexpr int TB = 64 * 128 * 2;   // 16 KB: 64 keys (or 64 rows) x 128 d bf16
  static constexpr int PANEL = 64 * 128;    // 8 KB: 64 rows x 64 cols, 128B swizzle
  static constexpr int SB = 8 * QSTR;       // stage: 8 queries x [Qlo|Qhi|dOlo|dOhi|LD] at 5 KB stride
  static constexpr int OFF_K = 0, OFF_V = TB, OFF_Q = 2 * TB;
  static constexpr int OFF_MASK = OFF_Q + NS * SB;
  static constexpr int OFF_LD = OFF_MASK + NS * 64;
  static constexpr int SMEM = OFF_LD + NS * 512 + 1024;
};

template <int MODE>   // 0: full; 1: loads only (no GEMMs); 2: GEMMs only (no Q/dO loads)
__device__ __forceinline__ void kv_body(
    const CUtensorMap& tmK, const CUtensorMap& tmV,
    const unsigned char* __restrict__ QD,
    const int4* __restrict__ items, const int* __restrict__ NIT, const int* __restrict__ EQ, const uint2* __restrict__ EM,
    const float2* __restrict__ LD, float* __restrict__ dk32, float* __restrict__ dv32,
    int T, int HKV, int NT, float sl2, int vb, bool reinit) {
  using C = KCfg;
  constexpr int D = 128, NS = C::NS, NO = 64, PANEL = C::PANEL;
  extern __shared__ unsigned char smem_raw[];
  static __shared__ __align__(8) unsigned long long bars[1 + 2 * NS];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - base);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned kvfull = bar0;
  auto full = [&](int s) { return bar0 + 8 * (1 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (1 + NS + s); };
  const unsigned sK = sbase + C::OFF_K, sV = sbase + C::OFF_V;
  auto sQ = [&](int s) { return sbase + C::OFF_Q + s * C::SB; };
  unsigned char* sMask = gbase + C::OFF_MASK;
  const float* sLDg = reinterpret_cast<const float*>(gbase + C::OFF_LD);
  const unsigned sLD = sbase + C::OFF_LD;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    if (reinit) { mbar_inval(kvfull); for (int s = 0; s < NS; ++s) { mbar_inval(full(s)); mbar_inval(empty(s)); } }
    mbar_init(kvfull, 1);
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 32); mbar_init(empty(s), 128); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int4 itm = items[vb];
  const int r = itm.x, s0 = itm.z, s1 = itm.w;
  const int bh = r / NT, tt = r - bh * NT, b = bh / HKV, hk = bh % HKV, K0 = tt * 64;
  const int ng = (s1 - s0 + 7) >> 3;

  if (warp >= 8) {
    reg_dealloc<40>();
    if (warp == 8) {
      if (lane == 0) {
        mbar_expect_tx(kvfull, 2 * C::TB);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          tma_load_4d(sK + p * PANEL, &tmK, p * 64, hk, K0, b, kvfull);
          tma_load_4d(sV + p * PANEL, &tmV, p * 64, hk, K0, b, kvfull);
        }
      }
      int qn = 0;
      uint2 mn = make_uint2(0u, 0u);
      auto ld_ent = [&](int g, int& q, uint2& m) {
        if (lane < 8) {
          const int e = s0 + 8 * g + lane;
          const bool v = e < s1;
          q = __ldg(EQ + (v ? e : s1 - 1));
          m = v ? __ldg(EM + e) : make_uint2(0u, 0u);
        }
      };
      ld_ent(0, qn, mn);
      for (int g = 0; g < ng; ++g) {
        const int q = qn;
        const uint2 m = mn;
        if (g + 1 < ng) ld_ent(g + 1, qn, mn);
        const int s = g % NS;
        if (g >= NS) mbar_wait(empty(s), ((g / NS) - 1) & 1);
        if (MODE != 2 && lane == 0) mbar_expect_tx_only(full(s), 8 * QREC);
        __syncwarp();
        if (MODE != 2 && lane < 8) bulk_load(sQ(s) + lane * QSTR, QD + ((size_t)bh * T + q) * QREC, QREC, full(s));
        unsigned b0 = 0u, b1 = 0u;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const unsigned lo = __shfl_sync(0xffffffffu, m.x, n), hi = __shfl_sync(0xffffffffu, m.y, n);
          b0 |= ((lo >> lane) & 1u) << n;
          b1 |= ((hi >> lane) & 1u) << n;
        }
        sMask[s * 64 + lane] = (unsigned char)b0;
        sMask[s * 64 + 32 + lane] = (unsigned char)b1;
        mbar_arrive(full(s));
      }
    }
    return;
  }

  reg_alloc<232>();
  const int wq = warp & 3, qd = lane & 3, t128 = tid & 127;
  const int k0 = wq * 16 + (lane >> 2);
  auto consumer = [&](auto CWc) {
    constexpr int c = decltype(CWc)::value;
    float dk[NO], dv[NO];
#pragma unroll
    for (int i = 0; i < NO; ++i) { dk[i] = 0.f; dv[i] = 0.f; }
    constexpr int bar_mine = 2 + c, bar_other = 3 - c;
    bool first = (c == 0);
    const unsigned long long dK_S = make_desc(sK, 16, 1024), dV_S = make_desc(sV, 16, 1024);
    const unsigned long long dQ0 = make_desc(sQ(0), 16, QSTR), dO0 = make_desc(sQ(0) + 2048, 16, QSTR);
    const unsigned long long dQt0 = make_desc(sQ(0), 1024, QSTR), dOt0 = make_desc(sQ(0) + 2048, 1024, QSTR);
    constexpr unsigned QSTEP = C::SB >> 4;
    mbar_wait(kvfull, 0);
    const int R = (ng + 1) >> 1;
    for (int rr = 0; rr < R; ++rr) {
      const int g = 2 * rr + c;
      if (g >= ng) {   // dummy turn keeps the ping-pong barriers balanced
        if (!first) bar_sync(bar_mine, 256);
        first = false;
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        bar_sync(bar_mine, 256);
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        continue;
      }
      const int s = g % NS;
      mbar_wait(full(s), (g / NS) & 1);
      if (MODE == 1) {
        if (!first) bar_sync(bar_mine, 256);
        first = false;
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        bar_sync(bar_mine, 256);
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        mbar_arrive(empty(s));
        continue;
      }
      const unsigned m0 = sMask[s * 64 + k0], m1 = sMask[s * 64 + k0 + 8];
      const bool isfull = __all_sync(0xffffffff, (m0 & m1) == 0xffu);
      const unsigned long long dQs = dQ0 + s * QSTEP, dOs = dO0 + s * QSTEP;
      const unsigned long long dQts = dQt0 + s * QSTEP, dOts = dOt0 + s * QSTEP;
      float sacc[32], pacc[32];
      if (!first) bar_sync(bar_mine, 256);
      first = false;
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
        wg_ss_n64(sacc, dadd(dK_S, (kk >> 2) * PANEL + (kk & 3) * 32), dadd(dQs, (kk >> 2) * 1024 + (kk & 3) * 32), kk > 0 ? 1 : 0);
      wg_commit();
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
        wg_ss_n64(pacc, dadd(dV_S, (kk >> 2) * PANEL + (kk & 3) * 32), dadd(dOs, (kk >> 2) * 1024 + (kk & 3) * 32), kk > 0 ? 1 : 0);
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
      unsigned pa[4][4], dsa[4][4];
      const float* LDs = reinterpret_cast<const float*>(gbase + C::OFF_Q + s * C::SB + 4096);
      float pv[32];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float4 L = *reinterpret_cast<const float4*>(LDs + n * (QSTR / 4) + 4 * qd);
        float p0 = exp2f(fmaf(sacc[4 * n], sl2, -L.x));
        float p1 = exp2f(fmaf(sacc[4 * n + 1], sl2, -L.z));
        float p2 = exp2f(fmaf(sacc[4 * n + 2], sl2, -L.x));
        float p3 = exp2f(fmaf(sacc[4 * n + 3], sl2, -L.z));
        if (!isfull) {
          if (!((m0 >> n) & 1u)) { p0 = 0.f; p1 = 0.f; }
          if (!((m1 >> n) & 1u)) { p2 = 0.f; p3 = 0.f; }
        }
        pv[4 * n] = p0; pv[4 * n + 1] = p1; pv[4 * n + 2] = p2; pv[4 * n + 3] = p3;
        pa[n >> 1][(n & 1) * 2] = pack_bf16(p0, p1);
        pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
      }
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float4 L = *reinterpret_cast<const float4*>(LDs + n * (QSTR / 4) + 4 * qd);
        const float d0 = pv[4 * n] * (pacc[4 * n] - L.y), d1 = pv[4 * n + 1] * (pacc[4 * n + 1] - L.w);
        const float d2 = pv[4 * n + 2] * (pacc[4 * n + 2] - L.y), d3 = pv[4 * n + 3] * (pacc[4 * n + 3] - L.w);
        dsa[n >> 1][(n & 1) * 2] = pack_bf16(d0, d1);
        dsa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(d2, d3);
      }
      bar_sync(bar_mine, 256);
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg_rs_n128_t(dv, pa[kk], dadd(dOts, kk * 2 * QSTR), 1);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg_rs_n128_t(dk, dsa[kk], dadd(dQts, kk * 2 * QSTR), 1);
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { fence_ureg(pa[i][jj]); fence_ureg(dsa[i][jj]); }
#pragma unroll
      for (int i = 0; i < NO; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
      mbar_arrive(empty(s));
    }
    if (c == 0) bar_sync(bar_mine, 256);
    // combine the two warpgroups' partial dK/dV, one fp32 reduction per item
    bar_sync(1, 256);
    float4* red = reinterpret_cast<float4*>(gbase + C::OFF_Q);
    if (c == 1) {
#pragma unroll
      for (int j = 0; j < 16; ++j) red[j * 128 + t128] = make_float4(dk[4 * j], dk[4 * j + 1], dk[4 * j + 2], dk[4 * j + 3]);
#pragma unroll
      for (int j = 0; j < 16; ++j) red[(16 + j) * 128 + t128] = make_float4(dv[4 * j], dv[4 * j + 1], dv[4 * j + 2], dv[4 * j + 3]);
    }
    bar_sync(1, 256);
    if (c == 0) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float4 x = red[j * 128 + t128];
        dk[4 * j] += x.x; dk[4 * j + 1] += x.y; dk[4 * j + 2] += x.z; dk[4 * j + 3] += x.w;
        const float4 y = red[(16 + j) * 128 + t128];
        dv[4 * j] += y.x; dv[4 * j + 1] += y.y; dv[4 * j + 2] += y.z; dv[4 * j + 3] += y.w;
      }
      const long kvb = ((long)b * T * HKV + hk) * D;
      const int j0 = K0 + k0, j1 = j0 + 8;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const int col = 8 * n + 2 * qd;
        red_add_f2(dk32 + kvb + (long)j0 * HKV * D + col, dk[4 * n], dk[4 * n + 1]);
        red_add_f2(dv32 + kvb + (long)j0 * HKV * D + col, dv[4 * n], dv[4 * n + 1]);
        red_add_f2(dk32 + kvb + (long)j1 * HKV * D + col, dk[4 * n + 2], dk[4 * n + 3]);
        red_add_f2(dv32 + kvb + (long)j1 * HKV * D + col, dv[4 * n + 2], dv[4 * n + 3]);
      }
    }
  };
  if (warp < 4) consumer(std::integral_constant<int, 0>{});
  else consumer(std::integral_constant<int, 1>{});
}

// grid = min(items, #SMs); CTAs loop over items (atomic counter, hardware-like dynamic order). A gated-off launch
// is therefore only #SMs CTAs. Barriers are invalidated and re-initialised between items.
template <int MODE>
__global__ void __launch_bounds__(384, 1) kv_main(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV,
    const unsigned char* __restrict__ QD,
    const int4* __restrict__ items, const int* __restrict__ NIT, const int* __restrict__ EQ, const uint2* __restrict__ EM,
    const float2* __restrict__ LD, float* __restrict__ dk32, float* __restrict__ dv32,
    int T, int HKV, int NT, float sl2, int* __restrict__ ctr) {
  __shared__ int sVb;
  const int nit = __ldg(NIT);
  int vb = blockIdx.x;
  while (vb < nit) {
    kv_body<MODE>(tmK, tmV, QD, items, NIT, EQ, EM, LD, dk32, dv32, T, HKV, NT, sl2, vb, vb != (int)blockIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) sVb = (int)gridDim.x + atomicAdd(ctr, 1);
    __syncthreads();
    vb = sVb;
  }
}

// ---------------- persistent key-major dK/dV kernel ----------------
// grid = #SMs; items (band-major order) are fetched from an atomic counter by the producer warp and published to
// the consumers through smem + the kvfull barrier. Per item the producer first issues up to NS groups (into ring
// slots freed by the previous item), then waits until the consumers are done with the K/V region (kvempty: they
// use it for the dK/dV combine) and loads the new K/V tile. Stage/phase bookkeeping runs on a global group count.
__global__ void __launch_bounds__(384, 1) kvp_main(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV,
    const unsigned char* __restrict__ QD,
    const int4* __restrict__ items, const int* __restrict__ NIT, int* __restrict__ ctr, const int* __restrict__ EQ,
    const uint2* __restrict__ EM, float* __restrict__ dk32, float* __restrict__ dv32, int T, int HKV, int NT, float sl2) {
  using C = KCfg;
  constexpr int D = 128, NS = C::NS, NO = 64, PANEL = C::PANEL;
  extern __shared__ unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[2 + 2 * NS];
  __shared__ int sItem[2];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - base);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned kvfull = bar0, kvempty = bar0 + 8;
  auto full = [&](int s) { return bar0 + 8 * (2 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (2 + NS + s); };
  const unsigned sK = sbase + C::OFF_K, sV = sbase + C::OFF_V;
  auto sQ = [&](int s) { return sbase + C::OFF_Q + s * C::SB; };
  unsigned char* sMask = gbase + C::OFF_MASK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    mbar_init(kvfull, 1);
    mbar_init(kvempty, 256);
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 32); mbar_init(empty(s), 128); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int nit = __ldg(NIT);

  if (warp >= 8) {
    reg_dealloc<40>();
    if (warp == 8) {
      int G = 0;
      for (int li = 0;; ++li) {
        int it = 0;
        if (lane == 0) it = atomicAdd(ctr, 1);
        it = __shfl_sync(0xffffffffu, it, 0);
        const bool valid = it < nit;
        int s0 = 0, s1 = 0, bh = 0, tt = 0;
        if (valid) {
          const int4 itm = __ldg(items + it);
          s0 = itm.z; s1 = itm.w; bh = itm.x / NT; tt = itm.x - bh * NT;
        }
        const int ng = valid ? (s1 - s0 + 7) >> 3 : 0;
        const int b = bh / HKV, hk = bh % HKV;
        int qn = 0;
        uint2 mn = make_uint2(0u, 0u);
        auto ld_ent = [&](int g) {
          if (lane < 8 && g < ng) {
            const int e = s0 + 8 * g + lane;
            const bool v = e < s1;
            qn = __ldg(EQ + (v ? e : s1 - 1));
            mn = v ? __ldg(EM + e) : make_uint2(0u, 0u);
          }
        };
        ld_ent(0);
        auto issue = [&](int g) {
          const int q = qn;
          const uint2 m = mn;
          ld_ent(g + 1);
          const int GG = G + g, s = GG % NS;
          if (GG >= NS) mbar_wait(empty(s), ((GG / NS) - 1) & 1);
          if (lane == 0) mbar_expect_tx_only(full(s), 8 * QREC);
          __syncwarp();
          if (lane < 8) bulk_load(sQ(s) + lane * QSTR, QD + ((size_t)bh * T + q) * QREC, QREC, full(s));
          unsigned b0 = 0u, b1 = 0u;
#pragma unroll
          for (int n = 0; n < 8; ++n) {
            const unsigned lo = __shfl_sync(0xffffffffu, m.x, n), hi = __shfl_sync(0xffffffffu, m.y, n);
            b0 |= ((lo >> lane) & 1u) << n;
            b1 |= ((hi >> lane) & 1u) << n;
          }
          sMask[s * 64 + lane] = (unsigned char)b0;
          sMask[s * 64 + 32 + lane] = (unsigned char)b1;
          mbar_arrive(full(s));
        };
        const int pre = min(ng, NS);
        for (int g = 0; g < pre; ++g) issue(g);
        if (lane == 0) {
          if (li > 0) mbar_wait(kvempty, (li - 1) & 1);
          sItem[li & 1] = valid ? it : -1;
          if (valid) {
            mbar_expect_tx(kvfull, 2 * C::TB);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              tma_load_4d(sK + p * PANEL, &tmK, p * 64, hk, tt * 64, b, kvfull);
              tma_load_4d(sV + p * PANEL, &tmV, p * 64, hk, tt * 64, b, kvfull);
            }
          } else {
            mbar_arrive(kvfull);
          }
        }
        __syncwarp();
        if (!valid) break;
        for (int g = pre; g < ng; ++g) issue(g);
        G += ng;
      }
    }
    return;
  }

  reg_alloc<232>();
  const int wq = warp & 3, qd = lane & 3, t128 = tid & 127;
  const int k0 = wq * 16 + (lane >> 2);
  auto consumer = [&](auto CWc) {
    constexpr int c = decltype(CWc)::value;
    constexpr int bar_mine = 2 + c, bar_other = 3 - c;
    const unsigned long long dK_S = make_desc(sK, 16, 1024), dV_S = make_desc(sV, 16, 1024);
    const unsigned long long dQ0 = make_desc(sQ(0), 16, QSTR), dO0 = make_desc(sQ(0) + 2048, 16, QSTR);
    const unsigned long long dQt0 = make_desc(sQ(0), 1024, QSTR), dOt0 = make_desc(sQ(0) + 2048, 1024, QSTR);
    constexpr unsigned QSTEP = C::SB >> 4;
    int G0 = 0;
    for (int li = 0;; ++li) {
      mbar_wait(kvfull, li & 1);
      const int it = *reinterpret_cast<volatile int*>(&sItem[li & 1]);
      if (it < 0) break;
      const int4 itm = __ldg(items + it);
      const int bh = itm.x / NT, tt = itm.x - (itm.x / NT) * NT, b = bh / HKV, hk = bh % HKV, K0 = tt * 64;
      const int ng = (itm.w - itm.z + 7) >> 3;
      float dk[NO], dv[NO];
#pragma unroll
      for (int i = 0; i < NO; ++i) { dk[i] = 0.f; dv[i] = 0.f; }
      bool first = (c == 0);
      const int R = (ng + 1) >> 1;
      for (int rr = 0; rr < R; ++rr) {
        const int g = 2 * rr + c;
        if (g >= ng) {
          if (!first) bar_sync(bar_mine, 256);
          first = false;
          asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
          bar_sync(bar_mine, 256);
          asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
          continue;
        }
        const int GG = G0 + g, s = GG % NS;
        mbar_wait(full(s), (GG / NS) & 1);
        const unsigned m0 = sMask[s * 64 + k0], m1 = sMask[s * 64 + k0 + 8];
        const bool isfull = __all_sync(0xffffffff, (m0 & m1) == 0xffu);
        const unsigned long long dQs = dQ0 + s * QSTEP, dOs = dO0 + s * QSTEP;
        const unsigned long long dQts = dQt0 + s * QSTEP, dOts = dOt0 + s * QSTEP;
        float sacc[32], pacc[32];
        if (!first) bar_sync(bar_mine, 256);
        first = false;
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          wg_ss_n64(sacc, dadd(dK_S, (kk >> 2) * PANEL + (kk & 3) * 32), dadd(dQs, (kk >> 2) * 1024 + (kk & 3) * 32), kk > 0 ? 1 : 0);
        wg_commit();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          wg_ss_n64(pacc, dadd(dV_S, (kk >> 2) * PANEL + (kk & 3) * 32), dadd(dOs, (kk >> 2) * 1024 + (kk & 3) * 32), kk > 0 ? 1 : 0);
        wg_commit();
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        wg_wait<1>();
#pragma unroll
        for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
        unsigned pa[4][4], dsa[4][4];
        const float* LDs = reinterpret_cast<const float*>(gbase + C::OFF_Q + s * C::SB + 4096);
        float pv[32];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const float4 L = *reinterpret_cast<const float4*>(LDs + n * (QSTR / 4) + 4 * qd);
          float p0 = exp2f(fmaf(sacc[4 * n], sl2, -L.x));
          float p1 = exp2f(fmaf(sacc[4 * n + 1], sl2, -L.z));
          float p2 = exp2f(fmaf(sacc[4 * n + 2], sl2, -L.x));
          float p3 = exp2f(fmaf(sacc[4 * n + 3], sl2, -L.z));
          if (!isfull) {
            if (!((m0 >> n) & 1u)) { p0 = 0.f; p1 = 0.f; }
            if (!((m1 >> n) & 1u)) { p2 = 0.f; p3 = 0.f; }
          }
          pv[4 * n] = p0; pv[4 * n + 1] = p1; pv[4 * n + 2] = p2; pv[4 * n + 3] = p3;
          pa[n >> 1][(n & 1) * 2] = pack_bf16(p0, p1);
          pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
        wg_wait<0>();
#pragma unroll
        for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const float4 L = *reinterpret_cast<const float4*>(LDs + n * (QSTR / 4) + 4 * qd);
          const float d0 = pv[4 * n] * (pacc[4 * n] - L.y), d1 = pv[4 * n + 1] * (pacc[4 * n + 1] - L.w);
          const float d2 = pv[4 * n + 2] * (pacc[4 * n + 2] - L.y), d3 = pv[4 * n + 3] * (pacc[4 * n + 3] - L.w);
          dsa[n >> 1][(n & 1) * 2] = pack_bf16(d0, d1);
          dsa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(d2, d3);
        }
        bar_sync(bar_mine, 256);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) wg_rs_n128_t(dv, pa[kk], dadd(dOts, kk * 2 * QSTR), 1);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) wg_rs_n128_t(dk, dsa[kk], dadd(dQts, kk * 2 * QSTR), 1);
        wg_commit();
        asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
        wg_wait<0>();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) { fence_ureg(pa[i][jj]); fence_ureg(dsa[i][jj]); }
#pragma unroll
        for (int i = 0; i < NO; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
        mbar_arrive(empty(s));
      }
      if (c == 0) bar_sync(bar_mine, 256);
      // combine the two warpgroups' dK/dV through the (now idle) 32 KB K/V region, dK then dV
      float4* red = reinterpret_cast<float4*>(gbase + C::OFF_K);
      const long kvb = ((long)b * T * HKV + hk) * D;
      const int j0 = K0 + k0, j1 = j0 + 8;
      bar_sync(1, 256);
      if (c == 1) {
#pragma unroll
        for (int j = 0; j < 16; ++j) red[j * 128 + t128] = make_float4(dk[4 * j], dk[4 * j + 1], dk[4 * j + 2], dk[4 * j + 3]);
      }
      bar_sync(1, 256);
      if (c == 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float4 x = red[j * 128 + t128];
          dk[4 * j] += x.x; dk[4 * j + 1] += x.y; dk[4 * j + 2] += x.z; dk[4 * j + 3] += x.w;
        }
      }
      bar_sync(1, 256);
      if (c == 1) {
#pragma unroll
        for (int j = 0; j < 16; ++j) red[j * 128 + t128] = make_float4(dv[4 * j], dv[4 * j + 1], dv[4 * j + 2], dv[4 * j + 3]);
      }
      bar_sync(1, 256);
      if (c == 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float4 y = red[j * 128 + t128];
          dv[4 * j] += y.x; dv[4 * j + 1] += y.y; dv[4 * j + 2] += y.z; dv[4 * j + 3] += y.w;
        }
      }
      mbar_arrive(kvempty);
      if (c == 0) {
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          const int col = 8 * n + 2 * qd;
          red_add_f2(dk32 + kvb + (long)j0 * HKV * D + col, dk[4 * n], dk[4 * n + 1]);
          red_add_f2(dv32 + kvb + (long)j0 * HKV * D + col, dv[4 * n], dv[4 * n + 1]);
          red_add_f2(dk32 + kvb + (long)j1 * HKV * D + col, dk[4 * n + 2], dk[4 * n + 3]);
          red_add_f2(dv32 + kvb + (long)j1 * HKV * D + col, dv[4 * n + 2], dv[4 * n + 3]);
        }
      }
      G0 += ng;
    }
  };
  if (warp < 4) consumer(std::integral_constant<int, 0>{});
  else consumer(std::integral_constant<int, 1>{});
}

// ---------------- query-major dQ kernel, version 2 ----------------
// CTA = one 8-query group (64 rows) of one kv head; the two consumer warpgroups take alternate gathered 64-key
// tiles of the group's union (own dQ accumulators, summed at the end); one producer warpgroup gathers K/V with
// cp.async into an NS2-stage ring.
struct Q2Cfg {
  static constexpr int NS = DQ2_NS;
  static constexpr int TB = 64 * 128 * 2, PANEL = 64 * 128;
  static constexpr int OFF_Q = 0, OFF_K = 8 * QSTR;        // Q/dO/LD records: 8 queries x 5 KB
  static constexpr int OFF_V = OFF_K + NS * TB;
  static constexpr int OFF_OWN = OFF_V + NS * TB;     // NS x 64 packed (key << 8 | owner) ints
  static constexpr int OFF_LD = OFF_OWN + NS * 64;
  static constexpr int SMEM = OFF_LD + 512 + 1024;
};

__global__ void __launch_bounds__(384, 1) dq2_main(
    const unsigned char* __restrict__ QD,
    const bf16* __restrict__ Kg, const bf16* __restrict__ Vg, int CAPQ, const int* __restrict__ CNTQ,
    const int* __restrict__ QK, const unsigned char* __restrict__ QM, bf16* __restrict__ dq, int T, int HKV, int NG, float sl2) {
  using C = Q2Cfg;
  constexpr int NS = C::NS, PANEL = C::PANEL, D = 128;
  extern __shared__ unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[1 + 2 * NS];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - base);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned qfull = bar0;
  auto full = [&](int s) { return bar0 + 8 * (1 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (1 + NS + s); };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int BH = gridDim.x / NG;
  const int bh = blockIdx.x % BH, g = NG - 1 - (int)(blockIdx.x / BH);
  const int b = bh / HKV, hk = bh % HKV, i0 = 8 * g;
  const unsigned sQ = sbase + C::OFF_Q, sDO = sQ + 2048;
  auto sK = [&](int s) { return sbase + C::OFF_K + s * C::TB; };
  auto sV = [&](int s) { return sbase + C::OFF_V + s * C::TB; };
  if (tid == 0) {
    mbar_init(qfull, 1);
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 128); mbar_init(empty(s), 128); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int cnt = __ldg(CNTQ + bh * NG + g), ntl = (cnt + 63) >> 6;
  const size_t off = (size_t)(bh * NG + g) * CAPQ;
  if (warp >= 8) {
    reg_dealloc<40>();
    const int pt = tid - 256;
    if (pt == 0) {
      mbar_expect_tx(qfull, 8 * QREC);
#pragma unroll
      for (int n = 0; n < 8; ++n) bulk_load(sQ + n * QSTR, QD + ((size_t)bh * T + i0 + n) * QREC, QREC, qfull);
    }
    const int cc = pt & 15, rb = pt >> 4;
    const size_t rs = (size_t)HKV * D;
    const bf16* Kb = Kg + ((size_t)b * T * HKV + hk) * D + cc * 8;
    const bf16* Vb = Vg + ((size_t)b * T * HKV + hk) * D + cc * 8;
    const unsigned coff = (cc >> 3) * PANEL + ((cc & 7) << 4);
    int jn[8];
    if (ntl > 0) {
#pragma unroll
      for (int k = 0; k < 8; ++k) jn[k] = __ldg(QK + off + rb + 8 * k);
    }
    for (int tl = 0; tl < ntl; ++tl) {
      int jc[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) jc[k] = jn[k];
      if (tl + 1 < ntl) {
#pragma unroll
        for (int k = 0; k < 8; ++k) jn[k] = __ldg(QK + off + (tl + 1) * 64 + rb + 8 * k);
      }
      const int s = tl % NS;
      if (tl >= NS) mbar_wait(empty(s), ((tl / NS) - 1) & 1);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int row = rb + 8 * k;
        const unsigned sw = (coff + row * 128) ^ ((row & 7) << 4);
        cp_async16(sK(s) + sw, Kb + (size_t)jc[k] * rs);
        cp_async16(sV(s) + sw, Vb + (size_t)jc[k] * rs);
      }
      if (pt < 4) cp_async16(sbase + C::OFF_OWN + s * 64 + pt * 16, QM + off + tl * 64 + pt * 16);
      cp_mbar_arrive(full(s));
    }
    cp_wait_all();
    return;
  }

  reg_alloc<232>();
  const int c = warp >> 2, wq = warp & 3, qd = lane & 3;
  const int r0 = wq * 16 + (lane >> 2);
  mbar_wait(qfull, 0);
  const float2* sLDf = reinterpret_cast<const float2*>(gbase + C::OFF_Q + 4096);
  const float2 ld0 = sLDf[(2 * wq) * (QSTR / 8) + (lane >> 2)], ld1 = sLDf[(2 * wq + 1) * (QSTR / 8) + (lane >> 2)];
  const unsigned char* sOwn = gbase + C::OFF_OWN;
  const unsigned long long dQd = make_desc(sQ, 16, QSTR), dOd = make_desc(sDO, 16, QSTR);
  const unsigned long long dK0 = make_desc(sK(0), 16, 1024), dV0 = make_desc(sV(0), 16, 1024);
  const unsigned long long dKt0 = make_desc(sK(0), PANEL, 1024);
  constexpr unsigned KSTEP = C::TB >> 4;
  const int sh0 = 2 * wq, sh1 = 2 * wq + 1;
  float dqa[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) dqa[i] = 0.f;
  for (int tl = c; tl < ntl; tl += 2) {
    const int s = tl % NS;
    mbar_wait(full(s), (tl / NS) & 1);
    float sacc[32], pacc[32];
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
      wg_ss_n64(sacc, dadd(dQd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dK0 + s * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
    wg_commit();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
      wg_ss_n64(pacc, dadd(dOd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dV0 + s * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
    wg_commit();
    unsigned vm0 = 0u, vm1 = 0u;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const unsigned short o2 = *reinterpret_cast<const unsigned short*>(gbase + C::OFF_OWN + s * 64 + 8 * n + 2 * qd);
      vm0 |= (((o2 >> sh0) & 1u) | (((o2 >> (8 + sh0)) & 1u) << 1)) << (2 * n);
      vm1 |= (((o2 >> sh1) & 1u) | (((o2 >> (8 + sh1)) & 1u) << 1)) << (2 * n);
    }
    wg_wait<1>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sacc[4 * n] = ((vm0 >> (2 * n)) & 1u) ? exp2f(fmaf(sacc[4 * n], sl2, -ld0.x)) : 0.f;
      sacc[4 * n + 1] = ((vm0 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(sacc[4 * n + 1], sl2, -ld0.x)) : 0.f;
      sacc[4 * n + 2] = ((vm1 >> (2 * n)) & 1u) ? exp2f(fmaf(sacc[4 * n + 2], sl2, -ld1.x)) : 0.f;
      sacc[4 * n + 3] = ((vm1 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(sacc[4 * n + 3], sl2, -ld1.x)) : 0.f;
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
    unsigned fa[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      fa[n >> 1][(n & 1) * 2] = pack_bf16(sacc[4 * n] * (pacc[4 * n] - ld0.y), sacc[4 * n + 1] * (pacc[4 * n + 1] - ld0.y));
      fa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(sacc[4 * n + 2] * (pacc[4 * n + 2] - ld1.y), sacc[4 * n + 3] * (pacc[4 * n + 3] - ld1.y));
    }
    wg_fence();
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) wg_rs_n128_t(dqa, fa[k2], dadd(dKt0 + s * KSTEP, k2 * 2048), 1);
    wg_commit();
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 64; ++i) fence_reg(dqa[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) fence_ureg(fa[i][jj]);
    mbar_arrive(empty(s));
  }
  bar_sync(1, 256);
  float4* red = reinterpret_cast<float4*>(gbase + C::OFF_K);
  const int t128 = tid & 127;
  if (c == 1) {
#pragma unroll
    for (int j = 0; j < 16; ++j) red[j * 128 + t128] = make_float4(dqa[4 * j], dqa[4 * j + 1], dqa[4 * j + 2], dqa[4 * j + 3]);
  }
  bar_sync(1, 256);
  if (c == 0) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float4 x = red[j * 128 + t128];
      dqa[4 * j] += x.x; dqa[4 * j + 1] += x.y; dqa[4 * j + 2] += x.z; dqa[4 * j + 3] += x.w;
    }
    const int HQ = HKV * 8;
    bf16* o0 = dq + (((size_t)b * T + i0 + 2 * wq) * HQ + hk * 8 + (lane >> 2)) * D + 2 * qd;
    bf16* o1 = o0 + (size_t)HQ * D;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      *reinterpret_cast<unsigned*>(o0 + 8 * n) = pack_bf16(dqa[4 * n], dqa[4 * n + 1]);
      *reinterpret_cast<unsigned*>(o1 + 8 * n) = pack_bf16(dqa[4 * n + 2], dqa[4 * n + 3]);
    }
  }

}

// ---------------- query-major dQ kernel, version 3 (16-query split) ----------------
// CTA = 16 consecutive queries (groups A = 0-7, B = 8-15) of one kv head. Its key list is [A-only | shared | B-only];
// each gathered 64-key K/V tile is loaded once; warpgroup c computes only tiles holding keys of its group
// (A-only and B-only tiles are interleaved so both warpgroups stay busy, shared tiles last). No cross-WG reduction.
struct Q3Cfg {
  static constexpr int NS = DQ3_NS;
  static constexpr int TB = 64 * 128 * 2, PANEL = 64 * 128;
  static constexpr int OFF_Q = 0, OFF_K = 16 * QSTR;
  static constexpr int OFF_V = OFF_K + NS * TB;
  static constexpr int OFF_OWN = OFF_V + NS * TB;     // NS x 64 u16 owner masks
  static constexpr int SMEM = OFF_OWN + NS * 128 + 1024;
};

__device__ __forceinline__ int dq3_sched(int k, int nAo, int nBo, int sh0, int sh1, int& who) {
  const int m = min(nAo, nBo);
  if (k < 2 * m) { who = (k & 1) ? 2 : 1; return (k & 1) ? sh1 + (k >> 1) : (k >> 1); }
  int r = k - 2 * m;
  if (r < nAo - m) { who = 1; return m + r; }
  r -= nAo - m;
  if (r < nBo - m) { who = 2; return sh1 + m + r; }
  r -= nBo - m;
  who = 3;
  return sh0 + r;
}

__device__ __forceinline__ void dq3_body(
    const unsigned char* __restrict__ QD, const bf16* __restrict__ Kg, const bf16* __restrict__ Vg, int CAPQ,
    const int* __restrict__ CNTQ, const int* __restrict__ QK, const unsigned short* __restrict__ QM, bf16* __restrict__ dq,
    int T, int HKV, int NC, float sl2, int BH, int vb, bool reinit) {
  using C = Q3Cfg;
  constexpr int NS = C::NS, PANEL = C::PANEL, D = 128;
  extern __shared__ unsigned char smem_raw[];
  static __shared__ __align__(8) unsigned long long bars[1 + 2 * NS];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - base);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned qfull = bar0;
  auto full = [&](int s) { return bar0 + 8 * (1 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (1 + NS + s); };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bh = vb % BH, pr = NC - 1 - vb / BH;
  const int b = bh / HKV, hk = bh % HKV, i0 = 16 * pr;
  auto sK = [&](int s) { return sbase + C::OFF_K + s * C::TB; };
  auto sV = [&](int s) { return sbase + C::OFF_V + s * C::TB; };
  if (tid == 0) {
    if (reinit) { mbar_inval(qfull); for (int s = 0; s < NS; ++s) { mbar_inval(full(s)); mbar_inval(empty(s)); } }
    mbar_init(qfull, 1);
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 128); mbar_init(empty(s), 256); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int* cn = CNTQ + 3 * (bh * NC + pr);
  const int nA = __ldg(cn), nS = __ldg(cn + 1), nB = __ldg(cn + 2);
  const int ntl = (nA + nS + nB + 63) >> 6;
  const int sh0 = nA >> 6, sh1 = (nA + nS + 63) >> 6;
  const int nAo = sh0, nBo = ntl - sh1;
  const size_t off = (size_t)(bh * NC + pr) * CAPQ;
  if (warp >= 8) {
    reg_dealloc<40>();
    const int pt = tid - 256;
    if (pt == 0) {
      mbar_expect_tx(qfull, 16 * QREC);
#pragma unroll
      for (int n = 0; n < 16; ++n) bulk_load(sbase + C::OFF_Q + n * QSTR, QD + ((size_t)bh * T + i0 + n) * QREC, QREC, qfull);
    }
    const int cc = pt & 15, rb = pt >> 4;
    const size_t rs = (size_t)HKV * D;
    const bf16* Kb = Kg + ((size_t)b * T * HKV + hk) * D + cc * 8;
    const bf16* Vb = Vg + ((size_t)b * T * HKV + hk) * D + cc * 8;
    const unsigned coff = (cc >> 3) * PANEL + ((cc & 7) << 4);
    int who;
    int jn[8];
    if (ntl > 0) {
      const int t0 = dq3_sched(0, nAo, nBo, sh0, sh1, who);
#pragma unroll
      for (int k = 0; k < 8; ++k) jn[k] = __ldg(QK + off + t0 * 64 + rb + 8 * k);
    }
    for (int kq = 0; kq < ntl; ++kq) {
      const int tcur = dq3_sched(kq, nAo, nBo, sh0, sh1, who);
      int jc[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) jc[k] = jn[k];
      if (kq + 1 < ntl) {
        const int tn = dq3_sched(kq + 1, nAo, nBo, sh0, sh1, who);
#pragma unroll
        for (int k = 0; k < 8; ++k) jn[k] = __ldg(QK + off + tn * 64 + rb + 8 * k);
      }
      const int s = kq % NS;
      if (kq >= NS) mbar_wait(empty(s), ((kq / NS) - 1) & 1);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int row = rb + 8 * k;
        const unsigned sw = (coff + row * 128) ^ ((row & 7) << 4);
        cp_async16(sK(s) + sw, Kb + (size_t)jc[k] * rs);
        cp_async16(sV(s) + sw, Vb + (size_t)jc[k] * rs);
      }
      if (pt < 8) cp_async16(sbase + C::OFF_OWN + s * 128 + pt * 16, QM + off + tcur * 64 + pt * 8);
      cp_mbar_arrive(full(s));
    }
    cp_wait_all();
    return;
  }

  reg_alloc<232>();
  const int c = warp >> 2, wq = warp & 3, qd = lane & 3;
  const unsigned sQ = sbase + C::OFF_Q + c * 8 * QSTR, sDO = sQ + 2048;
  mbar_wait(qfull, 0);
  const float2* sLDf = reinterpret_cast<const float2*>(gbase + C::OFF_Q + c * 8 * QSTR + 4096);
  const float2 ld0 = sLDf[(2 * wq) * (QSTR / 8) + (lane >> 2)], ld1 = sLDf[(2 * wq + 1) * (QSTR / 8) + (lane >> 2)];
  const unsigned long long dQd = make_desc(sQ, 16, QSTR), dOd = make_desc(sDO, 16, QSTR);
  const unsigned long long dK0 = make_desc(sK(0), 16, 1024), dV0 = make_desc(sV(0), 16, 1024);
  const unsigned long long dKt0 = make_desc(sK(0), PANEL, 1024);
  constexpr unsigned KSTEP = C::TB >> 4;
  const int sh0b = 8 * c + 2 * wq, sh1b = sh0b + 1;
  float dqa[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) dqa[i] = 0.f;
  auto mine = [&](int kq) { int who; dq3_sched(kq, nAo, nBo, sh0, sh1, who); return ((who >> c) & 1) != 0; };
#if DQ3_PIPE
  // software-pipelined: S/dP of the next own tile is issued before the softmax of the current one
  float sA[32], pA[32], sB[32], pB[32];
  unsigned fa[4][4];
#define DQ3_ISSUE(SX, PX, KQ) do { const int s_ = (KQ) % NS; \
    wg_fence(); \
    _Pragma("unroll") for (int kk = 0; kk < 8; ++kk) \
      wg_ss_n64(SX, dadd(dQd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dK0 + s_ * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0); \
    wg_commit(); \
    _Pragma("unroll") for (int kk = 0; kk < 8; ++kk) \
      wg_ss_n64(PX, dadd(dOd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dV0 + s_ * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0); \
    wg_commit(); } while (0)
#define DQ3_SOFT(SX, PX, KQ) do { const int s_ = (KQ) % NS; \
    unsigned vm0 = 0u, vm1 = 0u; \
    _Pragma("unroll") for (int n = 0; n < 8; ++n) { \
      const unsigned o2 = *reinterpret_cast<const unsigned*>(gbase + C::OFF_OWN + s_ * 128 + 2 * (8 * n + 2 * qd)); \
      vm0 |= (((o2 >> sh0b) & 1u) | (((o2 >> (16 + sh0b)) & 1u) << 1)) << (2 * n); \
      vm1 |= (((o2 >> sh1b) & 1u) | (((o2 >> (16 + sh1b)) & 1u) << 1)) << (2 * n); } \
    _Pragma("unroll") for (int n = 0; n < 8; ++n) { \
      const float q0 = ((vm0 >> (2 * n)) & 1u) ? exp2f(fmaf(SX[4 * n], sl2, -ld0.x)) : 0.f; \
      const float q1 = ((vm0 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(SX[4 * n + 1], sl2, -ld0.x)) : 0.f; \
      const float q2 = ((vm1 >> (2 * n)) & 1u) ? exp2f(fmaf(SX[4 * n + 2], sl2, -ld1.x)) : 0.f; \
      const float q3 = ((vm1 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(SX[4 * n + 3], sl2, -ld1.x)) : 0.f; \
      fa[n >> 1][(n & 1) * 2] = pack_bf16(q0 * (PX[4 * n] - ld0.y), q1 * (PX[4 * n + 1] - ld0.y)); \
      fa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(q2 * (PX[4 * n + 2] - ld1.y), q3 * (PX[4 * n + 3] - ld1.y)); } \
    wg_fence(); \
    _Pragma("unroll") for (int k2 = 0; k2 < 4; ++k2) wg_rs_n128_t(dqa, fa[k2], dadd(dKt0 + s_ * KSTEP, k2 * 2048), 1); \
    wg_commit(); } while (0)
#define DQ3_FENCE(SX, PX) do { _Pragma("unroll") for (int i = 0; i < 32; ++i) { fence_reg(SX[i]); fence_reg(PX[i]); } \
    _Pragma("unroll") for (int i = 0; i < 64; ++i) fence_reg(dqa[i]); \
    _Pragma("unroll") for (int i = 0; i < 4; ++i) _Pragma("unroll") for (int jj = 0; jj < 4; ++jj) fence_ureg(fa[i][jj]); } while (0)
  int kq = 0;
  while (kq < ntl && !mine(kq)) { mbar_wait(full(kq % NS), (kq / NS) & 1); mbar_arrive(empty(kq % NS)); ++kq; }
  int cur = kq, prev = -1;
  bool par = false;
  if (cur < ntl) { mbar_wait(full(cur % NS), (cur / NS) & 1); DQ3_ISSUE(sA, pA, cur); }
  // one step: current tile in (SX, PX), the next one goes to (SY, PY)
#define DQ3_STEP(SX, PX, SY, PY) do { \
    wg_wait<0>(); DQ3_FENCE(SX, PX); \
    if (prev >= 0) { mbar_arrive(empty(prev % NS)); prev = -1; } \
    int nxt = cur + 1; bool pre = false; \
    if (nxt < ntl && mine(nxt)) pre = true; \
    else if (nxt + 1 < ntl && mine(nxt + 1)) { mbar_wait(full(nxt % NS), (nxt / NS) & 1); mbar_arrive(empty(nxt % NS)); ++nxt; pre = true; } \
    if (pre) { mbar_wait(full(nxt % NS), (nxt / NS) & 1); DQ3_ISSUE(SY, PY, nxt); } \
    DQ3_SOFT(SX, PX, cur); \
    prev = cur; \
    if (pre) { cur = nxt; } \
    else { \
      wg_wait<0>(); DQ3_FENCE(SX, PX); mbar_arrive(empty(prev % NS)); prev = -1; \
      int k2 = cur + 1; \
      while (k2 < ntl && !mine(k2)) { mbar_wait(full(k2 % NS), (k2 / NS) & 1); mbar_arrive(empty(k2 % NS)); ++k2; } \
      cur = k2; \
      if (cur < ntl) { mbar_wait(full(cur % NS), (cur / NS) & 1); DQ3_ISSUE(SY, PY, cur); } \
    } } while (0)
  while (cur < ntl) {
    if (!par) DQ3_STEP(sA, pA, sB, pB);
    else DQ3_STEP(sB, pB, sA, pA);
    par = !par;
  }
  wg_wait<0>();
  DQ3_FENCE(sA, pA);
  DQ3_FENCE(sB, pB);
  if (prev >= 0) mbar_arrive(empty(prev % NS));
#else
  for (int kq = 0; kq < ntl; ++kq) {
    const int s = kq % NS;
    mbar_wait(full(s), (kq / NS) & 1);
    if (!mine(kq)) { mbar_arrive(empty(s)); continue; }
    float sacc[32], pacc[32];
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
      wg_ss_n64(sacc, dadd(dQd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dK0 + s * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
    wg_commit();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
      wg_ss_n64(pacc, dadd(dOd, (kk >> 2) * 1024 + (kk & 3) * 32), dadd(dV0 + s * KSTEP, (kk >> 2) * PANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
    wg_commit();
    unsigned vm0 = 0u, vm1 = 0u;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const unsigned o2 = *reinterpret_cast<const unsigned*>(gbase + C::OFF_OWN + s * 128 + 2 * (8 * n + 2 * qd));
      vm0 |= (((o2 >> sh0b) & 1u) | (((o2 >> (16 + sh0b)) & 1u) << 1)) << (2 * n);
      vm1 |= (((o2 >> sh1b) & 1u) | (((o2 >> (16 + sh1b)) & 1u) << 1)) << (2 * n);
    }
    wg_wait<1>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sacc[4 * n] = ((vm0 >> (2 * n)) & 1u) ? exp2f(fmaf(sacc[4 * n], sl2, -ld0.x)) : 0.f;
      sacc[4 * n + 1] = ((vm0 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(sacc[4 * n + 1], sl2, -ld0.x)) : 0.f;
      sacc[4 * n + 2] = ((vm1 >> (2 * n)) & 1u) ? exp2f(fmaf(sacc[4 * n + 2], sl2, -ld1.x)) : 0.f;
      sacc[4 * n + 3] = ((vm1 >> (2 * n + 1)) & 1u) ? exp2f(fmaf(sacc[4 * n + 3], sl2, -ld1.x)) : 0.f;
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
    unsigned fa[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      fa[n >> 1][(n & 1) * 2] = pack_bf16(sacc[4 * n] * (pacc[4 * n] - ld0.y), sacc[4 * n + 1] * (pacc[4 * n + 1] - ld0.y));
      fa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(sacc[4 * n + 2] * (pacc[4 * n + 2] - ld1.y), sacc[4 * n + 3] * (pacc[4 * n + 3] - ld1.y));
    }
    wg_fence();
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) wg_rs_n128_t(dqa, fa[k2], dadd(dKt0 + s * KSTEP, k2 * 2048), 1);
    wg_commit();
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 64; ++i) fence_reg(dqa[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) fence_ureg(fa[i][jj]);
    mbar_arrive(empty(s));
  }
#endif
  const int HQ = HKV * 8;
  bf16* o0 = dq + (((size_t)b * T + i0 + 8 * c + 2 * wq) * HQ + hk * 8 + (lane >> 2)) * D + 2 * qd;
  bf16* o1 = o0 + (size_t)HQ * D;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    *reinterpret_cast<unsigned*>(o0 + 8 * n) = pack_bf16(dqa[4 * n], dqa[4 * n + 1]);
    *reinterpret_cast<unsigned*>(o1 + 8 * n) = pack_bf16(dqa[4 * n + 2], dqa[4 * n + 3]);
  }
}

__global__ void __launch_bounds__(384, 1) dq3_main(
    const unsigned char* __restrict__ QD, const bf16* __restrict__ Kg, const bf16* __restrict__ Vg, int CAPQ,
    const int* __restrict__ CNTQ, const int* __restrict__ QK, const unsigned short* __restrict__ QM, bf16* __restrict__ dq,
    int T, int HKV, int NC, float sl2, int BH, int* __restrict__ ctr) {
  __shared__ int sVb;
  const int nvb = BH * NC;
  int vb = blockIdx.x;
  while (vb < nvb) {
    dq3_body(QD, Kg, Vg, CAPQ, CNTQ, QK, QM, dq, T, HKV, NC, sl2, BH, vb, vb != (int)blockIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) sVb = (int)gridDim.x + atomicAdd(ctr, 1);
    __syncthreads();
    vb = sVb;
  }
}
"""

HOST = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn get_encode() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* ptr = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &qres);
    TORCH_CHECK(ptr != nullptr, "cuTensorMapEncodeTiled not found");
    fn = (EncodeFn)ptr;
  }
  return fn;
}
static CUtensorMap make_map(void* ptr, int B, int T, int H, int D, int bh, int bt) {
  CUtensorMap m;
  cuuint64_t dims[4] = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T, (cuuint64_t)B};
  cuuint64_t strides[3] = {(cuuint64_t)D * 2, (cuuint64_t)H * D * 2, (cuuint64_t)T * H * D * 2};
  cuuint32_t box[4] = {64, (cuuint32_t)bh, (cuuint32_t)bt, 1};
  cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "tensor map encode failed ", (int)r);
  return m;
}
static int num_sms() {
  static int n = 0;
  if (!n) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); }
  return n;
}
static inline long al4(long x) { return (x + 3) & ~3L; }

// flags: bit0 kv kernel load-only, bit1 skip kv kernel, bit2 skip dq kernel, bit3 stop after index
std::vector<torch::Tensor> hyb(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t BQ, int64_t flags) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int HKV = k.size(2), TOPK = sel.size(3);
  TORCH_CHECK(D == 128 && HQ == 8 * HKV && T % 64 == 0, "hyb: d128, G8, T%64");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int NT = T / 64, NC = T / 16, NG = T / 8, BH = B * HKV;
  const int nb = (T + (int)BQ - 1) / (int)BQ, CPB = (int)BQ / 16;
  const int CAPC = 16 * (int)std::min((long)NT, (long)TOPK + window / 64 + 3);
  const long Emax = (long)BH * NC * CAPC;
  const int SPLIT = DQ_SPLIT;
  const int CAPQ = SPLIT ? (int)((std::min((long)16 * TOPK + window + 16 + sink, (long)T) + 63) & ~63L)
                         : (int)((std::min((long)8 * TOPK + window + 8 + sink, (long)T) + 63) & ~63L);
  const long Qmax = (long)BH * (SPLIT ? NC : NG) * CAPQ;
  const long nItems = (long)nb * BH * NT;
  // int buffer layout
  long o_items = 0, o_em = o_items + 4 * nItems, o_lm = o_em + 2 * Emax, o_nit = o_lm + 2 * Emax, o_cntk = o_nit + 4;
  long o_tot = al4(o_cntk + (long)BH * NT * NC), o_base = al4(o_tot + BH * NT), o_cntq = al4(o_base + BH * NT);
  long o_lpre = al4(o_cntq + 2 * (long)BH * NG), o_eq = al4(o_lpre + (long)BH * NC * (NT + 1)), o_lq = al4(o_eq + Emax);
  long o_qk = al4(o_lq + Emax), o_qm = al4(o_qk + Qmax), o_end = al4(o_qm + Qmax / 2 + 1);
  auto ibuf = torch::empty({o_end}, q.options().dtype(torch::kInt32));
  int* ib = ibuf.data_ptr<int>();
  int4* ITEMS = reinterpret_cast<int4*>(ib + o_items);
  uint2* EM = reinterpret_cast<uint2*>(ib + o_em);
  uint2* LM = reinterpret_cast<uint2*>(ib + o_lm);
  int* NIT = ib + o_nit; int* CNTK = ib + o_cntk; int* TOT = ib + o_tot; int* BASE = ib + o_base;
  int* CNTQ = ib + o_cntq; int* LPRE = ib + o_lpre; int* EQ = ib + o_eq; int* LQ = ib + o_lq; int* QK = ib + o_qk;
  unsigned char* QM = reinterpret_cast<unsigned char*>(ib + o_qm);
  const long nkv = k.numel();
  auto fbuf = torch::empty({(long)BH * T * 16 + 2 * nkv + (long)BH * T * 1040}, q.options().dtype(torch::kFloat32));
  float2* LD = reinterpret_cast<float2*>(fbuf.data_ptr<float>());
  float* dk32 = fbuf.data_ptr<float>() + (long)BH * T * 16;
  float* dv32 = dk32 + nkv;
  unsigned char* QD = reinterpret_cast<unsigned char*>(dv32 + nkv);
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);

  const long nthr = (long)B * T * HQ * 16;
  prep_k<<<(int)((nthr + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(o.data_ptr()),
      reinterpret_cast<const bf16*>(dout.data_ptr()), lse.data_ptr<float>(), LD, QD, dk32, dv32, nkv / 8, B, T, HQ, HKV,
      1.4426950408889634f, (float)std::log2(scale));
  const int ism = (16 * (2 * NT + 1) + NT + 2) * 4;
  static int iset = 0;
  if (ism > 48 * 1024 && ism > iset) { cudaFuncSetAttribute(idx_k, cudaFuncAttributeMaxDynamicSharedMemorySize, ism); iset = ism; }
  idx_k<<<dim3(NC, BH), 512, ism, stream>>>(sel.data_ptr<int>(), CNTK, LPRE, LQ, LM, CNTQ, QK, QM,
                                             T, TOPK, NT, NC, NG, CAPC, CAPQ, (int)window, (int)sink, SPLIT);
  scan_k<<<dim3(NT, BH), 256, 0, stream>>>(CNTK, TOT, NT, NC);
  plan_k<<<1, 1024, BH * NT * 4, stream>>>(CNTK, TOT, BASE, ITEMS, NIT, BH, NT, NC, NG, CPB, nb);
  move_k<<<dim3(NC, BH), 256, (NT + 2) * 4, stream>>>(CNTK, LPRE, LQ, LM, BASE, EQ, EM, NT, NC, CAPC);
  if (flags & 8) return {dq, dk, dv, ibuf};
  const float sl2 = (float)(scale * 1.4426950408889634);
  const bool side = (flags & 32) && !(flags & 4);
  static cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  if (!ev0) { cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming); cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming); }
  cudaStream_t s2 = side ? at::cuda::getStreamFromPool().stream() : stream;
  if (side) { cudaEventRecord(ev0, stream); cudaStreamWaitEvent(s2, ev0, 0); }
  if (!(flags & 4) && SPLIT) {
    static bool q3init = false;
    if (!q3init) { cudaFuncSetAttribute(dq3_main, cudaFuncAttributeMaxDynamicSharedMemorySize, Q3Cfg::SMEM); q3init = true; }
    dq3_main<<<std::min(BH * NC, num_sms()), 384, Q3Cfg::SMEM, s2>>>(QD, reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()), CAPQ, CNTQ, QK, reinterpret_cast<const unsigned short*>(QM),
        reinterpret_cast<bf16*>(dq.data_ptr()), T, HKV, NC, sl2, BH, NIT + 2);
  }
  if (!(flags & 4) && !SPLIT) {
    static bool q2init = false;
    if (!q2init) { cudaFuncSetAttribute(dq2_main, cudaFuncAttributeMaxDynamicSharedMemorySize, Q2Cfg::SMEM); q2init = true; }
    dq2_main<<<BH * NG, 384, Q2Cfg::SMEM, s2>>>(QD, reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()), CAPQ, CNTQ, QK, QM, reinterpret_cast<bf16*>(dq.data_ptr()),
        T, HKV, NG, sl2);
  }
  if (side) cudaEventRecord(ev1, s2);
  if (!(flags & 2)) {
    CUtensorMap tk = make_map(k.data_ptr(), B, T, HKV, D, 1, 64);
    CUtensorMap tv = make_map(v.data_ptr(), B, T, HKV, D, 1, 64);
    static bool kinit = false;
    if (!kinit) {
      cudaFuncSetAttribute(kv_main<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, KCfg::SMEM);
      cudaFuncSetAttribute(kv_main<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, KCfg::SMEM);
      cudaFuncSetAttribute(kv_main<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, KCfg::SMEM);
      kinit = true;
    }
    if (KV_PERSIST && !(flags & (64 | 1))) {
      static bool pinit = false;
      static int nsm = 0;
      if (!pinit) {
        cudaFuncSetAttribute(kvp_main, cudaFuncAttributeMaxDynamicSharedMemorySize, KCfg::SMEM);
        int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        pinit = true;
      }
      kvp_main<<<nsm, 384, KCfg::SMEM, stream>>>(tk, tv, QD, ITEMS, NIT, NIT + 1, EQ, EM, dk32, dv32, T, HKV, NT, sl2);
    } else if (flags & 64)
      kv_main<2><<<(int)std::min<long>(nItems, num_sms()), 384, KCfg::SMEM, stream>>>(tk, tv, QD, ITEMS, NIT, EQ, EM, LD, dk32, dv32, T, HKV, NT, sl2, NIT + 3);
    else if (flags & 1)
      kv_main<1><<<(int)std::min<long>(nItems, num_sms()), 384, KCfg::SMEM, stream>>>(tk, tv, QD, ITEMS, NIT, EQ, EM, LD, dk32, dv32, T, HKV, NT, sl2, NIT + 3);
    else
      kv_main<0><<<(int)std::min<long>(nItems, num_sms()), 384, KCfg::SMEM, stream>>>(tk, tv, QD, ITEMS, NIT, EQ, EM, LD, dk32, dv32, T, HKV, NT, sl2, NIT + 3);
  }
  cvt_k<<<(int)((nkv / 8 + 255) / 256), 256, 0, stream>>>(dk32, reinterpret_cast<bf16*>(dk.data_ptr()),
      dv32, reinterpret_cast<bf16*>(dv.data_ptr()), nkv / 8, 1.0f, (float)(1.0 / scale));
  if (side) cudaStreamWaitEvent(stream, ev1, 0);
  return {dq, dk, dv, ibuf};
}
"""

KV_NS = int(os.environ.get("HYB_KV_NS", "4"))
DQ_NS = int(os.environ.get("HYB_DQ_NS", "2"))
DQ2_NS = int(os.environ.get("HYB_DQ2_NS", "5"))
DQ_PP = int(os.environ.get("HYB_DQ_PP", "0"))
DQ3_NS = int(os.environ.get("HYB_DQ3_NS", "4"))
DQ_SPLIT = int(os.environ.get("HYB_DQSPLIT", "1"))
DQ3_PIPE = int(os.environ.get("HYB_DQ3_PIPE", "0"))
KV_PERSIST = int(os.environ.get("HYB_KV_PERSIST", "0"))


def _build():
    src = HEADER + _gen("wg_ss_n64", 64, False) + _gen("wg_rs_n128_t", 128, True, 0, 1)
    return f"#define KV_NS {KV_NS}\n#define DQ_NS {DQ_NS}\n#define DQ2_NS {DQ2_NS}\n#define DQ_PP {DQ_PP}\n#define DQ3_NS {DQ3_NS}\n#define DQ_SPLIT {DQ_SPLIT}\n#define DQ3_PIPE {DQ3_PIPE}\n#define KV_PERSIST {KV_PERSIST}\n" + src + KERN


_ext = load_inline(
    name="lsa_bwd_hybrid_kernel" + os.environ.get("HYB_SUFFIX", ""),
    cpp_sources="std::vector<torch::Tensor> hyb(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, "
    "torch::Tensor o, torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale, "
    "int64_t BQ, int64_t flags);",
    cuda_sources=_build() + HOST,
    functions=["hyb"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
    verbose=bool(int(os.environ.get("SA_VERBOSE", "0"))),
)

BQ = int(os.environ.get("HYB_BQ", "2048"))
FLAGS = int(os.environ.get("HYB_FLAGS", "0"))


def _prep(q, k, v, selected, o, lse, do):
    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    lse = lse.contiguous().float()
    selected = selected.to(torch.int32).contiguous()
    return q, k, v, selected, o, lse, do


def attention_backward(q, k, v, selected, o, lse, do, window=0, sink=0, scale=None, flags=None, bq=None):
    D = q.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    q, k, v, selected, o, lse, do = _prep(q, k, v, selected, o, lse, do)
    dq, dk, dv, _ = _ext.hyb(q, k, v, selected, o, lse, do, int(window), int(sink), float(scale),
                             BQ if bq is None else bq, FLAGS if flags is None else flags)
    return dq, dk, dv


def debug_index(q, k, v, selected, o, lse, do, window=0, sink=0, scale=None, bq=None):
    D = q.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    q, k, v, selected, o, lse, do = _prep(q, k, v, selected, o, lse, do)
    return _ext.hyb(q, k, v, selected, o, lse, do, int(window), int(sink), float(scale), BQ if bq is None else bq, 8)
