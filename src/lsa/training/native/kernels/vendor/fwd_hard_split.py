# Vendored from kernels/base_split.py (md5 745aca6d84ca90fd6c042345e0905514), by a local Claude Fable 5.1 session, 2026-10-14;
# see __init__.py. fwd_hard's split pipeline (base_split.py). One edit: load_inline's name "sfwd_push_split" renamed.
import math
import torch
from torch.utils.cpp_extension import load_inline


def _gen(N, rs) -> str:
    nreg = N // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    if rs:
        a = "{" + ",".join(f"%{nreg + i}" for i in range(4)) + "}"
        b = f"%{nreg + 4}"
        sc = f"%{nreg + 5}"
        tail = "p, 1, 1, 1"
        ins = ", ".join(f'"r"(a[{i}])' for i in range(4)) + ', "l"(db), "r"(sd)'
        sig = "float* d, const unsigned* a, unsigned long long db, int sd"
    else:
        a = f"%{nreg}"
        b = f"%{nreg + 1}"
        sc = f"%{nreg + 2}"
        tail = "p, 1, 1, 0, 0"
        ins = '"l"(da), "l"(db), "r"(sd)'
        sig = "float* d, unsigned long long da, unsigned long long db, int sd"
    name = f"wgmma_{'rs' if rs else 'ss'}_n{N}"
    return (f"__device__ __forceinline__ void {name}({sig}) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sc}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {{{d}}}, {a}, {b}, {tail};\\n}}\\n"\n'
            f"    : {outs} : {ins} : \"memory\");\n}}\n")


HEADER = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
namespace {
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ void cp_async16s(unsigned s, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async16h(unsigned s, const void* gmem, unsigned long long pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, 16, %2;\n" :: "r"(s), "l"(gmem), "l"(pol) : "memory");
}
__device__ __forceinline__ unsigned long long make_pol_last() {
  unsigned long long p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p)); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void cp_mbar_arrive(unsigned addr) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
__device__ __forceinline__ void fence_ureg(unsigned& r) { asm volatile("" : "+r"(r) :: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
template <int N> __device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> __device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned addr, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned addr, unsigned parity) {
  unsigned ok = 0;
  while (!ok) {
    asm volatile("{\n.reg .pred pp;\nmbarrier.try_wait.parity.shared::cta.b64 pp, [%1], %2;\nselp.u32 %0, 1, 0, pp;\n}\n"
                 : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void tma_load_4d(unsigned dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, unsigned mbar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
               :: "r"(dst), "l"((unsigned long long)map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar) : "memory");
}
__device__ __forceinline__ void tma_prefetch_4d(const CUtensorMap* map, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];\n"
               :: "l"((unsigned long long)map), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void st_shared_u32(unsigned addr, unsigned v) {
  asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_shared_v4(unsigned addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long mkd(unsigned lo, unsigned hi) {
  unsigned long long d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "r"(lo), "r"(hi));
  return d;
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}
"""

KERN = r"""
#ifndef NSV128
#define NSV128 3
#endif
__device__ __forceinline__ unsigned range_bits(int a, int b, int w) {
  int lo = max(a, w * 32), hi = min(b, w * 32 + 31);
  if (lo > hi) return 0u;
  int n = hi - lo + 1;
  unsigned m = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
  return m << (lo - w * 32);
}
__device__ __forceinline__ unsigned perm_range(int lo, int hi, int q) {
  unsigned r = 0;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int a = lo - 2 * q - j, b = hi - 2 * q - j;
    const int ka = a <= 0 ? 0 : (a + 7) >> 3;
    const int kb = b < 0 ? -1 : min(15, b >> 3);
    if (ka <= kb) r |= ((0x55555555u >> (30 - 2 * kb)) & (0xFFFFFFFFu << (2 * ka))) << j;
  }
  return r;
}

template <int C>
__device__ __forceinline__ void load_ids(const int* __restrict__ sb, int x0, int* ids) {
  if constexpr (C % 4 == 0) {
#pragma unroll
    for (int q = 0; q < C / 4; ++q) {
      int4 v = __ldg(reinterpret_cast<const int4*>(sb + x0) + q);
      ids[4 * q] = v.x; ids[4 * q + 1] = v.y; ids[4 * q + 2] = v.z; ids[4 * q + 3] = v.w;
    }
  } else {
#pragma unroll
    for (int e = 0; e < C; ++e) ids[e] = __ldg(sb + x0 + e);
  }
}

template <int C, class F>
__device__ __forceinline__ void proc_ids(const int* ids, int lim, int p, unsigned* arr, F& f) {
  int cur = -1;
  unsigned cb = 0;
#pragma unroll
  for (int e = 0; e < C; ++e) {
    const int j = ids[e];
    if (j >= 0 && j <= lim) {
      int idx; unsigned bit;
      f(j, p, idx, bit);
      if (idx != cur) {
        if (cur >= 0) atomicOr(arr + cur, cb);
        cur = idx; cb = 0;
      }
      cb |= bit;
    }
  }
  if (cur >= 0) atomicOr(arr + cur, cb);
}

template <int C, class F>
__device__ __forceinline__ void fold_ids(const int* __restrict__ sb, int nsel, int TOPK, int pf, int tid, int NTH,
                                         unsigned* arr, F f) {
  const int nch = nsel / C;
  for (int ch = tid; ch < nch; ch += 2 * NTH) {
    int a[C], b[C];
    const int ch2 = ch + NTH;
    const bool hb = ch2 < nch;
    load_ids<C>(sb, ch * C, a);
    if (hb) load_ids<C>(sb, ch2 * C, b);
    {
      const int p = (ch * C) / TOPK;
      proc_ids<C>(a, pf + p, p, arr, f);
    }
    if (hb) {
      const int p = (ch2 * C) / TOPK;
      proc_ids<C>(b, pf + p, p, arr, f);
    }
  }
}

template <class F>
__device__ __forceinline__ void for_ids(const int* sb, int nsel, int TOPK, int pf, int tid, int NTH, unsigned* arr, F f) {
  if ((TOPK & 7) == 0) fold_ids<8>(sb, nsel, TOPK, pf, tid, NTH, arr, f);
  else if ((TOPK & 3) == 0) fold_ids<4>(sb, nsel, TOPK, pf, tid, NTH, arr, f);
  else fold_ids<1>(sb, nsel, TOPK, pf, tid, NTH, arr, f);
}

template <class F>
__device__ __forceinline__ void fold_simple(const int* __restrict__ sb, int nsel, int TOPK, int pf, int tid, int NTH, F f) {
  const int nch = nsel >> 3;
  for (int ch = tid; ch < nch; ch += 2 * NTH) {
    const int ch2 = ch + NTH;
    const int4 a0 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch), a1 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch + 1);
    int4 b0 = make_int4(-1, -1, -1, -1), b1 = b0;
    if (ch2 < nch) { b0 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch2); b1 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch2 + 1); }
    const int pa = (ch * 8) / TOPK, pb = (ch2 * 8) / TOPK;
    const unsigned la = (unsigned)(pf + pa), lb = (unsigned)(pf + pb);
    const int ida[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const int idb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int e = 0; e < 8; ++e) if ((unsigned)ida[e] <= la) f(ida[e], pa);
#pragma unroll
    for (int e = 0; e < 8; ++e) if ((unsigned)idb[e] <= lb) f(idb[e], pb);
  }
}

struct Item { int hk, b, R0, pf, pl, NP, nbq, NW, wlo, nband, sinkEnd, tl0; };

__device__ __forceinline__ Item get_item(int it, int HB, int HKV, int nqblk, int G, int TG, int window, int sink) {
  Item x;
  const int qblk = nqblk - 1 - it / HB;
  const int rest = it % HB;
  x.hk = rest % HKV; x.b = rest / HKV;
  x.R0 = qblk * 128;
  x.pf = x.R0 / G; x.pl = (min(x.R0 + 128, TG) - 1) / G;
  x.NP = x.pl - x.pf + 1;
  x.nbq = x.pl / 128 + 1; x.NW = x.nbq * 4;
  x.wlo = window > 0 ? max(0, x.pf - window + 1) : x.pl + 1;
  x.nband = window > 0 ? (x.pl - x.wlo + 1) >> 7 : 0;
  x.tl0 = window > 0 ? x.wlo + (x.nband << 7) : x.pl + 1;
  x.sinkEnd = min(sink, x.wlo);
  return x;
}

template <bool MG>
__global__ void __launch_bounds__(256) plan_k(const int* __restrict__ SEL, unsigned* __restrict__ PL, int* __restrict__ ctr,
    int T, int HKV, int G, int TOPK, int window, int sink, int THR, int nqblk, int HB, int planWords, int offG, int offM,
    float dfac, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  // Split-union plan: queries p < QA belong to consumer warpgroup A, the rest to B. Gathered keys are
  // ordered [A-only | shared | B-only] so that each warpgroup computes only tiles holding its keys.
  constexpr int NTH = 256;
  extern __shared__ __align__(16) unsigned psm[];
  __shared__ unsigned long long s_red[NTH / 32];
  __shared__ int s_redi[NTH / 32];
  pdl_trigger();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  if (it == 0 && tid == 0) *ctr = 0;
  const Item x = get_item(it, HB, HKV, nqblk, G, T * G, window, sink);
  const int NW = x.NW, NP = x.NP, pf = x.pf, sinkEnd = x.sinkEnd, nbq = x.nbq, tl0 = x.tl0, pl = x.pl;
  const int QA = 64 / G;
  const int* sb = SEL + ((size_t)(x.b * HKV + x.hk) * T + pf) * TOPK;
  const int nsel = NP * TOPK;
  for (int xx = (2 * NTH + tid) * 32; xx < nsel; xx += NTH * 32) prefetch_l2(sb + xx);
  unsigned* P = PL + (size_t)it * planWords;
  int* dBlk = reinterpret_cast<int*>(P + 8);
  int* gi = reinterpret_cast<int*>(P + offG);
  unsigned* UA = psm;
  unsigned* UB = psm + NW;
  unsigned long long* PX = reinterpret_cast<unsigned long long*>(psm + 2 * NW);
  int* blkTile = reinterpret_cast<int*>(psm + 4 * NW);
  unsigned* Ms = psm + ((4 * NW + nbq + 3) & ~3);
  const int aHi = min(pl, pf + QA - 1);
  for (int w = tid; w < NW; w += NTH) {
    const unsigned sk = sinkEnd > 0 ? range_bits(0, sinkEnd - 1, w) : 0u;
    UA[w] = sk | (tl0 <= aHi ? range_bits(tl0, aHi, w) : 0u);
    UB[w] = NP > QA ? (sk | (tl0 <= pl ? range_bits(tl0, pl, w) : 0u)) : 0u;
  }
  __syncthreads();
  for_ids(sb, nsel, TOPK, pf, tid, NTH, UA, [&](int j, int p, int& idx, unsigned& bit) {
    idx = (p >= QA ? NW : 0) + (j >> 5); bit = 1u << (j & 31);
  });
  __syncthreads();
  int utot = 0;
  for (int w = tid; w < NW; w += NTH) utot += __popc(UA[w] | UB[w]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) utot += __shfl_xor_sync(0xffffffff, utot, off);
  if (lane == 0) s_redi[warp] = utot;
  __syncthreads();
  utot = 0;
#pragma unroll
  for (int k = 0; k < NTH / 32; ++k) utot += s_redi[k];
  const bool dense = (float)nbq <= dfac * (float)(x.nband + (utot + 127) / 128);
  const int thrE = dense ? 1 : THR;
  const int bandBlk = (dense && window > 0) ? (x.wlo >> 7) : 0x7fffffff;
  unsigned long long carry = 0;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const unsigned a = w < NW ? UA[w] : 0u, b = w < NW ? UB[w] : 0u;
    const int pc = __popc(a | b);
    int cnt = pc + __shfl_xor_sync(0xffffffff, pc, 1);
    cnt += __shfl_xor_sync(0xffffffff, cnt, 2);
    const bool dn = w < NW && (cnt >= thrE || (w >> 2) >= bandBlk);
    const unsigned long long val = dn ? ((w & 3) == 0 ? (1ull << 48) : 0ull)
        : ((unsigned long long)__popc(a & ~b) | ((unsigned long long)__popc(a & b) << 16) |
           ((unsigned long long)__popc(b & ~a) << 32));
    unsigned long long xs = val;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      unsigned long long y = __shfl_up_sync(0xffffffff, xs, off);
      if (lane >= off) xs += y;
    }
    if (lane == 31) s_red[warp] = xs;
    __syncthreads();
    unsigned long long pre = carry, tot = 0;
#pragma unroll
    for (int k = 0; k < NTH / 32; ++k) { unsigned long long v = s_red[k]; if (k < warp) pre += v; tot += v; }
    const unsigned long long excl = pre + xs - val;
    if (w < NW) {
      if (dn) {
        if ((w & 3) == 0) { const int ti = (int)(excl >> 48); blkTile[w >> 2] = ti; dBlk[ti] = w >> 2; }
      } else if ((w & 3) == 0) blkTile[w >> 2] = -1;
      PX[w] = excl;
    }
    carry += tot;
    __syncthreads();
  }
  const int nd = (int)(carry >> 48), nA = (int)(carry & 0xFFFF), nS = (int)((carry >> 16) & 0xFFFF),
            nB = (int)((carry >> 32) & 0xFFFF);
  const int ng = nA + nS + nB, ngt = (ng + 127) >> 7;
  if (tid == 0) { P[0] = nd; P[1] = ng; P[2] = dense ? 1u : 0u; P[3] = nA; P[4] = nS; }
  auto kpos = [&](unsigned a, unsigned b, unsigned long long px, int bit) -> int {
    const unsigned lm = (1u << bit) - 1u;
    const bool ia = (a >> bit) & 1u, ib = (b >> bit) & 1u;
    if (ia && ib) return nA + (int)((px >> 16) & 0xFFFF) + __popc(a & b & lm);
    if (ia) return (int)(px & 0xFFFF) + __popc(a & ~b & lm);
    return nA + nS + (int)((px >> 32) & 0xFFFF) + __popc(b & ~a & lm);
  };
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const bool live = w < NW && blkTile[w >> 2] < 0;
    const unsigned a = live ? UA[w] : 0u, b = live ? UB[w] : 0u;
    const unsigned long long px = live ? PX[w] : 0ull;
    const int wb = bw + warp * 32;
    unsigned act = __ballot_sync(0xffffffff, (a | b) != 0u);
    while (act) {
      const int k = __ffs(act) - 1;
      act &= act - 1;
      const unsigned ak = __shfl_sync(0xffffffff, a, k), bk = __shfl_sync(0xffffffff, b, k);
      const unsigned long long pk = __shfl_sync(0xffffffff, px, k);
      if (((ak | bk) >> lane) & 1u) gi[kpos(ak, bk, pk, lane)] = (wb + k) * 32 + lane;
    }
  }
  unsigned* M = MG ? (P + offM) : Ms;
  const int nmw = (nd + ngt) * NP * 4;
  for (int xx = tid; xx < nmw; xx += NTH) M[xx] = 0u;
  for (int xx = ng + tid; xx < ngt * 128; xx += NTH) gi[xx] = 0;
  __syncthreads();
  auto slotf = [&](int j, int p, int& idx, unsigned& bit) {
    const int bt = blkTile[j >> 7];
    int mt, pos;
    if (bt >= 0) { mt = bt; pos = j & 127; }
    else {
      const int w = j >> 5;
      const int slot = kpos(UA[w], UB[w], PX[w], j & 31);
      mt = nd + (slot >> 7); pos = slot & 127;
    }
    idx = (mt * NP + p) * 4 + ((pos >> 1) & 3);
    bit = 1u << (((pos >> 3) << 1) | (pos & 1));
  };
  for_ids(sb, nsel, TOPK, pf, tid, NTH, M, slotf);
  if (sinkEnd > 0) {
    for (int xx = tid; xx < NP * sinkEnd; xx += NTH) {
      const int p = xx / sinkEnd, j = xx - p * sinkEnd;
      if (j <= pf + p) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (!dense && tl0 <= pl) {
    const int tlen = pl - tl0 + 1;
    for (int xx = tid; xx < NP * tlen; xx += NTH) {
      const int p = xx / tlen, j = tl0 + xx - p * tlen, i = pf + p;
      if (j <= i && (i - j < window || j < sink)) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (dense && (window > 0 || sink > 0)) {
    for (int xx = tid; xx < NP * nbq * 4; xx += NTH) {
      const int w4 = xx & 3, r = xx >> 2, bb = r % nbq, p = r / nbq;
      const int bt = blkTile[bb];
      if (bt < 0) continue;
      const int i = pf + p, kb0 = bb * 128;
      unsigned bits = 0;
      if (window > 0) bits |= perm_range(max(i - window + 1, 0) - kb0, i - kb0, w4);
      if (sink > 0) bits |= perm_range(-kb0, min(sink - 1, i) - kb0, w4);
      if (bits) atomicOr(M + (bt * NP + p) * 4 + w4, bits);
    }
  }
  if constexpr (!MG) {
    __syncthreads();
    uint4* Mg = reinterpret_cast<uint4*>(P + offM);
    const uint4* M4 = reinterpret_cast<const uint4*>(Ms);
    for (int xx = tid; xx < nmw / 4; xx += NTH) Mg[xx] = M4[xx];
  }
}


// Plan v2: the item's ids are read once into registers (R rounds of 8 per thread), and each key word
// gets one 16-byte record {UA, UB, A/S bases, B base} so the mask pass costs one LDS.128 per id.
template <int R>
__global__ void __launch_bounds__(256) plan2_k(const int* __restrict__ SEL, unsigned* __restrict__ PL, int* __restrict__ ctr,
    int T, int HKV, int G, int TOPK, int window, int sink, int THR, int nqblk, int HB, int planWords, int offG, int offM,
    float dfac, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int NTH = 256;
  extern __shared__ __align__(16) unsigned psm[];
  __shared__ unsigned long long s_red[NTH / 32];
  __shared__ int s_redi[NTH / 32];
  pdl_trigger();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  if (it == 0 && tid == 0) *ctr = 0;
  const Item x = get_item(it, HB, HKV, nqblk, G, T * G, window, sink);
  const int NW = x.NW, NP = x.NP, pf = x.pf, sinkEnd = x.sinkEnd, nbq = x.nbq, tl0 = x.tl0, pl = x.pl;
  const int QA = 64 / G;
  const int* sb = SEL + ((size_t)(x.b * HKV + x.hk) * T + pf) * TOPK;
  const int nch = NP * TOPK / 8;
  int id[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int ch = r * NTH + tid;
    if (ch < nch) {
      const int4 v0 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch);
      const int4 v1 = __ldg(reinterpret_cast<const int4*>(sb) + 2 * ch + 1);
      id[r][0] = v0.x; id[r][1] = v0.y; id[r][2] = v0.z; id[r][3] = v0.w;
      id[r][4] = v1.x; id[r][5] = v1.y; id[r][6] = v1.z; id[r][7] = v1.w;
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) id[r][e] = -1;
    }
  }
  unsigned* P = PL + (size_t)it * planWords;
  int* dBlk = reinterpret_cast<int*>(P + 8);
  int* gi = reinterpret_cast<int*>(P + offG);
  unsigned* UA = psm;
  unsigned* UB = psm + NW;
  unsigned long long* PX = reinterpret_cast<unsigned long long*>(psm + 2 * NW);
  uint4* REC = reinterpret_cast<uint4*>(psm + 4 * NW);
  int* blkTile = reinterpret_cast<int*>(psm + 8 * NW);
  unsigned* Ms = psm + ((8 * NW + nbq + 3) & ~3);
  const int aHi = min(pl, pf + QA - 1);
  for (int w = tid; w < NW; w += NTH) {
    const unsigned sk = sinkEnd > 0 ? range_bits(0, sinkEnd - 1, w) : 0u;
    UA[w] = sk | (tl0 <= aHi ? range_bits(tl0, aHi, w) : 0u);
    UB[w] = NP > QA ? (sk | (tl0 <= pl ? range_bits(tl0, pl, w) : 0u)) : 0u;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int ch = r * NTH + tid;
    const int p = (ch * 8) / TOPK;
    const int lim = pf + p;
    unsigned* arr = p >= QA ? UB : UA;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int j = id[r][e];
      if ((unsigned)j <= (unsigned)lim) atomicOr(arr + (j >> 5), 1u << (j & 31));
    }
  }
  __syncthreads();
  int utot = 0;
  for (int w = tid; w < NW; w += NTH) utot += __popc(UA[w] | UB[w]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) utot += __shfl_xor_sync(0xffffffff, utot, off);
  if (lane == 0) s_redi[warp] = utot;
  __syncthreads();
  utot = 0;
#pragma unroll
  for (int k = 0; k < NTH / 32; ++k) utot += s_redi[k];
  const bool dense = (float)nbq <= dfac * (float)(x.nband + (utot + 127) / 128);
  const int thrE = dense ? 1 : THR;
  const int bandBlk = (dense && window > 0) ? (x.wlo >> 7) : 0x7fffffff;
  unsigned long long carry = 0;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const unsigned a = w < NW ? UA[w] : 0u, b = w < NW ? UB[w] : 0u;
    const int pc = __popc(a | b);
    int cnt = pc + __shfl_xor_sync(0xffffffff, pc, 1);
    cnt += __shfl_xor_sync(0xffffffff, cnt, 2);
    const bool dn = w < NW && (cnt >= thrE || (w >> 2) >= bandBlk);
    const unsigned long long val = dn ? ((w & 3) == 0 ? (1ull << 48) : 0ull)
        : ((unsigned long long)__popc(a & ~b) | ((unsigned long long)__popc(a & b) << 16) |
           ((unsigned long long)__popc(b & ~a) << 32));
    unsigned long long xs = val;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      unsigned long long y = __shfl_up_sync(0xffffffff, xs, off);
      if (lane >= off) xs += y;
    }
    if (lane == 31) s_red[warp] = xs;
    __syncthreads();
    unsigned long long pre = carry, tot = 0;
#pragma unroll
    for (int k = 0; k < NTH / 32; ++k) { unsigned long long v = s_red[k]; if (k < warp) pre += v; tot += v; }
    const unsigned long long excl = pre + xs - val;
    if (w < NW) {
      if (dn) {
        if ((w & 3) == 0) { const int ti = (int)(excl >> 48); blkTile[w >> 2] = ti; dBlk[ti] = w >> 2; }
      } else if ((w & 3) == 0) blkTile[w >> 2] = -1;
      PX[w] = excl;
    }
    carry += tot;
    __syncthreads();
  }
  const int nd = (int)(carry >> 48), nA = (int)(carry & 0xFFFF), nS = (int)((carry >> 16) & 0xFFFF),
            nB = (int)((carry >> 32) & 0xFFFF);
  const int ng = nA + nS + nB, ngt = (ng + 127) >> 7;
  if (tid == 0) { P[0] = nd; P[1] = ng; P[2] = dense ? 1u : 0u; P[3] = nA; P[4] = nS; }
  const int gb = nd * 128;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    unsigned a = 0, b = 0;
    if (w < NW) {
      const int bt = blkTile[w >> 2];
      uint4 rec;
      if (bt >= 0) {
        rec = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, (unsigned)(bt * 128 + (w & 3) * 32) << 16, 0u);
      } else {
        a = UA[w]; b = UB[w];
        const unsigned long long px = PX[w];
        const unsigned bA = gb + (unsigned)(px & 0xFFFF), bS = gb + nA + (unsigned)((px >> 16) & 0xFFFF),
                       bB = gb + nA + nS + (unsigned)((px >> 32) & 0xFFFF);
        rec = make_uint4(a, b, bA | (bS << 16), bB);
      }
      REC[w] = rec;
    }
    if (a | b) {
      const unsigned long long px = PX[w];
      const int pA = (int)(px & 0xFFFF), pS = nA + (int)((px >> 16) & 0xFFFF), pB = nA + nS + (int)((px >> 32) & 0xFFFF);
      unsigned xa = a & ~b, xs = a & b, xb = b & ~a;
      int q = pA;
      while (xa) { const int bt = __ffs(xa) - 1; xa &= xa - 1; gi[q++] = w * 32 + bt; }
      q = pS;
      while (xs) { const int bt = __ffs(xs) - 1; xs &= xs - 1; gi[q++] = w * 32 + bt; }
      q = pB;
      while (xb) { const int bt = __ffs(xb) - 1; xb &= xb - 1; gi[q++] = w * 32 + bt; }
    }
  }
  unsigned* M = Ms;
  const int nmw = (nd + ngt) * NP * 4;
  for (int xx = tid; xx < nmw; xx += NTH) M[xx] = 0u;
  for (int xx = ng + tid; xx < ngt * 128; xx += NTH) gi[xx] = 0;
  __syncthreads();
  auto slotf = [&](int j, int p, int& idx, unsigned& bit) {
    const uint4 r = REC[j >> 5];
    const int bb = j & 31;
    const unsigned lm = (1u << bb) - 1u;
    const bool ia = (r.x >> bb) & 1u, ib = (r.y >> bb) & 1u;
    const unsigned both = r.x & r.y;
    const unsigned cw = ia ? (ib ? both : (r.x & ~r.y)) : (r.y & ~r.x);
    const unsigned base = ia ? (ib ? (r.z >> 16) : (r.z & 0xFFFFu)) : r.w;
    const int gs = (int)base + __popc(cw & lm);
    const int mt = gs >> 7, pos = gs & 127;
    idx = (mt * NP + p) * 4 + ((pos >> 1) & 3);
    bit = 1u << (((pos >> 3) << 1) | (pos & 1));
  };
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int ch = r * NTH + tid;
    const int p = (ch * 8) / TOPK;
    const int lim = pf + p;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int j = id[r][e];
      if ((unsigned)j <= (unsigned)lim) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (sinkEnd > 0) {
    for (int xx = tid; xx < NP * sinkEnd; xx += NTH) {
      const int p = xx / sinkEnd, j = xx - p * sinkEnd;
      if (j <= pf + p) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (!dense && tl0 <= pl) {
    const int tlen = pl - tl0 + 1;
    for (int xx = tid; xx < NP * tlen; xx += NTH) {
      const int p = xx / tlen, j = tl0 + xx - p * tlen, i = pf + p;
      if (j <= i && (i - j < window || j < sink)) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (dense && (window > 0 || sink > 0)) {
    for (int xx = tid; xx < NP * nbq * 4; xx += NTH) {
      const int w4 = xx & 3, r = xx >> 2, bb = r % nbq, p = r / nbq;
      const int bt = blkTile[bb];
      if (bt < 0) continue;
      const int i = pf + p, kb0 = bb * 128;
      unsigned bits = 0;
      if (window > 0) bits |= perm_range(max(i - window + 1, 0) - kb0, i - kb0, w4);
      if (sink > 0) bits |= perm_range(-kb0, min(sink - 1, i) - kb0, w4);
      if (bits) atomicOr(M + (bt * NP + p) * 4 + w4, bits);
    }
  }
  __syncthreads();
  uint4* Mg = reinterpret_cast<uint4*>(P + offM);
  const uint4* M4 = reinterpret_cast<const uint4*>(Ms);
  for (int xx = tid; xx < nmw / 4; xx += NTH) Mg[xx] = M4[xx];
}

template <int MCAP>
__global__ void __launch_bounds__(256) plan3_k(const int* __restrict__ SEL, unsigned* __restrict__ PL, int* __restrict__ ctr,
    int T, int HKV, int G, int TOPK, int window, int sink, int THR, int nqblk, int HB, int planWords, int offG, int offM,
    float dfac, float splitThr, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int NTH = 256;
  extern __shared__ __align__(16) unsigned psm[];
  __shared__ unsigned long long s_red[NTH / 32];
  __shared__ int s_redi[NTH / 32];
  __shared__ int s_redi2[NTH / 32];
  pdl_trigger();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  if (it == 0 && tid == 0) *ctr = 0;
  const Item x = get_item(it, HB, HKV, nqblk, G, T * G, window, sink);
  const int NW = x.NW, NP = x.NP, pf = x.pf, sinkEnd = x.sinkEnd, nbq = x.nbq, tl0 = x.tl0, pl = x.pl;
  const int QA = 64 / G;
  const int* sb = SEL + ((size_t)(x.b * HKV + x.hk) * T + pf) * TOPK;
  const int nsel = NP * TOPK;
  for (int xx = (2 * NTH + tid) * 32; xx < nsel; xx += NTH * 32) prefetch_l2(sb + xx);
  unsigned* P = PL + (size_t)it * planWords;
  int* dBlk = reinterpret_cast<int*>(P + 8);
  int* gi = reinterpret_cast<int*>(P + offG);
  unsigned* UA = psm;
  unsigned* UB = psm + NW;
  unsigned long long* PX = reinterpret_cast<unsigned long long*>(psm + 2 * NW);
  uint4* REC = reinterpret_cast<uint4*>(psm);   // replaces UA/UB/PX after the scan
  int* blkTile = reinterpret_cast<int*>(psm + 4 * NW);
  unsigned* Ms = psm + ((4 * NW + nbq + 3) & ~3);
  const int aHi = min(pl, pf + QA - 1);
  for (int w = tid; w < NW; w += NTH) {
    const unsigned sk = sinkEnd > 0 ? range_bits(0, sinkEnd - 1, w) : 0u;
    UA[w] = sk | (tl0 <= aHi ? range_bits(tl0, aHi, w) : 0u);
    UB[w] = NP > QA ? (sk | (tl0 <= pl ? range_bits(tl0, pl, w) : 0u)) : 0u;
  }
  __syncthreads();
  fold_simple(sb, nsel, TOPK, pf, tid, NTH, [&](int j, int p) { atomicOr((p >= QA ? UB : UA) + (j >> 5), 1u << (j & 31)); });
  __syncthreads();
  // per-item choice: split the union between the two warpgroups only when they share few keys
  // (shared fraction <= splitThr); otherwise both compute every tile (no exclusive tiles)
  int utot = 0, uab = 0;
  for (int w = tid; w < NW; w += NTH) { const unsigned a = UA[w], b = UB[w]; utot += __popc(a | b); uab += __popc(a & b); }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) { utot += __shfl_xor_sync(0xffffffff, utot, off); uab += __shfl_xor_sync(0xffffffff, uab, off); }
  if (lane == 0) { s_redi[warp] = utot; s_redi2[warp] = uab; }
  __syncthreads();
  utot = 0; uab = 0;
#pragma unroll
  for (int k = 0; k < NTH / 32; ++k) { utot += s_redi[k]; uab += s_redi2[k]; }
  if ((float)uab > splitThr * (float)utot) {
    for (int w = tid; w < NW; w += NTH) { const unsigned u = UA[w] | UB[w]; UA[w] = u; UB[w] = u; }
    __syncthreads();
  }
  const bool dense = (float)nbq <= dfac * (float)(x.nband + (utot + 127) / 128);
  const int thrE = dense ? 1 : THR;
  const int bandBlk = (dense && window > 0) ? (x.wlo >> 7) : 0x7fffffff;
  unsigned long long carry = 0;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const unsigned a = w < NW ? UA[w] : 0u, b = w < NW ? UB[w] : 0u;
    const int pc = __popc(a | b);
    int cnt = pc + __shfl_xor_sync(0xffffffff, pc, 1);
    cnt += __shfl_xor_sync(0xffffffff, cnt, 2);
    const bool dn = w < NW && (cnt >= thrE || (w >> 2) >= bandBlk);
    const unsigned long long val = dn ? ((w & 3) == 0 ? (1ull << 48) : 0ull)
        : ((unsigned long long)__popc(a & ~b) | ((unsigned long long)__popc(a & b) << 16) |
           ((unsigned long long)__popc(b & ~a) << 32));
    unsigned long long xs = val;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      unsigned long long y = __shfl_up_sync(0xffffffff, xs, off);
      if (lane >= off) xs += y;
    }
    if (lane == 31) s_red[warp] = xs;
    __syncthreads();
    unsigned long long pre = carry, tot = 0;
#pragma unroll
    for (int k = 0; k < NTH / 32; ++k) { unsigned long long v = s_red[k]; if (k < warp) pre += v; tot += v; }
    const unsigned long long excl = pre + xs - val;
    if (w < NW) {
      if (dn) {
        if ((w & 3) == 0) { const int ti = (int)(excl >> 48); blkTile[w >> 2] = ti; dBlk[ti] = w >> 2; }
      } else if ((w & 3) == 0) blkTile[w >> 2] = -1;
      PX[w] = excl;
    }
    carry += tot;
    __syncthreads();
  }
  const int nd = (int)(carry >> 48), nA = (int)(carry & 0xFFFF), nS = (int)((carry >> 16) & 0xFFFF),
            nB = (int)((carry >> 32) & 0xFFFF);
  const int ng = nA + nS + nB, ngt = (ng + 127) >> 7;
  if (tid == 0) { P[0] = nd; P[1] = ng; P[2] = dense ? 1u : 0u; P[3] = nA; P[4] = nS; }
  const int gb = nd * 128;
  constexpr int WPT = 8;   // words per thread (NW <= 2048)
  uint4 recs[WPT];
#pragma unroll
  for (int r = 0; r < WPT; ++r) {
    const int w = r * NTH + tid;
    unsigned a = 0, b = 0;
    recs[r] = make_uint4(0u, 0u, 0u, 0u);
    if (w < NW) {
      const int bt = blkTile[w >> 2];
      if (bt >= 0) {
        recs[r] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, (unsigned)(bt * 128 + (w & 3) * 32) << 16, 0u);
      } else {
        a = UA[w]; b = UB[w];
        const unsigned long long px = PX[w];
        const int pA = (int)(px & 0xFFFF), pS = nA + (int)((px >> 16) & 0xFFFF), pB = nA + nS + (int)((px >> 32) & 0xFFFF);
        recs[r] = make_uint4(a, b, (unsigned)(gb + pA) | ((unsigned)(gb + pS) << 16), (unsigned)(gb + pB));
        unsigned xa = a & ~b, xs = a & b, xb = b & ~a;
        int q = pA;
        while (xa) { const int bq = __ffs(xa) - 1; xa &= xa - 1; gi[q++] = w * 32 + bq; }
        q = pS;
        while (xs) { const int bq = __ffs(xs) - 1; xs &= xs - 1; gi[q++] = w * 32 + bq; }
        q = pB;
        while (xb) { const int bq = __ffs(xb) - 1; xb &= xb - 1; gi[q++] = w * 32 + bq; }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < WPT; ++r) {
    const int w = r * NTH + tid;
    if (w < NW) REC[w] = recs[r];
  }
  const bool mg = nd + ngt > MCAP;
  unsigned* M = mg ? (P + offM) : Ms;
  const int nmw = (nd + ngt) * NP * 4;
  const int MT = nd + ngt;
  for (int xx = tid; xx < nmw; xx += NTH) M[xx] = 0u;
  for (int xx = ng + tid; xx < ngt * 128; xx += NTH) gi[xx] = 0;
  __syncthreads();
  auto slotf = [&](int j, int p, int& idx, unsigned& bit) {
    const uint4 r = REC[j >> 5];
    const int bb = j & 31;
    const unsigned lm = (1u << bb) - 1u;
    const bool ia = (r.x >> bb) & 1u, ib = (r.y >> bb) & 1u;
    const unsigned both = r.x & r.y;
    const unsigned cw = ia ? (ib ? both : (r.x & ~r.y)) : (r.y & ~r.x);
    const unsigned base = ia ? (ib ? (r.z >> 16) : (r.z & 0xFFFFu)) : r.w;
    const int gs = (int)base + __popc(cw & lm);
    const int mt = gs >> 7, pos = gs & 127;
    idx = (mg ? (mt * NP + p) : (p * MT + mt)) * 4 + ((pos >> 1) & 3);
    bit = 1u << (((pos >> 3) << 1) | (pos & 1));
  };
  fold_simple(sb, nsel, TOPK, pf, tid, NTH, [&](int j, int p) { int idx; unsigned bit; slotf(j, p, idx, bit); atomicOr(M + idx, bit); });
  if (sinkEnd > 0) {
    for (int xx = tid; xx < NP * sinkEnd; xx += NTH) {
      const int p = xx / sinkEnd, j = xx - p * sinkEnd;
      if (j <= pf + p) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (!dense && tl0 <= pl) {
    const int tlen = pl - tl0 + 1;
    for (int xx = tid; xx < NP * tlen; xx += NTH) {
      const int p = xx / tlen, j = tl0 + xx - p * tlen, i = pf + p;
      if (j <= i && (i - j < window || j < sink)) {
        int idx; unsigned bit;
        slotf(j, p, idx, bit);
        atomicOr(M + idx, bit);
      }
    }
  }
  if (dense && (window > 0 || sink > 0)) {
    for (int xx = tid; xx < NP * nbq * 4; xx += NTH) {
      const int w4 = xx & 3, r = xx >> 2, bb = r % nbq, p = r / nbq;
      const int bt = blkTile[bb];
      if (bt < 0) continue;
      const int i = pf + p, kb0 = bb * 128;
      unsigned bits = 0;
      if (window > 0) bits |= perm_range(max(i - window + 1, 0) - kb0, i - kb0, w4);
      if (sink > 0) bits |= perm_range(-kb0, min(sink - 1, i) - kb0, w4);
      if (bits) atomicOr(M + (mg ? (bt * NP + p) : (p * MT + bt)) * 4 + w4, bits);
    }
  }
  if (!mg) {
    __syncthreads();
    uint4* Mg = reinterpret_cast<uint4*>(P + offM);
    const uint4* M4 = reinterpret_cast<const uint4*>(Ms);
    for (int xx = tid; xx < nmw / 4; xx += NTH) { const int mt = xx / NP, p = xx - mt * NP; Mg[xx] = M4[p * MT + mt]; }
  }
}

struct Seq { int nbandE, nd, ngt, eA, aEnd, eB, nm, mn, ntile, S, rem; };
__device__ __forceinline__ Seq make_seq(int nbandE, int nd, int ng, int nA, int nS) {
  Seq s;
  s.nbandE = nbandE; s.nd = nd; s.ngt = (ng + 127) >> 7;
  s.eA = nA >> 7; s.aEnd = (nA + nS + 127) >> 7;
  s.eB = s.ngt - s.aEnd; s.nm = s.aEnd - s.eA; s.mn = min(s.eA, s.eB);
  s.ntile = nbandE + nd + s.ngt;
  s.S = nbandE + nd + s.nm;
  s.rem = s.eA > s.eB ? 0 : (s.eB > s.eA ? 1 : -1);
  return s;
}
// position t of the load sequence -> j: band index (t < nbandE), dense index, or gather tile; own: 1 A, 2 B, 3 both
__device__ __forceinline__ void seq_at(const Seq& s, int t, int& j, int& own) {
  own = 3;
  if (t < s.nbandE) { j = t; return; }
  if (t < s.nbandE + s.nd) { j = t - s.nbandE; return; }
  const int u = t - s.nbandE - s.nd;
  if (u < s.nm) { j = s.eA + u; return; }
  const int q = u - s.nm;
  if (q < 2 * s.mn) {
    if (q & 1) { j = s.aEnd + (q >> 1); own = 2; } else { j = q >> 1; own = 1; }
    return;
  }
  const int r = q - 2 * s.mn;
  if (s.eA > s.eB) { j = s.mn + r; own = 1; } else { j = s.aEnd + s.mn + r; own = 2; }
}
__device__ __forceinline__ void mbar_arrive_cnt(unsigned addr, unsigned cnt) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" :: "r"(addr), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive_warp(unsigned addr, unsigned cnt) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) mbar_arrive_cnt(addr, 32u * cnt);
}

template <int D>
struct KCfg {
  static constexpr int BM = 128, BN = 128;
  static constexpr int NSK = D == 128 ? 2 : 4, NSV = D == 128 ? NSV128 : 4;
  static constexpr int QB = BM * D * 2, KB = BN * D * 2;
  static constexpr int SMEM = 2 * QB + (NSK + NSV) * KB + 1024;
};

template <int D>
__global__ void __launch_bounds__(384, 1) attn_pers(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmQ,
    const bf16* __restrict__ Q, const bf16* __restrict__ Kp, const bf16* __restrict__ Vp,
    const unsigned* __restrict__ PL, int* __restrict__ ctr, bf16* __restrict__ O,
    float* __restrict__ LSE, int B, int T, int HQ, int HKV, int window, int sink,
    float sl2, int nqblk, int planWords, int offG, int offM, int useQT, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int BM = 128, BN = 128, CPR = D / 8, NO = D / 2, RPI = 128 / CPR;
  constexpr int NSK = KCfg<D>::NSK, NSV = KCfg<D>::NSV, QB = KCfg<D>::QB, KB = KCfg<D>::KB;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * NSK + 2 * NSV + 6];
  __shared__ int slot[2][6];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const float NEG = -INFINITY;
  auto fullK = [&](int s) { return bar0 + 8 * s; };
  auto emptyK = [&](int s) { return bar0 + 8 * (NSK + s); };
  auto fullV = [&](int s) { return bar0 + 8 * (2 * NSK + s); };
  auto emptyV = [&](int s) { return bar0 + 8 * (2 * NSK + NSV + s); };
  auto qfull = [&](int b) { return bar0 + 8 * (2 * NSK + 2 * NSV + b); };
  auto qempty = [&](int b) { return bar0 + 8 * (2 * NSK + 2 * NSV + 2 + b); };
  auto qinfo = [&](int b) { return bar0 + 8 * (2 * NSK + 2 * NSV + 4 + b); };
  auto sQb = [&](int b) { return sbase + b * QB; };
  auto sK = [&](int s) { return sbase + 2 * QB + s * KB; };
  auto sV = [&](int s) { return sbase + 2 * QB + (NSK + s) * KB; };

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int HB = HKV * B;
  const int G = HQ / HKV, TG = T * G;
  const int nItems = nqblk * HB;

  if (tid == 0) {
    for (int s = 0; s < NSK; ++s) { mbar_init(fullK(s), 64); mbar_init(emptyK(s), 256); }
    for (int s = 0; s < NSV; ++s) { mbar_init(fullV(s), 64); mbar_init(emptyV(s), 256); }
    for (int b = 0; b < 2; ++b) { mbar_init(qfull(b), 128); mbar_init(qempty(b), 256); mbar_init(qinfo(b), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp >= 8) {
    reg_dealloc<104>();
    const int ptid = tid - 256;
    int gt = 0;
    int it = blockIdx.x;
    int nextIt = -1, tick = -1, hnd = 0, hng = 0, hdm = 0, hnA = 0, hnS = 0;
    const int cc = ptid % CPR, rr = ptid / CPR;
    const unsigned soff0 = (cc >> 3) * (BN * 128) + rr * 128 + (((cc & 7) ^ (rr & 7)) << 4);
    const unsigned long long pol = make_pol_last();
    for (int n = 0;; ++n) {
      const int qb = n & 1;
      if (n >= 2) mbar_wait(qempty(qb), ((n >> 1) - 1) & 1);
      if (n > 0) {
        if (ptid == 0) {
          volatile int* sw = slot[qb];
          sw[0] = nextIt; sw[1] = hnd; sw[2] = hng; sw[3] = hdm; sw[4] = hnA; sw[5] = hnS;
          mbar_arrive(qinfo(qb));
        }
        bar_sync(3, 128);
        it = slot[qb][0];
        if (it < 0) break;
      }
      const Item x = get_item(it, HB, HKV, nqblk, G, TG, window, sink);
      const unsigned sQ = sQb(qb);
      for (int xx = ptid; xx < BM * CPR; xx += 128) {
        const int r = xx / CPR, c2 = xx % CPR;
        const int rg = min(x.R0 + r, TG - 1);
        const int i = rg / G, g = rg - i * G;
        const bf16* src = Q + ((size_t)(x.b * T + i) * HQ + x.hk * G + g) * D + c2 * 8;
        const unsigned dst = sQ + (c2 >> 3) * (BM * 128) + r * 128 + (((c2 & 7) ^ (r & 7)) << 4);
        cp_async16s(dst, src);
      }
      cp_mbar_arrive(qfull(qb));
      if (n == 0) {
        pdl_wait();
        if (ptid == 0) {
          const unsigned* P0 = PL + (size_t)it * planWords;
          volatile int* sw = slot[0];
          sw[0] = it; sw[1] = __ldg(P0); sw[2] = __ldg(P0 + 1); sw[3] = __ldg(P0 + 2); sw[4] = __ldg(P0 + 3); sw[5] = __ldg(P0 + 4);
          mbar_arrive(qinfo(0));
        }
        bar_sync(3, 128);
      }
      const int nd = slot[qb][1], ng = slot[qb][2], dm = slot[qb][3], nA = slot[qb][4], nS = slot[qb][5];
      if (ptid == 0) tick = (int)gridDim.x + atomicAdd(ctr, 1);
      const unsigned* P = PL + (size_t)it * planWords;
      const int nbandE = dm ? 0 : x.nband;
      const Seq sq = make_seq(nbandE, nd, ng, nA, nS);
      const int* dBlk = reinterpret_cast<const int*>(P + 8);
      const int* gi = reinterpret_cast<const int*>(P + offG) + rr;
      const int ntile = sq.ntile;
      const int g0 = nbandE + nd;
      const int tpre = min(2, ntile - 1);
      // two independent streams: threads 0-63 load K tiles, 64-127 load V tiles
      constexpr int RP = 64 / CPR, NPASS = 128 / RP;
      const int strm = ptid >> 6, sp = ptid & 63;
      const int cc2 = sp % CPR, rr2 = sp / CPR;
      // thread (cc2, rr2) copies chunk cc2 of rows NPASS*rr2 .. NPASS*rr2+NPASS-1: its row ids are contiguous
      // (vector loads) and the row's swizzle phase is the compile-time i & 7
      const unsigned sob = (cc2 >> 3) * (BN * 128) + rr2 * NPASS * 128;
      const unsigned cx = (unsigned)(cc2 & 7);
      const int* gib = reinterpret_cast<const int*>(P + offG) + rr2 * NPASS;
      int4 jv[NPASS / 4];
      if (sq.ngt > 0 && g0 < ntile) {
        int j0, o0;
        seq_at(sq, g0, j0, o0);
#pragma unroll
        for (int i2 = 0; i2 < NPASS / 4; ++i2) jv[i2] = __ldg(reinterpret_cast<const int4*>(gib + j0 * 128) + i2);
      }
      const char* Xb = reinterpret_cast<const char*>((strm == 0 ? Kp : Vp) + ((size_t)x.b * T * HKV + x.hk) * D + cc2 * 8);
      const CUtensorMap* tmX = strm == 0 ? &tmK : &tmV;
      for (int t = 0; t < ntile; ++t) {
        const int gg = gt + t;
        const int ns = strm == 0 ? NSK : NSV;
        const int sx = gg % ns, ux = gg / ns;
        const unsigned fullX = strm == 0 ? fullK(sx) : fullV(sx);
        const unsigned emptyX = strm == 0 ? emptyK(sx) : emptyV(sx);
        const unsigned sX = strm == 0 ? sK(sx) : sV(sx);
        if (t == tpre && ptid == 0) {
          nextIt = tick < nItems ? tick : -1;
          if (nextIt >= 0) {
            const unsigned* P2 = PL + (size_t)nextIt * planWords;
            hnd = __ldg(P2); hng = __ldg(P2 + 1); hdm = __ldg(P2 + 2); hnA = __ldg(P2 + 3); hnS = __ldg(P2 + 4);
            if (useQT) {
              const Item xn = get_item(nextIt, HB, HKV, nqblk, G, TG, window, sink);
#pragma unroll
              for (int p = 0; p < D / 64; ++p) tma_prefetch_4d(&tmQ, p * 64, xn.hk * G, xn.pf, xn.b);
            }
          }
        }
        if (ux > 0) mbar_wait(emptyX, (ux - 1) & 1);
        if (t < g0) {
          const int kd = t < nbandE ? x.wlo + 128 * t : __ldg(dBlk + t - nbandE) * 128;
          if (sp == 0) {
            mbar_expect_tx(fullX, KB);
#pragma unroll
            for (int p = 0; p < D / 64; ++p) tma_load_4d(sX + p * BN * 128, tmX, p * 64, x.hk, kd, x.b, fullX);
          } else mbar_arrive(fullX);
        } else {
#pragma unroll
          for (int i2 = 0; i2 < NPASS; ++i2) {
            const int4 v4 = jv[i2 >> 2];
            const int id = (i2 & 3) == 0 ? v4.x : (i2 & 3) == 1 ? v4.y : (i2 & 3) == 2 ? v4.z : v4.w;
            cp_async16h(sX + sob + i2 * 128 + ((cx ^ (unsigned)(i2 & 7)) << 4), Xb + (unsigned)id * (unsigned)(HKV * D * 2), pol);
          }
          cp_mbar_arrive(fullX);
          if (t + 1 < ntile) {
            int j1, o1;
            seq_at(sq, t + 1, j1, o1);
#pragma unroll
            for (int i2 = 0; i2 < NPASS / 4; ++i2) jv[i2] = __ldg(reinterpret_cast<const int4*>(gib + j1 * 128) + i2);
          }
        }
      }
      gt += ntile;
    }
    cp_wait_all();
    return;
  }
  reg_alloc<200>();
  pdl_wait();

  const int c = warp >> 2;
  const int wq = warp & 3;
  const int wtid = tid & 127;
  const int qd = lane & 3;
  const int r0 = c * 64 + wq * 16 + (lane >> 2);
  const int sh = 2 * qd;
  const int my_bar = 1 + c, other_bar = 2 - c;
  const unsigned DHI = 64u | (1u << 30);
  const unsigned qlo0 = (((sQb(0) + c * 64 * 128) >> 4) & 0x3FFF) | (1u << 16);
  const unsigned qlo1 = (((sQb(1) + c * 64 * 128) >> 4) & 0x3FFF) | (1u << 16);
  unsigned klo[NSK], vlo[NSV];
#pragma unroll
  for (int s2 = 0; s2 < NSK; ++s2) klo[s2] = ((sK(s2) >> 4) & 0x3FFF) | (1u << 16);
#pragma unroll
  for (int s2 = 0; s2 < NSV; ++s2) vlo[s2] = ((sV(s2) >> 4) & 0x3FFF) | ((unsigned)((BN * 128) >> 4) << 16);

  float o[NO];
  float s[64];
  unsigned p[8][4];
  float m0, m1, l0, l1;
  unsigned sQ = sQb(0);
  unsigned qlo = qlo0;

  auto issue_S = [&](int st) {
    unsigned kl = klo[0];
#pragma unroll
    for (int s2 = 1; s2 < NSK; ++s2) kl = (st == s2) ? klo[s2] : kl;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      const unsigned off = (kk >> 2) * 1024 + (kk & 3) * 2;
      wgmma_ss_n128(s, mkd(qlo + off, DHI), mkd(kl + off, DHI), kk > 0 ? 1 : 0);
    }
  };
  auto issue_PV = [&](int st) {
    unsigned vl = vlo[0];
#pragma unroll
    for (int s2 = 1; s2 < NSV; ++s2) vl = (st == s2) ? vlo[s2] : vl;
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
      const unsigned long long db = mkd(vl + kk * 128, DHI);
      if constexpr (D == 128) wgmma_rs_n128(o, p[kk], db, 1);
      else wgmma_rs_n64(o, p[kk], db, 1);
    }
  };
  auto softmax = [&](const unsigned x0, const unsigned x1, float& c0, float& c1) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      s[4 * n] = (x0 & (1u << (2 * n))) ? s[4 * n] : NEG;
      s[4 * n + 1] = (x0 & (2u << (2 * n))) ? s[4 * n + 1] : NEG;
      s[4 * n + 2] = (x1 & (1u << (2 * n))) ? s[4 * n + 2] : NEG;
      s[4 * n + 3] = (x1 & (2u << (2 * n))) ? s[4 * n + 3] : NEG;
    }
    float ma[4], mb[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { ma[k] = fmaxf(s[4 * k], s[4 * k + 1]); mb[k] = fmaxf(s[4 * k + 2], s[4 * k + 3]); }
#pragma unroll
    for (int n = 4; n < 16; ++n) {
      ma[n & 3] = fmaxf(ma[n & 3], fmaxf(s[4 * n], s[4 * n + 1]));
      mb[n & 3] = fmaxf(mb[n & 3], fmaxf(s[4 * n + 2], s[4 * n + 3]));
    }
    float mx0 = fmaxf(fmaxf(ma[0], ma[1]), fmaxf(ma[2], ma[3]));
    float mx1 = fmaxf(fmaxf(mb[0], mb[1]), fmaxf(mb[2], mb[3]));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const bool k0 = (mx0 - m0) * sl2 <= 8.f, k1 = (mx1 - m1) * sl2 <= 8.f;
    const float mn0 = k0 ? m0 : fmaxf(m0, mx0), mn1 = k1 ? m1 : fmaxf(m1, mx1);
    const float mu0 = mn0 == NEG ? 0.f : mn0 * sl2, mu1 = mn1 == NEG ? 0.f : mn1 * sl2;
    c0 = k0 ? 1.f : exp2f(m0 * sl2 - mu0); c1 = k1 ? 1.f : exp2f(m1 * sl2 - mu1);
    m0 = mn0; m1 = mn1;
    float sa[4], sb[4];
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      s[4 * n] = exp2f(fmaf(s[4 * n], sl2, -mu0));
      s[4 * n + 1] = exp2f(fmaf(s[4 * n + 1], sl2, -mu0));
      s[4 * n + 2] = exp2f(fmaf(s[4 * n + 2], sl2, -mu1));
      s[4 * n + 3] = exp2f(fmaf(s[4 * n + 3], sl2, -mu1));
      if (n < 4) { sa[n] = s[4 * n] + s[4 * n + 1]; sb[n] = s[4 * n + 2] + s[4 * n + 3]; }
      else { sa[n & 3] += s[4 * n] + s[4 * n + 1]; sb[n & 3] += s[4 * n + 2] + s[4 * n + 3]; }
    }
    l0 = l0 * c0 + ((sa[0] + sa[1]) + (sa[2] + sa[3]));
    l1 = l1 * c1 + ((sb[0] + sb[1]) + (sb[2] + sb[3]));
  };
  auto pack_p = [&]() {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      p[n >> 1][(n & 1) * 2 + 0] = pack_bf16(s[4 * n], s[4 * n + 1]);
      p[n >> 1][(n & 1) * 2 + 1] = pack_bf16(s[4 * n + 2], s[4 * n + 3]);
    }
  };

  int gt = 0;
  for (int n = 0;; ++n) {
    const int qb = n & 1;
    mbar_wait(qinfo(qb), (n >> 1) & 1);
    volatile int* sl = slot[qb];
    const int it = sl[0], nd = sl[1], ng = sl[2], dm = sl[3], nA = sl[4], nS = sl[5];
    if (it < 0) break;
    const Item x = get_item(it, HB, HKV, nqblk, G, TG, window, sink);
    const unsigned* P = PL + (size_t)it * planWords;
    const int nband = dm ? 0 : x.nband, wlo = x.wlo, pf = x.pf, NP = x.NP;
    const Seq sq = make_seq(nband, nd, ng, nA, nS);
    const int ntile = sq.ntile;
    const int myown = 1 << c;
    const unsigned* MW = P + offM + qd;
    const int rg0 = x.R0 + r0, rg1 = rg0 + 8;
    const int i0 = min(rg0, TG - 1) / G, i1 = min(rg1, TG - 1) / G;
    sQ = sQb(qb);
    qlo = qb ? qlo1 : qlo0;

    auto arith = [&](int i, int kb0) -> unsigned {
      const int lo = max(i - window + 1, kb0) - kb0, hi = min(i, kb0 + 127) - kb0;
      const int shi = min(sink - 1, i) - kb0;
      return perm_range(lo, hi, qd) | perm_range(0, shi, qd);
    };
    // closed forms of this warpgroup's own positions (c = 0: A, c = 1: B)
    auto own_mt = [&](int t) -> int {   // mask tile index of own position t >= nband
      if (t < nband + nd) return t - nband;
      if (t < sq.S) return sq.eA + t - nband;   // middle gather tile eA + u, mask index nd + eA + u
      const int q = t - sq.S;
      const int base = c ? sq.aEnd : 0;
      return nd + base + (q < 2 * sq.mn ? (q >> 1) : sq.mn + (q - 2 * sq.mn));
    };
    auto tile_mask = [&](int t, unsigned& a, unsigned& bm) {
      if (t >= ntile) return;
      if (t < nband) {
        const int kb0 = wlo + t * 128;
        a = arith(i0, kb0); bm = arith(i1, kb0);
      } else {
        const int mt = own_mt(t);
        const unsigned* mm = MW + (size_t)mt * NP * 4;
        a = __ldg(mm + (i0 - pf) * 4); bm = __ldg(mm + (i1 - pf) * 4);
      }
    };
    auto next_own = [&](int t) -> int {
      if (t < sq.S) return t;
      int q = t - sq.S;
      if (q < 2 * sq.mn) {
        const int q2 = q + ((q ^ c) & 1);
        if (q2 < 2 * sq.mn) return sq.S + q2;
        q = 2 * sq.mn;
      }
      return (sq.rem == c && sq.S + q < ntile) ? sq.S + q : ntile;
    };

#pragma unroll
    for (int k2 = 0; k2 < NO; ++k2) o[k2] = 0.f;
    m0 = NEG; m1 = NEG; l0 = 0.f; l1 = 0.f;
    float c0, c1;
    unsigned mw0 = 0, mw1 = 0, nw0 = 0, nw1 = 0;
    tile_mask(next_own(0), mw0, mw1);
    mbar_wait(qfull(qb), (n >> 1) & 1);
    fence_proxy_async();
    // own positions are at most 2 apart (sequence design), so deferring PV by one own tile never
    // blocks the producer (NSV >= 2)
    int tc = next_own(0);
    if (tc < ntile) {
      bool ex = tc >= sq.S;
      int tn = next_own(tc + 1);
      tile_mask(tn, nw0, nw1);
      {
        const int g = gt + tc, s0 = g % NSK;
        mbar_wait(fullK(s0), (g / NSK) & 1);
        fence_proxy_async();
        wg_fence();
        issue_S(s0);
        wg_commit();
        wg_wait<0>();
#pragma unroll
        for (int k2 = 0; k2 < 64; ++k2) fence_reg(s[k2]);
        mbar_arrive_warp(emptyK(s0), ex ? 2u : 1u);
        softmax(mw0, mw1, c0, c1);
        pack_p();
        mw0 = nw0; mw1 = nw1;
      }
      while (tn < ntile) {
        const int tp = tc;
        const bool pex = ex;
        tc = tn;
        ex = tc >= sq.S;
        tn = next_own(tc + 1);
        tile_mask(tn, nw0, nw1);
        const int g = gt + tc, st = g % NSK, gp = gt + tp, sp = gp % NSV;
        mbar_wait(fullK(st), (g / NSK) & 1);
        mbar_wait(fullV(sp), (gp / NSV) & 1);
        fence_proxy_async();
        wg_fence();
        issue_S(st);
        wg_commit();
        issue_PV(sp);
        wg_commit();
        wg_wait<1>();
#pragma unroll
        for (int k2 = 0; k2 < 64; ++k2) fence_reg(s[k2]);
        mbar_arrive_warp(emptyK(st), ex ? 2u : 1u);
        softmax(mw0, mw1, c0, c1);
        wg_wait<0>();
#pragma unroll
        for (int k2 = 0; k2 < NO; ++k2) fence_reg(o[k2]);
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2)
#pragma unroll
          for (int j2 = 0; j2 < 4; ++j2) fence_ureg(p[k2][j2]);
        mbar_arrive_warp(emptyV(sp), pex ? 2u : 1u);
        if (__any_sync(0xffffffff, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
          for (int k2 = 0; k2 < NO / 4; ++k2) {
            o[4 * k2] *= c0; o[4 * k2 + 1] *= c0; o[4 * k2 + 2] *= c1; o[4 * k2 + 3] *= c1;
          }
        }
        pack_p();
        mw0 = nw0; mw1 = nw1;
      }
      {
        const int gl = gt + tc, sl2v = gl % NSV;
        mbar_wait(fullV(sl2v), (gl / NSV) & 1);
        fence_proxy_async();
        wg_fence();
        issue_PV(sl2v);
        wg_commit();
        wg_wait<0>();
#pragma unroll
        for (int k2 = 0; k2 < NO; ++k2) fence_reg(o[k2]);
        mbar_arrive_warp(emptyV(sl2v), ex ? 2u : 1u);
      }
    }
    gt += ntile;
    float L0 = l0, L1 = l1;
    L0 += __shfl_xor_sync(0xffffffff, L0, 1);
    L0 += __shfl_xor_sync(0xffffffff, L0, 2);
    L1 += __shfl_xor_sync(0xffffffff, L1, 1);
    L1 += __shfl_xor_sync(0xffffffff, L1, 2);
    const float inv0 = L0 > 0.f ? 1.f / L0 : 0.f, inv1 = L1 > 0.f ? 1.f / L1 : 0.f;
    {
      const int ra = r0, rb = r0 + 8;
#pragma unroll
      for (int k2 = 0; k2 < D / 8; ++k2) {
        const unsigned pa = sQ + (k2 >> 3) * (BM * 128) + ra * 128 + (((k2 & 7) ^ (ra & 7)) << 4) + sh * 2;
        const unsigned pb = sQ + (k2 >> 3) * (BM * 128) + rb * 128 + (((k2 & 7) ^ (rb & 7)) << 4) + sh * 2;
        st_shared_u32(pa, pack_bf16(o[4 * k2] * inv0, o[4 * k2 + 1] * inv0));
        st_shared_u32(pb, pack_bf16(o[4 * k2 + 2] * inv1, o[4 * k2 + 3] * inv1));
      }
    }
    if ((lane & 3) == 0) {
      if (rg0 < TG) {
        const int i = rg0 / G, g = rg0 - i * G;
        LSE[((size_t)x.b * HQ + x.hk * G + g) * T + i] = L0 > 0.f ? (m0 * sl2 + log2f(L0)) * 0.69314718055994531f : NEG;
      }
      if (rg1 < TG) {
        const int i = rg1 / G, g = rg1 - i * G;
        LSE[((size_t)x.b * HQ + x.hk * G + g) * T + i] = L1 > 0.f ? (m1 * sl2 + log2f(L1)) * 0.69314718055994531f : NEG;
      }
    }
    bar_sync(5 + c, 128);
#pragma unroll
    for (int j = 0; j < (64 * CPR) / 128; ++j) {
      const int xx = j * 128 + wtid;
      const int lr = xx / CPR, ch = xx % CPR;
      const int r = c * 64 + lr;
      const int rg = x.R0 + r;
      if (rg < TG) {
        const int i = rg / G, g = rg - i * G;
        const uint4 v = ld_shared_v4(sQ + (ch >> 3) * (BM * 128) + r * 128 + (((ch & 7) ^ (r & 7)) << 4));
        *reinterpret_cast<uint4*>(O + ((size_t)(x.b * T + i) * HQ + x.hk * G + g) * D + ch * 8) = v;
      }
    }
    mbar_arrive(qempty(qb));
  }
}
"""

HOST = r"""
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
struct MapCache { void* p = nullptr; int B = 0, T = 0, H = 0, D = 0, bh = 0, bt = 0; CUtensorMap m; };
static const CUtensorMap& get_map(MapCache& c, void* ptr, int B, int T, int H, int D, int bh, int bt) {
  if (c.p == ptr && c.B == B && c.T == T && c.H == H && c.D == D && c.bh == bh && c.bt == bt) return c.m;
  cuuint64_t dims[4] = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T, (cuuint64_t)B};
  cuuint64_t strides[3] = {(cuuint64_t)D * 2, (cuuint64_t)H * D * 2, (cuuint64_t)T * H * D * 2};
  cuuint32_t box[4] = {64, (cuuint32_t)bh, (cuuint32_t)bt, 1};
  cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult r = get_encode()(&c.m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "tensor map encode failed ", (int)r);
  c.p = ptr; c.B = B; c.T = T; c.H = H; c.D = D; c.bh = bh; c.bt = bt;
  return c.m;
}
static int num_sms() {
  static int n = 0;
  if (!n) {
    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}
static int* get_ctr() {
  static int* c = nullptr;
  if (!c) { cudaMalloc(&c, 64); cudaMemset(c, 0, 64); }
  return c;
}
template <int D>
void launch_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, const unsigned* pl, int* ctr, torch::Tensor o, torch::Tensor lse,
                 int window, int sink, float sl2, int planWords, int offG, int offM, int grid, cudaStream_t stream, const int* gate, int gwant) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), HKV = k.size(2);
  const int G = HQ / HKV;
  constexpr int SM = KCfg<D>::SMEM;
  static bool init = false;
  if (!init) {
    cudaError_t ea = cudaFuncSetAttribute(attn_pers<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    TORCH_CHECK(ea == cudaSuccess, "smem attr failed ", SM, " ", cudaGetErrorString(ea));
    init = true;
  }
  MapCache ck, cv, cq;   // encoded on every call (no pointer-keyed caching)
  const CUtensorMap& tk = get_map(ck, k.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tv = get_map(cv, v.data_ptr(), B, T, HKV, D, 1, 128);
  const int useQT = (128 % G == 0) ? 1 : 0;
  const CUtensorMap& tq = useQT ? get_map(cq, q.data_ptr(), B, T, HQ, D, G, 128 / G) : tk;
  const int nqblk = (T * G + 127) / 128;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, attn_pers<D>, tk, tv, tq,
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()), pl, ctr,
      reinterpret_cast<bf16*>(o.data_ptr()), lse.data_ptr<float>(), B, T, HQ, HKV, window, sink, sl2,
      nqblk, planWords, offG, offM, useQT, gate, gwant);
  TORCH_CHECK(e == cudaSuccess, "launch failed ", cudaGetErrorString(e));
}
}  // namespace
void sparse_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel,
                torch::Tensor o, torch::Tensor lse, int64_t window, int64_t sink, double scale, int64_t thr, double dfac, double split_thr, torch::Tensor gate_t, int64_t want) {
  const int* gate = gate_t.numel() ? gate_t.data_ptr<int>() : nullptr;
  const int B = q.size(0), T = q.size(1), D = q.size(3), HQ = q.size(2);
  const int HKV = k.size(2), TOPK = sel.size(3);
  const int G = HQ / HKV;
  const int nqblk = (T * G + 127) / 128;
  const int HB = HKV * B;
  const long nItems = (long)nqblk * HB;
  const int grid = (int)std::min<long>(nItems, num_sms());
  const long nb = (T + 127) / 128;
  const long NPM = 127 / G + 2;
  const int THR = (int)std::max<int64_t>(1, thr);
  long ngmax = std::min<long>(nb * 128, NPM * TOPK + sink + 128);
  long ngtmax = (ngmax + 127) / 128;
  const long nbandmax = window > 0 ? (window + NPM) / 128 + 2 : 0;
  const long tm1 = ngmax / THR + 2;
  const long tm2 = (long)std::ceil(dfac * (double)(nbandmax + ngtmax)) + 2;
  long tilesmax = std::min<long>(nb, std::max<long>(tm1, tm2));
  const long offG = (8 + nb + 3) & ~3L;
  TORCH_CHECK(G == 8 && 8L * TOPK + window + 64 < 65536, "split plan needs G == 8 and 8*TOPK < 65536");
  const long offM = offG + ngtmax * 128;
  const long maskWords = tilesmax * NPM * 4;
  const long planWords = (offM + maskWords + 3) & ~3L;
  auto plan = torch::empty({nItems * planWords}, q.options().dtype(torch::kInt32));
  unsigned* pl = reinterpret_cast<unsigned*>(plan.data_ptr());
  int* ctr = get_ctr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long smw = ((4 * nb * 4 + nb + 3) & ~3L);
  const long psmB = (smw + maskWords) * 4;
  // mask staging capacity (tiles): long sequences have bigger unions; items over capacity fall back to
  // global-memory atomics, which is slow
  const int MCAP = T > 40000 ? 128 : 104;
  const long smw3 = ((4 * nb * 4 + nb + 3) & ~3L) + MCAP * 16 * 4;
  const long psm3 = smw3 * 4;
  if (TOPK % 8 == 0 && nb * 4 <= 2048 && 16L * TOPK + 2 * window + 1024 < 65536 && psm3 <= 200 * 1024) {
    auto kern = MCAP == 128 ? plan3_k<128> : plan3_k<104>;
    static long pset3[2] = {0, 0};
    const int si = MCAP == 128 ? 1 : 0;
    if (psm3 > pset3[si]) {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)psm3);
      pset3[si] = psm3;
    }
    kern<<<(int)nItems, 256, psm3, stream>>>(sel.data_ptr<int>(), pl, ctr, T, HKV, G, TOPK, (int)window, (int)sink,
        THR, nqblk, HB, (int)planWords, (int)offG, (int)offM, (float)dfac, (float)split_thr, gate, (int)want);
  } else if (psmB <= 200 * 1024) {
    static long pset = 0;
    if (psmB > pset) {
      cudaFuncSetAttribute(plan_k<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)psmB);
      pset = psmB;
    }
    plan_k<false><<<(int)nItems, 256, psmB, stream>>>(sel.data_ptr<int>(), pl, ctr, T, HKV, G, TOPK, (int)window, (int)sink,
        THR, nqblk, HB, (int)planWords, (int)offG, (int)offM, (float)dfac, gate, (int)want);
  } else {
    const long b2 = smw * 4;
    static long pset2 = 0;
    if (b2 > pset2) {
      cudaFuncSetAttribute(plan_k<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b2);
      pset2 = b2;
    }
    plan_k<true><<<(int)nItems, 256, b2, stream>>>(sel.data_ptr<int>(), pl, ctr, T, HKV, G, TOPK, (int)window, (int)sink,
        THR, nqblk, HB, (int)planWords, (int)offG, (int)offM, (float)dfac, gate, (int)want);
  }
  const float sl2 = (float)(scale * 1.4426950408889634);
  if (D == 64) launch_attn<64>(q, k, v, pl, ctr, o, lse, (int)window, (int)sink, sl2, (int)planWords, (int)offG, (int)offM, grid, stream, gate, (int)want);
  else launch_attn<128>(q, k, v, pl, ctr, o, lse, (int)window, (int)sink, sl2, (int)planWords, (int)offG, (int)offM, grid, stream, gate, (int)want);
}
"""

DEV = HEADER + _gen(128, False) + _gen(128, True) + _gen(64, True) + KERN

_ext = load_inline(
    name="lsa_fwd_hard_split",
    cpp_sources="void sparse_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, "
    "torch::Tensor sel, torch::Tensor o, torch::Tensor lse, int64_t window, int64_t sink, double scale, int64_t thr, double dfac, double split_thr, torch::Tensor gate_t, int64_t want);",
    cuda_sources=DEV + HOST,
    functions=["sparse_fwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"],
)

THR = 112
DFAC = 1.0
SPLIT_THR = 0.5


_NG = {}


def _nogate(dev):
    if dev not in _NG:
        _NG[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return _NG[dev]


def attention(q, k, v, selected, window=0, sink=0, scale=None):
    b, t, hq, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    selected = selected.contiguous()
    if selected.dtype != torch.int32:
        selected = selected.to(torch.int32)
    o = torch.empty_like(q)
    lse = torch.empty(b, hq, t, dtype=torch.float32, device=q.device)
    _ext.sparse_fwd(q, k, v, selected, o, lse, int(window), int(sink), float(scale), THR, DFAC, SPLIT_THR, _nogate(q.device), 0)
    return o, lse


attention_sorted = attention
