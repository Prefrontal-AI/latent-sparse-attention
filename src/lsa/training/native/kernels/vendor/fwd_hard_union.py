# Vendored from kernels/c13.py (md5 9ff9870bf3a1b6ea7955b684ce9fc554), by a local Claude Fable 5.1 session, 2026-10-14;
# see __init__.py. fwd_hard's union pipeline (c13.py). One edit: load_inline's name "sfwd_c13" renamed.
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
__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map, unsigned src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];\n"
               :: "l"((unsigned long long)map), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
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

#ifndef PLAN_MTS
#define PLAN_MTS 48
#endif
constexpr int MTS = PLAN_MTS;
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
__global__ void __launch_bounds__(256, 6) plan_k(const int* __restrict__ SEL, unsigned* __restrict__ PL, int* __restrict__ ctr,
    int T, int HKV, int G, int TOPK, int window, int sink, int THR, int nqblk, int HB, int planWords, int offG, int offM,
    float dfac, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int NTH = 256;
  extern __shared__ __align__(16) unsigned psm[];
  __shared__ int s_red[NTH / 32];
  pdl_trigger();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  if (it == 0 && tid == 0) *ctr = 0;
  const Item x = get_item(it, HB, HKV, nqblk, G, T * G, window, sink);
  const int NW = x.NW, NP = x.NP, pf = x.pf, sinkEnd = x.sinkEnd, nbq = x.nbq, tl0 = x.tl0, pl = x.pl;
  const int* sb = SEL + ((size_t)(x.b * HKV + x.hk) * T + pf) * TOPK;
  const int nsel = NP * TOPK;
  for (int xx = (2 * NTH + tid) * 32; xx < nsel; xx += NTH * 32) prefetch_l2(sb + xx);
  unsigned* P = PL + (size_t)it * planWords;
  int* dBlk = reinterpret_cast<int*>(P + 4);
  int* gi = reinterpret_cast<int*>(P + offG);
  unsigned* U = psm;
  int* wrank = reinterpret_cast<int*>(psm + NW);
  int* blkTile = wrank + NW;
  uint2* W2 = reinterpret_cast<uint2*>(psm + ((2 * NW + nbq + 3) & ~3));
  unsigned* Ms = psm + ((2 * NW + nbq + 3) & ~3) + 2 * NW;
  for (int w = tid; w < NW; w += NTH)
    U[w] = (sinkEnd > 0 ? range_bits(0, sinkEnd - 1, w) : 0u) | (tl0 <= pl ? range_bits(tl0, pl, w) : 0u);
  __syncthreads();
  if ((TOPK & 7) == 0) fold_simple(sb, nsel, TOPK, pf, tid, NTH, [&](int j, int p) { atomicOr(U + (j >> 5), 1u << (j & 31)); });
  else for_ids(sb, nsel, TOPK, pf, tid, NTH, U, [&](int j, int p, int& idx, unsigned& bit) {
    idx = j >> 5; bit = 1u << (j & 31);
  });
  __syncthreads();
  int utot = 0;
  for (int w = tid; w < NW; w += NTH) utot += __popc(U[w]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) utot += __shfl_xor_sync(0xffffffff, utot, off);
  if (lane == 0) s_red[warp] = utot;
  __syncthreads();
  utot = 0;
#pragma unroll
  for (int k = 0; k < NTH / 32; ++k) utot += s_red[k];
  __syncthreads();
  const bool dense = (float)nbq <= dfac * (float)(x.nband + (utot + 127) / 128);
  const int thrE = dense ? 1 : THR;
  const int bandBlk = (dense && window > 0) ? (x.wlo >> 7) : 0x7fffffff;
  int carry = 0;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const unsigned u = w < NW ? U[w] : 0u;
    const int pc = __popc(u);
    int cnt = pc + __shfl_xor_sync(0xffffffff, pc, 1);
    cnt += __shfl_xor_sync(0xffffffff, cnt, 2);
    const bool dn = w < NW && (cnt >= thrE || (w >> 2) >= bandBlk);
    const int val = dn ? ((w & 3) == 0 ? (1 << 21) : 0) : pc;
    int xs = val;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int y = __shfl_up_sync(0xffffffff, xs, off);
      if (lane >= off) xs += y;
    }
    if (lane == 31) s_red[warp] = xs;
    __syncthreads();
    int pre = carry, tot = 0;
#pragma unroll
    for (int k = 0; k < NTH / 32; ++k) { int v = s_red[k]; if (k < warp) pre += v; tot += v; }
    const int excl = pre + xs - val;
    const int rb = excl & 0x1FFFFF;
    if (w < NW) {
      if (dn) {
        if ((w & 3) == 0) { const int ti = excl >> 21; blkTile[w >> 2] = ti; dBlk[ti] = w >> 2; }
      } else {
        if ((w & 3) == 0) blkTile[w >> 2] = -1;
        wrank[w] = rb;
      }
    }
    unsigned um = (w < NW && !dn) ? u : 0u;
    const int wb = bw + warp * 32;
    const unsigned lmask = (1u << lane) - 1u;
    unsigned act = __ballot_sync(0xffffffff, um != 0u);
    while (act) {
      const int k = __ffs(act) - 1;
      act &= act - 1;
      const unsigned uk = __shfl_sync(0xffffffff, um, k);
      const int rk = __shfl_sync(0xffffffff, rb, k);
      if ((uk >> lane) & 1u) gi[rk + __popc(uk & lmask)] = (wb + k) * 32 + lane;
    }
    carry += tot;
    __syncthreads();
  }
  const int nd = carry >> 21, ng = carry & 0x1FFFFF, ngt = (ng + 127) >> 7;
  if (tid == 0) { P[0] = nd; P[1] = ng; P[2] = dense ? 1u : 0u; }
  const int MT = nd + ngt;
  const int nmw = MT * NP * 4;
  if constexpr (MG) {
    unsigned* M = P + offM;
    for (int xx = tid; xx < nmw; xx += NTH) M[xx] = 0u;
  } else {
    // smem staging holds the first MTS tiles ([p][MTS][4], so one warp's atomics spread over banks); tiles
    // beyond MTS (only for very large unions) go straight to the global plan with global atomics
    const int MTl = min(MT, MTS);
    for (int xx = tid; xx < NP * MTl * 4; xx += NTH) { const int p = xx / (MTl * 4), r = xx - p * MTl * 4; Ms[p * MTS * 4 + r] = 0u; }
    for (int xx = MTS * NP * 4 + tid; xx < nmw; xx += NTH) P[offM + xx] = 0u;
  }
  for (int xx = ng + tid; xx < ngt * 128; xx += NTH) gi[xx] = 0;
  for (int w = tid; w < NW; w += NTH) {
    const int bt = blkTile[w >> 2];
    W2[w] = bt >= 0 ? make_uint2(0xFFFFFFFFu, (unsigned)(bt * 128 + (w & 3) * 32)) : make_uint2(U[w], (unsigned)(nd * 128 + wrank[w]));
  }
  __syncthreads();
  auto mor = [&](int mt, int p, int w4, unsigned bits) {
    if (MG || mt >= MTS) atomicOr(P + offM + (mt * NP + p) * 4 + w4, bits);
    else atomicOr(Ms + (p * MTS + mt) * 4 + w4, bits);
  };
  auto setbit = [&](int j, int p) {
    const uint2 r = W2[j >> 5];
    const int gs = (int)r.y + __popc(r.x & ((1u << (j & 31)) - 1u));
    const int mt = gs >> 7, pos = gs & 127;
    mor(mt, p, (pos >> 1) & 3, 1u << (((pos >> 3) << 1) | (pos & 1)));
  };
  if ((TOPK & 7) == 0) fold_simple(sb, nsel, TOPK, pf, tid, NTH, setbit);
  else {
    for (int e = tid; e < nsel; e += NTH) {
      const int p = e / TOPK, j = __ldg(sb + e);
      if (j >= 0 && j <= pf + p) setbit(j, p);
    }
  }
  if (sinkEnd > 0) {
    for (int xx = tid; xx < NP * sinkEnd; xx += NTH) {
      const int p = xx / sinkEnd, j = xx - p * sinkEnd;
      if (j <= pf + p) setbit(j, p);
    }
  }
  if (!dense && tl0 <= pl) {
    const int tlen = pl - tl0 + 1;
    for (int xx = tid; xx < NP * tlen; xx += NTH) {
      const int p = xx / tlen, j = tl0 + xx - p * tlen, i = pf + p;
      if (j <= i && (i - j < window || j < sink)) setbit(j, p);
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
      if (bits) mor(bt, p, w4, bits);
    }
  }
  if constexpr (!MG) {
    __syncthreads();
    uint4* Mg = reinterpret_cast<uint4*>(P + offM);
    const uint4* M4 = reinterpret_cast<const uint4*>(Ms);
    const int MTl = min(MT, MTS);
    for (int xx = tid; xx < MTl * NP; xx += NTH) { const int mt = xx / NP, p = xx - mt * NP; Mg[xx] = M4[p * MTS + mt]; }
  }
}


#define D_FIX 128

// Exact recomputation of the items whose fast path overflowed (a row sum of inf/NaN, which needs logits more than
// ~160 log2 units above the first tile's row max). One row per thread, two passes over the item's tiles.
__global__ void __launch_bounds__(128) fixup_k(const bf16* __restrict__ Q, const bf16* __restrict__ Kp, const bf16* __restrict__ Vp,
    const unsigned* __restrict__ PL, bf16* __restrict__ O, float* __restrict__ LSE, const int* __restrict__ bad, int epoch, int nItems,
    int B, int T, int HQ, int HKV, int window, int sink, float scale, int nqblk, int planWords, int offG, int offM) {
  const int G = HQ / HKV, TG = T * G, HB = HKV * B;
  for (int base = blockIdx.x * 128; base < nItems; base += gridDim.x * 128) {
   const bool mine = base + (int)threadIdx.x < nItems && __ldcg(bad + base + threadIdx.x) == epoch;
   unsigned todo = __ballot_sync(0xffffffff, mine);
   __shared__ unsigned s_todo[4];
   if ((threadIdx.x & 31) == 0) s_todo[threadIdx.x >> 5] = todo;
   __syncthreads();
   for (int wq = 0; wq < 4; ++wq) {
   unsigned m = s_todo[wq];
   while (m) {
    const int it = base + wq * 32 + (__ffs(m) - 1);
    m &= m - 1;
    const Item x = get_item(it, HB, HKV, nqblk, G, TG, window, sink);
    const unsigned* P = PL + (size_t)it * planWords;
    const int nd = (int)P[0], ng = (int)P[1], dm = (int)P[2];
    const int nband = dm ? 0 : x.nband, ngt = (ng + 127) >> 7, ntile = nband + nd + ngt, g0 = nband + nd;
    const int* dBlk = reinterpret_cast<const int*>(P + 4);
    const int* gi = reinterpret_cast<const int*>(P + offG);
    const unsigned* M = P + offM;
    const int rg = x.R0 + threadIdx.x;
    if (rg < TG) {
    const int i = rg / G, g = rg - i * G, p = i - x.pf;
    const bf16* qr = Q + ((size_t)(x.b * T + i) * HQ + x.hk * G + g) * D_FIX;
    const bf16* Kb = Kp + ((size_t)x.b * T * HKV + x.hk) * D_FIX;
    const bf16* Vb = Vp + ((size_t)x.b * T * HKV + x.hk) * D_FIX;
    float q[D_FIX];
    for (int d = 0; d < D_FIX; ++d) q[d] = __bfloat162float(qr[d]);
    float o[D_FIX];
    for (int d = 0; d < D_FIX; ++d) o[d] = 0.f;
    float m = -INFINITY, l = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
      for (int t = 0; t < ntile; ++t) {
        for (int pos = 0; pos < 128; ++pos) {
          int key; bool ok;
          if (t < nband) {
            key = x.wlo + 128 * t + pos;
            ok = key <= i && (i - key < window || key < sink);
          } else {
            key = t < g0 ? dBlk[t - nband] * 128 + pos : gi[(t - g0) * 128 + pos];
            const unsigned w = M[((t - nband) * x.NP + p) * 4 + ((pos >> 1) & 3)];
            ok = (w >> (((pos >> 3) << 1) | (pos & 1))) & 1u;
          }
          if (!ok) continue;
          const bf16* kr = Kb + (size_t)key * HKV * D_FIX;
          float sc = 0.f;
          for (int d = 0; d < D_FIX; ++d) sc += q[d] * __bfloat162float(kr[d]);
          sc *= scale;
          if (pass == 0) { m = fmaxf(m, sc); continue; }
          const float e = expf(sc - m);
          l += e;
          const bf16* vr = Vb + (size_t)key * HKV * D_FIX;
          for (int d = 0; d < D_FIX; ++d) o[d] += e * __bfloat162float(vr[d]);
        }
      }
    }
    bf16* orow = O + ((size_t)(x.b * T + i) * HQ + x.hk * G + g) * D_FIX;
    const float inv = l > 0.f ? 1.f / l : 0.f;
    for (int d = 0; d < D_FIX; ++d) orow[d] = __float2bfloat16(o[d] * inv);
    LSE[((size_t)x.b * HQ + x.hk * G + g) * T + i] = l > 0.f ? m + logf(l) : -INFINITY;
    }
   }
   }
   __syncthreads();
  }
}


// Plan kernel, second version: window and sink mask bits are set per (query, 32-key word) as slot RANGES
// instead of per key (~160 range operations per item instead of ~4100 per-key atomics), the gather list is
// emitted per lane into a u16 staging area (the mask staging region, free at that point) and copied out
// coalesced, and a dense item only visits the key blocks its window and sinks touch.
template <bool MG>
__global__ void __launch_bounds__(256, 6) plan_k2(const int* __restrict__ SEL, unsigned* __restrict__ PL, int* __restrict__ ctr,
    int T, int HKV, int G, int TOPK, int window, int sink, int THR, int nqblk, int HB, int planWords, int offG, int offM,
    float dfac, const int* __restrict__ gate, int gwant) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int NTH = 256;
  extern __shared__ __align__(16) unsigned psm[];
  __shared__ int s_red[NTH / 32];
  pdl_trigger();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  if (it == 0 && tid == 0) *ctr = 0;
  const Item x = get_item(it, HB, HKV, nqblk, G, T * G, window, sink);
  const int NW = x.NW, NP = x.NP, pf = x.pf, sinkEnd = x.sinkEnd, nbq = x.nbq, tl0 = x.tl0, pl = x.pl;
  const int* sb = SEL + ((size_t)(x.b * HKV + x.hk) * T + pf) * TOPK;
  const int nsel = NP * TOPK;
  for (int xx = (2 * NTH + tid) * 32; xx < nsel; xx += NTH * 32) prefetch_l2(sb + xx);
  unsigned* P = PL + (size_t)it * planWords;
  int* dBlk = reinterpret_cast<int*>(P + 4);
  int* gi = reinterpret_cast<int*>(P + offG);
  unsigned* U = psm;
  int* wrank = reinterpret_cast<int*>(psm + NW);
  int* blkTile = wrank + NW;
  uint2* W2 = reinterpret_cast<uint2*>(psm + ((2 * NW + nbq + 3) & ~3));
  unsigned* Ms = psm + ((2 * NW + nbq + 3) & ~3) + 2 * NW;
  const int NPM = 127 / G + 2;
  unsigned short* G16 = reinterpret_cast<unsigned short*>(Ms);
  const int cap16 = MG ? 0 : MTS * NPM * 4 * 2;
  for (int w = tid; w < NW; w += NTH)
    U[w] = (sinkEnd > 0 ? range_bits(0, sinkEnd - 1, w) : 0u) | (tl0 <= pl ? range_bits(tl0, pl, w) : 0u);
  __syncthreads();
  if ((TOPK & 7) == 0) fold_simple(sb, nsel, TOPK, pf, tid, NTH, [&](int j, int p) { atomicOr(U + (j >> 5), 1u << (j & 31)); });
  else for_ids(sb, nsel, TOPK, pf, tid, NTH, U, [&](int j, int p, int& idx, unsigned& bit) {
    idx = j >> 5; bit = 1u << (j & 31);
  });
  __syncthreads();
  int utot = 0;
  for (int w = tid; w < NW; w += NTH) utot += __popc(U[w]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) utot += __shfl_xor_sync(0xffffffff, utot, off);
  if (lane == 0) s_red[warp] = utot;
  __syncthreads();
  utot = 0;
#pragma unroll
  for (int k = 0; k < NTH / 32; ++k) utot += s_red[k];
  __syncthreads();
  const bool dense = (float)nbq <= dfac * (float)(x.nband + (utot + 127) / 128);
  const bool stage16 = !MG && utot <= cap16 && T <= 65536;
  const int thrE = dense ? 1 : THR;
  const int bandBlk = (dense && window > 0) ? (x.wlo >> 7) : 0x7fffffff;
  int carry = 0;
  for (int bw = 0; bw < NW; bw += NTH) {
    const int w = bw + tid;
    const unsigned u = w < NW ? U[w] : 0u;
    const int pc = __popc(u);
    int cnt = pc + __shfl_xor_sync(0xffffffff, pc, 1);
    cnt += __shfl_xor_sync(0xffffffff, cnt, 2);
    const bool dn = w < NW && (cnt >= thrE || (w >> 2) >= bandBlk);
    const int val = dn ? ((w & 3) == 0 ? (1 << 21) : 0) : pc;
    int xs = val;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int y = __shfl_up_sync(0xffffffff, xs, off);
      if (lane >= off) xs += y;
    }
    if (lane == 31) s_red[warp] = xs;
    __syncthreads();
    int pre = carry, tot = 0;
#pragma unroll
    for (int k = 0; k < NTH / 32; ++k) { int v = s_red[k]; if (k < warp) pre += v; tot += v; }
    const int excl = pre + xs - val;
    const int rb = excl & 0x1FFFFF;
    if (w < NW) {
      if (dn) {
        if ((w & 3) == 0) { const int ti = excl >> 21; blkTile[w >> 2] = ti; dBlk[ti] = w >> 2; }
      } else {
        if ((w & 3) == 0) blkTile[w >> 2] = -1;
        wrank[w] = rb;
      }
    }
    unsigned um = (w < NW && !dn) ? u : 0u;
    // sparse words (at most 8 keys in every word of the warp): each lane emits its own word's keys; clustered
    // words: the warp emits one word per step. Both go to the u16 staging area when it is in use.
    const int pcmax = __reduce_max_sync(0xffffffff, __popc(um));
    if (stage16 && pcmax <= 24) {
      int r = rb;
      while (um) { const int b = __ffs(um) - 1; um &= um - 1; G16[r++] = (unsigned short)((w << 5) + b); }
    } else {
      const int wb = bw + warp * 32;
      const unsigned lmask = (1u << lane) - 1u;
      unsigned act = __ballot_sync(0xffffffff, um != 0u);
      while (act) {
        const int k = __ffs(act) - 1;
        act &= act - 1;
        const unsigned uk = __shfl_sync(0xffffffff, um, k);
        const int rk = __shfl_sync(0xffffffff, rb, k);
        if ((uk >> lane) & 1u) {
          const int slot = rk + __popc(uk & lmask), key = (wb + k) * 32 + lane;
          if (stage16) G16[slot] = (unsigned short)key; else gi[slot] = key;
        }
      }
    }
    carry += tot;
    __syncthreads();
  }
  const int nd = carry >> 21, ng = carry & 0x1FFFFF, ngt = (ng + 127) >> 7;
  if (tid == 0) { P[0] = nd; P[1] = ng; P[2] = dense ? 1u : 0u; }
  if (stage16) {
    for (int xx = tid; xx < ng; xx += NTH) gi[xx] = (int)G16[xx];
    __syncthreads();
  }
  const int MT = nd + ngt;
  const int nmw = MT * NP * 4;
  if constexpr (MG) {
    unsigned* M = P + offM;
    for (int xx = tid; xx < nmw; xx += NTH) M[xx] = 0u;
  } else {
    const int MTl = min(MT, MTS);
    for (int xx = tid; xx < NP * MTl * 4; xx += NTH) { const int p = xx / (MTl * 4), r = xx - p * MTl * 4; Ms[p * MTS * 4 + r] = 0u; }
    for (int xx = MTS * NP * 4 + tid; xx < nmw; xx += NTH) P[offM + xx] = 0u;
  }
  for (int xx = ng + tid; xx < ngt * 128; xx += NTH) gi[xx] = 0;
  for (int w = tid; w < NW; w += NTH) {
    const int bt = blkTile[w >> 2];
    W2[w] = bt >= 0 ? make_uint2(0xFFFFFFFFu, (unsigned)(bt * 128 + (w & 3) * 32)) : make_uint2(U[w], (unsigned)(nd * 128 + wrank[w]));
  }
  __syncthreads();
  auto mor = [&](int mt, int p, int w4, unsigned bits) {
    if (MG || mt >= MTS) atomicOr(P + offM + (mt * NP + p) * 4 + w4, bits);
    else atomicOr(Ms + (p * MTS + mt) * 4 + w4, bits);
  };
  auto setbit = [&](int j, int p) {
    const uint2 r = W2[j >> 5];
    const int gs = (int)r.y + __popc(r.x & ((1u << (j & 31)) - 1u));
    const int mt = gs >> 7, pos = gs & 127;
    mor(mt, p, (pos >> 1) & 3, 1u << (((pos >> 3) << 1) | (pos & 1)));
  };
  // all union-slot positions of the keys [lo, hi] (ids) of query p: contiguous per word, then per tile
  auto setrange = [&](int lo, int hi, int p) {
    for (int w = lo >> 5; w <= (hi >> 5); ++w) {
      const uint2 r = W2[w];
      const unsigned um = r.x & range_bits(lo, hi, w);
      if (!um) continue;
      const int s0 = (int)r.y + __popc(r.x & ((1u << (__ffs(um) - 1)) - 1u));
      const int s1 = s0 + __popc(um) - 1;
      for (int mt = s0 >> 7; mt <= (s1 >> 7); ++mt) {
        const int a = max(s0, mt * 128) - mt * 128, b = min(s1, mt * 128 + 127) - mt * 128;
#pragma unroll
        for (int w4 = 0; w4 < 4; ++w4) { const unsigned bits = perm_range(a, b, w4); if (bits) mor(mt, p, w4, bits); }
      }
    }
  };
  if ((TOPK & 7) == 0) fold_simple(sb, nsel, TOPK, pf, tid, NTH, setbit);
  else {
    for (int e = tid; e < nsel; e += NTH) {
      const int p = e / TOPK, j = __ldg(sb + e);
      if (j >= 0 && j <= pf + p) setbit(j, p);
    }
  }
  if (!dense) {
    // per query: sinks below the band ([0, sinkEnd)), the window tail [max(tl0, i-window+1), i], and (tiny T only)
    // sinks inside the tail [tl0, min(sink-1, i)]; one thread per (query, range)
    for (int xx = tid; xx < NP * 3; xx += NTH) {
      const int p = xx / 3, rg = xx - p * 3, i = pf + p;
      int lo, hi;
      if (rg == 0) { lo = 0; hi = min(sinkEnd, i + 1) - 1; }
      else if (rg == 1) { lo = max(tl0, i - window + 1); hi = (tl0 <= pl && window > 0) ? i : -1; }
      else { lo = tl0; hi = min(sink - 1, i); if (tl0 > pl) hi = -1; }
      if (lo <= hi) setrange(lo, hi, p);
    }
  } else if (window > 0 || sink > 0) {
    // dense item: only the blocks the window and the sinks touch
    const int b0 = window > 0 ? (x.wlo >> 7) : nbq, b1 = pl >> 7;
    const int nbw = b1 - b0 + 1 + (sink > 0 && b0 > 0 ? 1 : 0);
    for (int xx = tid; xx < NP * nbw * 4; xx += NTH) {
      const int w4 = xx & 3, r = xx >> 2, bi = r % nbw, p = r / nbw;
      const int bb = bi < b1 - b0 + 1 ? b0 + bi : 0;
      const int bt = blkTile[bb];
      if (bt < 0) continue;
      const int i = pf + p, kb0 = bb * 128;
      unsigned bits = 0;
      if (window > 0) bits |= perm_range(max(i - window + 1, 0) - kb0, i - kb0, w4);
      if (sink > 0) bits |= perm_range(-kb0, min(sink - 1, i) - kb0, w4);
      if (bits) mor(bt, p, w4, bits);
    }
  }
  if constexpr (!MG) {
    __syncthreads();
    uint4* Mg = reinterpret_cast<uint4*>(P + offM);
    const uint4* M4 = reinterpret_cast<const uint4*>(Ms);
    const int MTl = min(MT, MTS);
    for (int xx = tid; xx < MTl * NP; xx += NTH) { const int mt = xx / NP, p = xx - mt * NP; Mg[xx] = M4[p * MTS + mt]; }
  }
}

template <int D>
struct KCfg {
  static constexpr int BM = 128, BN = 128;
  static constexpr int NSK = 2, NSV = 3;
  static constexpr int QB = BM * D * 2, KB = BN * D * 2;
  static constexpr int SMEM = 2 * QB + (NSK + NSV) * KB + 1024;
};

template <int V_> struct IC { static constexpr int value = V_; };

#ifdef TRACE
__device__ unsigned long long g_trace[2 * 512 * 8];
__device__ __forceinline__ unsigned long long clk() { unsigned long long v; asm volatile("mov.u64 %0, %%clock64;" : "=l"(v) :: "memory"); return v; }
#define TS(e) do { if (trc && gtr < 512) { g_trace[((c * 512) + gtr) * 8 + (e)] = clk(); } } while (0)
#else
#define TS(e) do {} while (0)
#endif

#ifndef PROD_REGS
#define PROD_REGS 96
#endif
#ifndef CONS_REGS
#define CONS_REGS 200
#endif
#ifndef NOFENCE
#define NOFENCE 0
#endif
#if NOFENCE
#define tile_fence() do {} while (0)
#else
#define tile_fence() fence_proxy_async()
#endif
#ifndef PINGPONG
#define PINGPONG 1
#endif

template <int D>
__global__ void __launch_bounds__(384, 1) attn_pers(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmQ,
    const __grid_constant__ CUtensorMap tmO,
    const bf16* __restrict__ Q, const bf16* __restrict__ Kp, const bf16* __restrict__ Vp,
    const unsigned* __restrict__ PL, int* __restrict__ ctr, bf16* __restrict__ O,
    float* __restrict__ LSE, int B, int T, int HQ, int HKV, int window, int sink,
    float sl2, int nqblk, int planWords, int offG, int offM, int useQT, int useOT, const int* __restrict__ gate, int gwant,
    int* __restrict__ bad, int epoch) {
  if (gate != nullptr && __ldg(gate) != gwant) return;
  constexpr int BM = 128, BN = 128, CPR = D / 8, NO = D / 2;
  constexpr int NSK = KCfg<D>::NSK, NSV = KCfg<D>::NSV, QB = KCfg<D>::QB, KB = KCfg<D>::KB;
  static_assert(NSK == 2 && NSV == 3, "K ring of two (tile loop unrolled by two), V ring of three (runtime stage)");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * NSK + 2 * NSV + 6];
  __shared__ int slot[2][4];
  __shared__ float anchor_s[16];
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
    for (int s = 0; s < NSK; ++s) { mbar_init(fullK(s), 128); mbar_init(emptyK(s), 256); }
    for (int s = 0; s < NSV; ++s) { mbar_init(fullV(s), 128); mbar_init(emptyV(s), 256); }
    for (int b = 0; b < 2; ++b) { mbar_init(qfull(b), 128); mbar_init(qempty(b), 256); mbar_init(qinfo(b), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  // Ring discipline shared by producer and consumers: every item starts at stage 0 (tile t of an item uses K stage
  // t & 1 and V stage t & 1), and the mbarrier phase of each stage is counted per stage (kuse0/kuse1, vuse0/vuse1),
  // so the consumer's tile loop can be unrolled by two with compile-time stages.
  if (warp >= 8) {
    reg_dealloc<PROD_REGS>();
    const int ptid = tid - 256;
    int it = blockIdx.x;
    int nextIt = -1, tick = -1, hnd = 0, hng = 0, hdm = 0;
    int kuse0 = 0, kuse1 = 0, vuse0 = 0, vuse1 = 0, vuse2 = 0;
    const int cc = ptid % CPR, rr = ptid / CPR;
    const unsigned sob = (cc >> 3) * (BN * 128) + rr * CPR * 128;
    const unsigned cx = (unsigned)(cc & 7);
    const unsigned long long pol = make_pol_last();
    for (int n = 0;; ++n) {
      const int qb = n & 1;
      if (n >= 2) mbar_wait(qempty(qb), ((n >> 1) - 1) & 1);
      if (n > 0) {
        if (ptid == 0) {
          volatile int* sw = slot[qb];
          sw[0] = nextIt; sw[1] = hnd; sw[2] = hng; sw[3] = hdm;
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
          sw[0] = it; sw[1] = __ldg(P0); sw[2] = __ldg(P0 + 1); sw[3] = __ldg(P0 + 2);
          mbar_arrive(qinfo(0));
        }
        bar_sync(3, 128);
      }
      const int nd = slot[qb][1], ng = slot[qb][2], dm = slot[qb][3];
      if (ptid == 0) tick = (int)gridDim.x + atomicAdd(ctr, 1);
      const unsigned* P = PL + (size_t)it * planWords;
      const int ngt = (ng + 127) >> 7;
      const int nbandE = dm ? 0 : x.nband;
      const int* dBlk = reinterpret_cast<const int*>(P + 4);
      const int* gi = reinterpret_cast<const int*>(P + offG) + rr * CPR;
      const int ntile = nbandE + nd + ngt;
      const int g0 = nbandE + nd;
      const int tpre = min(2, ntile - 1);
      int4 jv[CPR / 4];
      if (ngt > 0) {
#pragma unroll
        for (int i2 = 0; i2 < CPR / 4; ++i2) jv[i2] = __ldg(reinterpret_cast<const int4*>(gi) + i2);
      }
      const char* Kb = reinterpret_cast<const char*>(Kp + ((size_t)x.b * T * HKV + x.hk) * D + cc * 8);
      const char* Vb = reinterpret_cast<const char*>(Vp + ((size_t)x.b * T * HKV + x.hk) * D + cc * 8);
      int sv = 0, tv = 0;   // V stage of tile t = t % 3, tv = t / 3
      for (int t = 0; t < ntile; ++t) {
        const int sk = t & 1;
        const int uk = (sk ? kuse1 : kuse0) + (t >> 1), uv = (sv == 0 ? vuse0 : sv == 1 ? vuse1 : vuse2) + tv;
        if (t == tpre && ptid == 0) {
          nextIt = tick < nItems ? tick : -1;
          if (nextIt >= 0) {
            const unsigned* P2 = PL + (size_t)nextIt * planWords;
            hnd = __ldg(P2); hng = __ldg(P2 + 1); hdm = __ldg(P2 + 2);
            if (useQT) {
              const Item xn = get_item(nextIt, HB, HKV, nqblk, G, TG, window, sink);
#pragma unroll
              for (int p = 0; p < D / 64; ++p) tma_prefetch_4d(&tmQ, p * 64, xn.hk * G, xn.pf, xn.b);
            }
          }
        }
        if (t < g0) {
          const int kd = t < nbandE ? x.wlo + 128 * t : __ldg(dBlk + t - nbandE) * 128;
          if (uk > 0) mbar_wait(emptyK(sk), (uk - 1) & 1);
          if (ptid == 0) {
            mbar_expect_tx(fullK(sk), KB);
#pragma unroll
            for (int p = 0; p < D / 64; ++p) tma_load_4d(sK(sk) + p * BN * 128, &tmK, p * 64, x.hk, kd, x.b, fullK(sk));
          } else mbar_arrive(fullK(sk));
          if (uv > 0) mbar_wait(emptyV(sv), (uv - 1) & 1);
          if (ptid == 0) {
            mbar_expect_tx(fullV(sv), KB);
#pragma unroll
            for (int p = 0; p < D / 64; ++p) tma_load_4d(sV(sv) + p * BN * 128, &tmV, p * 64, x.hk, kd, x.b, fullV(sv));
          } else mbar_arrive(fullV(sv));
        } else {
          if (uk > 0) mbar_wait(emptyK(sk), (uk - 1) & 1);
          const unsigned kbase = sK(sk) + sob;
#pragma unroll
          for (int i2 = 0; i2 < CPR; ++i2) { const int4 v4 = jv[i2 >> 2]; const int id = (i2 & 3) == 0 ? v4.x : (i2 & 3) == 1 ? v4.y : (i2 & 3) == 2 ? v4.z : v4.w;
            cp_async16h(kbase + i2 * 128 + ((cx ^ (unsigned)(i2 & 7)) << 4), Kb + (unsigned)id * (unsigned)(HKV * D * 2), pol); }
          cp_mbar_arrive(fullK(sk));
          if (uv > 0) mbar_wait(emptyV(sv), (uv - 1) & 1);
          const unsigned vbase = sV(sv) + sob;
#pragma unroll
          for (int i2 = 0; i2 < CPR; ++i2) { const int4 v4 = jv[i2 >> 2]; const int id = (i2 & 3) == 0 ? v4.x : (i2 & 3) == 1 ? v4.y : (i2 & 3) == 2 ? v4.z : v4.w;
            cp_async16h(vbase + i2 * 128 + ((cx ^ (unsigned)(i2 & 7)) << 4), Vb + (unsigned)id * (unsigned)(HKV * D * 2), pol); }
          cp_mbar_arrive(fullV(sv));
          if (t + 1 < ntile) {
            const int* gn = gi + (t + 1 - g0) * 128;
#pragma unroll
            for (int i2 = 0; i2 < CPR / 4; ++i2) jv[i2] = __ldg(reinterpret_cast<const int4*>(gn) + i2);
          }
        }
        if (sv == 2) { sv = 0; ++tv; } else ++sv;
      }
      kuse0 += (ntile + 1) >> 1; kuse1 += ntile >> 1; vuse0 += (ntile + 2) / 3; vuse1 += (ntile + 1) / 3; vuse2 += ntile / 3;
    }
    cp_wait_all();
    return;
  }
  reg_alloc<CONS_REGS>();
  pdl_wait();

  const int c = warp >> 2;
  const int wq = warp & 3;
  const int wtid = tid & 127;
  const int qd = lane & 3;
  const int r0 = c * 64 + wq * 16 + (lane >> 2);
  const int sh = 2 * qd;
  const int my_bar = 1 + c, other_bar = 2 - c;
  const unsigned anchor = (unsigned)__cvta_generic_to_shared(&anchor_s[warp]);
#ifdef TRACE
  const bool trc = blockIdx.x == 0 && (warp & 3) == 0 && lane == 0;
  int gtr = 0;
#endif
  const unsigned long long DHI = (unsigned long long)(64u | (1u << 30)) << 32;
  const unsigned long long qdesc0 = DHI | ((((sQb(0) + c * 64 * 128) >> 4) & 0x3FFF) | (1u << 16));
  const unsigned long long qdesc1 = DHI | ((((sQb(1) + c * 64 * 128) >> 4) & 0x3FFF) | (1u << 16));
  const unsigned long long kdesc0 = DHI | (((sK(0) >> 4) & 0x3FFF) | (1u << 16));
  const unsigned long long kdesc1 = DHI | (((sK(1) >> 4) & 0x3FFF) | (1u << 16));
  const unsigned long long vdesc0 = DHI | (((sV(0) >> 4) & 0x3FFF) | ((unsigned)((BN * 128) >> 4) << 16));
  const unsigned long long vdesc1 = DHI | (((sV(1) >> 4) & 0x3FFF) | ((unsigned)((BN * 128) >> 4) << 16));
  const unsigned long long vdesc2 = DHI | (((sV(2) >> 4) & 0x3FFF) | ((unsigned)((BN * 128) >> 4) << 16));
  auto vdesc_of = [&](int sv) { return sv == 0 ? vdesc0 : sv == 1 ? vdesc1 : vdesc2; };

  float o[NO];
  float s[64];
  unsigned p[8][4];
  float mu0, mu1, l0, l1;
  int kuse0 = 0, kuse1 = 0, vuse0 = 0, vuse1 = 0, vuse2 = 0;

  auto issue_S = [&](unsigned long long qdesc, unsigned long long kdesc) {
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      const unsigned off = (kk >> 2) * 1024 + (kk & 3) * 2;
      wgmma_ss_n128(s, qdesc + off, kdesc + off, kk > 0 ? 1 : 0);
    }
  };
  auto issue_PV = [&](const unsigned (*pp)[4], unsigned long long vdesc) {
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) wgmma_rs_n128(o, pp[kk], vdesc + kk * 128, 1);
  };
  auto fence_s = [&]() {
#pragma unroll
    for (int k2 = 0; k2 < 64; ++k2) fence_reg(s[k2]);
  };
  auto fence_op = [&](unsigned (*pp)[4]) {
#pragma unroll
    for (int k2 = 0; k2 < NO; ++k2) fence_reg(o[k2]);
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2)
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) fence_ureg(pp[k2][j2]);
  };
  // first tile of an item: exact row max -> reference mu (log2 units) with a 2^-32 headroom bias
  auto softmax_first = [&](const unsigned x0, const unsigned x1) {
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
    mu0 = (mx0 == NEG ? 0.f : mx0 * sl2) + 32.f; mu1 = (mx1 == NEG ? 0.f : mx1 * sl2) + 32.f;
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
    l0 = (sa[0] + sa[1]) + (sa[2] + sa[3]);
    l1 = (sb[0] + sb[1]) + (sb[2] + sb[3]);
  };
  // later tiles: no row max; f0/f1 are the factors to apply to o once this tile's PV has landed
  auto softmax_fast = [&](const unsigned x0, const unsigned x1, float& f0, float& f1) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      s[4 * n] = (x0 & (1u << (2 * n))) ? s[4 * n] : NEG;
      s[4 * n + 1] = (x0 & (2u << (2 * n))) ? s[4 * n + 1] : NEG;
      s[4 * n + 2] = (x1 & (1u << (2 * n))) ? s[4 * n + 2] : NEG;
      s[4 * n + 3] = (x1 & (2u << (2 * n))) ? s[4 * n + 3] : NEG;
    }
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
    const float tl0 = (sa[0] + sa[1]) + (sa[2] + sa[3]), tl1 = (sb[0] + sb[1]) + (sb[2] + sb[3]);
    l0 += tl0; l1 += tl1;
    f0 = 1.f; f1 = 1.f;
    if (__any_sync(0xffffffff, tl0 > 1048576.f || tl1 > 1048576.f)) {
      // rare: a row's reference moves by 32; the decision is per row, i.e. uniform over the row's 4 lanes
#ifdef NOQUADFIX
      const bool b0 = tl0 > 1048576.f, b1 = tl1 > 1048576.f;
#else
      const unsigned qm = 0xFu << (lane & 28);
      const bool b0 = __any_sync(qm, tl0 > 1048576.f), b1 = __any_sync(qm, tl1 > 1048576.f);
#endif
      if (b0) { f0 = 0x1p-32f; mu0 += 32.f; l0 *= f0; }
      if (b1) { f1 = 0x1p-32f; mu1 += 32.f; l1 *= f1; }
    }
  };
  auto pack_p = [&](unsigned (*pp)[4]) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      pp[n >> 1][(n & 1) * 2 + 0] = pack_bf16(s[4 * n], s[4 * n + 1]);
      pp[n >> 1][(n & 1) * 2 + 1] = pack_bf16(s[4 * n + 2], s[4 * n + 3]);
    }
  };
  auto rescale = [&](float c0, float c1) {
    if (__any_sync(0xffffffff, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
      for (int k2 = 0; k2 < NO / 4; ++k2) {
        o[4 * k2] *= c0; o[4 * k2 + 1] *= c0; o[4 * k2 + 2] *= c1; o[4 * k2 + 3] *= c1;
      }
    }
  };

  for (int n = 0;; ++n) {
    const int qb = n & 1;
    mbar_wait(qinfo(qb), (n >> 1) & 1);
    volatile int* sl = slot[qb];
    const int it = sl[0], nd = sl[1], ng = sl[2], dm = sl[3];
    if (it < 0) break;
    const Item x = get_item(it, HB, HKV, nqblk, G, TG, window, sink);
    const unsigned* P = PL + (size_t)it * planWords;
    const int ngt = (ng + 127) >> 7;
    const int nband = dm ? 0 : x.nband, wlo = x.wlo, pf = x.pf, NP = x.NP;
    const int ntile = nband + nd + ngt;
    const unsigned* MW = P + offM + qd;
    const int rg0 = x.R0 + r0, rg1 = rg0 + 8;
    const int i0 = min(rg0, TG - 1) / G, i1 = min(rg1, TG - 1) / G;
    const unsigned sQ = sQb(qb);
    const unsigned long long qdesc = qb ? qdesc1 : qdesc0;

    auto arith = [&](int i, int kb0) -> unsigned {
      const int lo = max(i - window + 1, kb0) - kb0, hi = min(i, kb0 + 127) - kb0;
      const int shi = min(sink - 1, i) - kb0;
      return perm_range(lo, hi, qd) | perm_range(0, shi, qd);
    };
    auto tile_mask = [&](int t, unsigned& a, unsigned& bm) {
      if (t >= ntile) return;
      if (t < nband) {
        const int kb0 = wlo + t * 128;
        a = arith(i0, kb0); bm = arith(i1, kb0);
      } else {
        const unsigned* mm = MW + (size_t)(t - nband) * NP * 4;
        a = __ldg(mm + (i0 - pf) * 4); bm = __ldg(mm + (i1 - pf) * 4);
      }
    };

#pragma unroll
    for (int k2 = 0; k2 < NO; ++k2) o[k2] = 0.f;
    mu0 = 32.f; mu1 = 32.f; l0 = 0.f; l1 = 0.f;
    float pf0 = 1.f, pf1 = 1.f, f0, f1;   // pending o factors (this tile's shift applies after its PV)
    unsigned mw0 = 0, mw1 = 0, nw0 = 0, nw1 = 0;
    tile_mask(0, mw0, mw1);
    tile_mask(1, nw0, nw1);
    mbar_wait(qfull(qb), (n >> 1) & 1);
    fence_proxy_async();
#if PINGPONG
    if (c == 1) bar_arrive(1, 256);
#endif
    {
      mbar_wait(fullK(0), kuse0 & 1);
      tile_fence();
#if PINGPONG
      bar_sync(my_bar, 256);
#endif
      wg_fence();
      issue_S(qdesc, kdesc0);
      wg_commit();
#if PINGPONG
      bar_arrive(other_bar, 256);
#endif
      wg_wait<0>();
      fence_s();
      mbar_arrive(emptyK(0));
      softmax_first(mw0, mw1);
      pack_p(p);
      mw0 = nw0; mw1 = nw1;
    }
    // Tile t (t >= 1): S on K stage ST = t & 1, PV of tile t-1 on V stage SP = ST ^ 1.
    int svp = 0, tvp = 0;   // V stage of tile t-1 and its index (t-1)/3 within the item
    auto body = [&](auto stc, int t) {
      constexpr int ST = decltype(stc)::value;
      const int kp = ((ST ? kuse1 : kuse0) + (t >> 1)) & 1;
      const int SP = svp;
      const int vp = ((SP == 0 ? vuse0 : SP == 1 ? vuse1 : vuse2) + tvp) & 1;
      TS(0);
      mbar_wait(fullK(ST), kp);
      mbar_wait(fullV(SP), vp);
      TS(1);
      tile_fence();
      tile_mask(t + 1, nw0, nw1);
#if PINGPONG
      bar_sync(my_bar, 256);
#endif
      TS(2);
      wg_fence();
      issue_S(qdesc, ST ? kdesc1 : kdesc0);
      wg_commit();
      issue_PV(p, vdesc_of(SP));
      wg_commit();
#if PINGPONG
      bar_arrive(other_bar, 256);
#endif
      TS(3);
      wg_wait<1>();
      fence_s();
      TS(4);
      mbar_arrive(emptyK(ST));
      softmax_fast(mw0, mw1, f0, f1);
      // anchor: a volatile shared store that depends on every exp (through the row sums) keeps ptxas from
      // hoisting the PV wait above the softmax
      asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(anchor), "f"(l0 + l1) : "memory");
      TS(5);
      wg_wait<0>();
      fence_op(p);
      TS(6);
      mbar_arrive(emptyV(SP));
      rescale(pf0, pf1);
      pf0 = f0; pf1 = f1;
      pack_p(p);
      mw0 = nw0; mw1 = nw1;
      if (svp == 2) { svp = 0; ++tvp; } else ++svp;
      TS(7);
#ifdef TRACE
      ++gtr;
#endif
    };
    int t = 1;
    for (; t + 1 < ntile; t += 2) { body(IC<1>{}, t); body(IC<0>{}, t + 1); }
    if (t < ntile) body(IC<1>{}, t);
    {
      const int slv = svp;
      const int vp = ((slv == 0 ? vuse0 : slv == 1 ? vuse1 : vuse2) + tvp) & 1;
      mbar_wait(fullV(slv), vp);
      tile_fence();
#if PINGPONG
      bar_sync(my_bar, 256);
#endif
      wg_fence();
      issue_PV(p, vdesc_of(slv));
      wg_commit();
#if PINGPONG
      if (c == 0) bar_arrive(other_bar, 256);
#endif
      wg_wait<0>();
#pragma unroll
      for (int k2 = 0; k2 < NO; ++k2) fence_reg(o[k2]);
      mbar_arrive(emptyV(slv));
      rescale(pf0, pf1);
    }
    kuse0 += (ntile + 1) >> 1; kuse1 += ntile >> 1; vuse0 += (ntile + 2) / 3; vuse1 += (ntile + 1) / 3; vuse2 += ntile / 3;
    float L0 = l0, L1 = l1;
    L0 += __shfl_xor_sync(0xffffffff, L0, 1);
    L0 += __shfl_xor_sync(0xffffffff, L0, 2);
    L1 += __shfl_xor_sync(0xffffffff, L1, 1);
    L1 += __shfl_xor_sync(0xffffffff, L1, 2);
    const float inv0 = L0 > 0.f ? 1.f / L0 : 0.f, inv1 = L1 > 0.f ? 1.f / L1 : 0.f;
    // a row sum that is not finite means the fast path overflowed: flag the item for the exact fixup kernel
#ifndef NOBADFLAG
    if (__any_sync(0xffffffff, !(L0 < INFINITY) || !(L1 < INFINITY))) { if (lane == 0) bad[it] = epoch; }
#endif
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
        LSE[((size_t)x.b * HQ + x.hk * G + g) * T + i] = L0 > 0.f ? (mu0 + log2f(L0)) * 0.69314718055994531f : NEG;
      }
      if (rg1 < TG) {
        const int i = rg1 / G, g = rg1 - i * G;
        LSE[((size_t)x.b * HQ + x.hk * G + g) * T + i] = L1 > 0.f ? (mu1 + log2f(L1)) * 0.69314718055994531f : NEG;
      }
    }
    if (useOT) {
      fence_proxy_async();
      bar_sync(5 + c, 128);
      if (wtid == 0) {
#pragma unroll
        for (int p2 = 0; p2 < D / 64; ++p2) tma_store_4d(&tmO, sQ + p2 * (BM * 128) + c * 64 * 128, p2 * 64, x.hk * G, x.pf + c * (64 / G), x.b);
        bulk_commit();
        bulk_wait_read0();
      }
    } else {
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
static int* get_bad(long n) {
  static int* r = nullptr;
  static long cap = 0;
  if (n > cap) {
    if (r) cudaFree(r);
    cap = std::max<long>(n, 1 << 16);
    cudaMalloc(&r, cap * 4); cudaMemset(r, 0, cap * 4);
  }
  return r;
}
static int next_epoch() {
  static int e = 0;
  return ++e;
}
static int* get_ctr() {
  static int* c = nullptr;
  if (!c) { cudaMalloc(&c, 64); cudaMemset(c, 0, 64); }
  return c;
}
template <int D>
void launch_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, const unsigned* pl, int* ctr, torch::Tensor o, torch::Tensor lse,
                 int window, int sink, float sl2, int planWords, int offG, int offM, int grid, cudaStream_t stream, const int* gate, int gwant,
                 int* bad, int epoch) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), HKV = k.size(2);
  const int G = HQ / HKV;
  constexpr int SM = KCfg<D>::SMEM;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_pers<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    init = true;
  }
  MapCache ck, cv, cq;   // encoded on every call (no pointer-keyed caching)
  const CUtensorMap& tk = get_map(ck, k.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tv = get_map(cv, v.data_ptr(), B, T, HKV, D, 1, 128);
  const int useQT = (128 % G == 0) ? 1 : 0;
  const CUtensorMap& tq = useQT ? get_map(cq, q.data_ptr(), B, T, HQ, D, G, 128 / G) : tk;
  MapCache co;
  const int useOT = (64 % G == 0) ? 1 : 0;
  const CUtensorMap& to = useOT ? get_map(co, o.data_ptr(), B, T, HQ, D, G, 64 / G) : tk;
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
  cudaError_t e = cudaLaunchKernelEx(&cfg, attn_pers<D>, tk, tv, tq, to,
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()), pl, ctr,
      reinterpret_cast<bf16*>(o.data_ptr()), lse.data_ptr<float>(), B, T, HQ, HKV, window, sink, sl2,
      nqblk, planWords, offG, offM, useQT, useOT, gate, gwant, bad, epoch);
  TORCH_CHECK(e == cudaSuccess, "launch failed ", cudaGetErrorString(e));
}
}  // namespace
void sparse_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel,
                torch::Tensor o, torch::Tensor lse, int64_t window, int64_t sink, double scale, int64_t thr, double dfac, torch::Tensor gate_t, int64_t want) {
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
  const long offG = (4 + nb + 3) & ~3L;
  const long offM = offG + ngtmax * 128;
  const long maskWords = tilesmax * NPM * 4;
  const long planWords = (offM + maskWords + 3) & ~3L;
  auto plan = torch::empty({nItems * planWords}, q.options().dtype(torch::kInt32));
  unsigned* pl = reinterpret_cast<unsigned*>(plan.data_ptr());
  int* ctr = get_ctr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long smw = ((2 * nb * 4 + nb + 3) & ~3L) + 2 * nb * 4;
  const long psmB = (smw + (long)MTS * NPM * 4) * 4;
  if (psmB <= 200 * 1024) {
    static long pset = 0;
    if (psmB > pset) {
      cudaFuncSetAttribute(plan_k2<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)psmB);
      pset = psmB;
    }
    plan_k2<false><<<(int)nItems, 256, psmB, stream>>>(sel.data_ptr<int>(), pl, ctr, T, HKV, G, TOPK, (int)window, (int)sink,
        THR, nqblk, HB, (int)planWords, (int)offG, (int)offM, (float)dfac, gate, (int)want);
  } else {
    const long b2 = smw * 4;
    static long pset2 = 0;
    if (b2 > pset2) {
      cudaFuncSetAttribute(plan_k2<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b2);
      pset2 = b2;
    }
    plan_k2<true><<<(int)nItems, 256, b2, stream>>>(sel.data_ptr<int>(), pl, ctr, T, HKV, G, TOPK, (int)window, (int)sink,
        THR, nqblk, HB, (int)planWords, (int)offG, (int)offM, (float)dfac, gate, (int)want);
  }
  const float sl2 = (float)(scale * 1.4426950408889634);
  int* bad = get_bad(nItems);
  const int epoch = next_epoch();
  TORCH_CHECK(D == 128, "d=128 only");
  launch_attn<128>(q, k, v, pl, ctr, o, lse, (int)window, (int)sink, sl2, (int)planWords, (int)offG, (int)offM, grid, stream, gate, (int)want, bad, epoch);
  // exact recomputation of flagged items (none for sane logits); a gated-off pipeline flags nothing
#ifndef NOFIXUP
  fixup_k<<<32, 128, 0, stream>>>(reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()), pl, reinterpret_cast<bf16*>(o.data_ptr()), lse.data_ptr<float>(), bad, epoch, (int)nItems,
      B, T, HQ, HKV, (int)window, (int)sink, (float)scale, nqblk, (int)planWords, (int)offG, (int)offM);
#endif
}
"""

DEV = HEADER + _gen(128, False) + _gen(128, True) + _gen(64, True) + KERN

_ext = load_inline(
    name="lsa_fwd_hard_union",
    cpp_sources="void sparse_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, "
    "torch::Tensor sel, torch::Tensor o, torch::Tensor lse, int64_t window, int64_t sink, double scale, int64_t thr, double dfac, torch::Tensor gate_t, int64_t want);",
    cuda_sources=DEV + HOST,
    functions=["sparse_fwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"],
)

THR = 112
DFAC = 1.0


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
    _ext.sparse_fwd(q, k, v, selected, o, lse, int(window), int(sink), float(scale), THR, DFAC, _nogate(q.device), 0)
    return o, lse


attention_sorted = attention
