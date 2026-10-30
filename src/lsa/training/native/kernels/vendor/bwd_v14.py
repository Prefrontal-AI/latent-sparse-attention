# Vendored from v14.py (md5 1dc055564bfc6f3550d0e081dcbb334b), by a local Claude Opus 5.5 session, 2026-10-13;
# see __init__.py. One edit: load_inline's name "sabwd_v14" renamed.
import math
import torch
from torch.utils.cpp_extension import load_inline


def _gen(name, N, rs, ta=0, tb=0) -> str:
    nreg = N // 2
    d = ",".join(f"%{i}" for i in range(nreg))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    if rs:
        a = "{" + ",".join(f"%{nreg + i}" for i in range(4)) + "}"
        b = f"%{nreg + 4}"
        sc = f"%{nreg + 5}"
        tail = f"p, 1, 1, {tb}"
        ins = ", ".join(f'"r"(a[{i}])' for i in range(4)) + ', "l"(db), "r"(sd)'
        sig = "float* d, const unsigned* a, unsigned long long db, int sd"
    else:
        a = f"%{nreg}"
        b = f"%{nreg + 1}"
        sc = f"%{nreg + 2}"
        tail = f"p, 1, 1, {ta}, {tb}"
        ins = '"l"(da), "l"(db), "r"(sd)'
        sig = "float* d, unsigned long long da, unsigned long long db, int sd"
    return (f"__device__ __forceinline__ void {name}({sig}) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sc}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {{{d}}}, {a}, {b}, {tail};\\n}}\\n"\n'
            f"    : {outs} : {ins} : \"memory\");\n}}\n")


HEADER = r"""
#include <cuda.h>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
__device__ __forceinline__ void fence_ureg(unsigned& r) { asm volatile("" : "+r"(r) :: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
template <int N> __device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> __device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
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
__device__ __forceinline__ void tma_reduce_4d(const CUtensorMap* map, int c0, int c1, int c2, int c3, unsigned src) {
  asm volatile("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];\n"
               :: "l"((unsigned long long)map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(src) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_load(unsigned dst, const void* src, unsigned bytes, unsigned mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void st_shared_u32(unsigned addr, unsigned v) {
  asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void st_shared_f2(unsigned addr, float a, float b) {
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" :: "r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ unsigned long long make_desc(unsigned addr, unsigned lbo, unsigned sbo) {
  unsigned long long d = 0;
  d |= (unsigned long long)((addr >> 4) & 0x3FFF);
  d |= (unsigned long long)((lbo >> 4) & 0x3FFF) << 16;
  d |= (unsigned long long)((sbo >> 4) & 0x3FFF) << 32;
  d |= 1ull << 62;
  return d;
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}
"""

KERN = r"""
// generic fallback (unsupported group sizes / head dims): one block per (query, head)
__global__ void ref_bwd_kernel(const bf16* q, const bf16* k, const bf16* v, const int* sel, const bf16* o, const float* lse,
                               const bf16* dout, float* dq, float* dk, float* dv, int T, int HQ, int HKV, int D, int TOPK,
                               int window, int sink, float scale) {
  const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
  const int hk = h / (HQ / HKV);
  const long row = (((long)b * T + i) * HQ + h) * D + d;
  const float qd = __bfloat162float(q[row]);
  const float dod = __bfloat162float(dout[row]);
  const float lse_i = lse[((long)b * HQ + h) * T + i];
  const int* ids = sel + (((long)b * HKV + hk) * T + i) * TOPK;
  __shared__ float partial[32];
  auto block_sum = [&](float x) {
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off);
    __syncthreads();
    if ((d & 31) == 0) partial[d >> 5] = x;
    __syncthreads();
    x = 0.f;
    for (int w = 0; w < (D >> 5); ++w) x += partial[w];
    return x;
  };
  const float delta = block_sum(__bfloat162float(o[row]) * dod);
  float dq_acc = 0.f;
  auto visit = [&](int j) {
    const long at = (((long)b * T + j) * HKV + hk) * D + d;
    const float kd = __bfloat162float(k[at]);
    const float s = block_sum(qd * kd) * scale;
    const float p = __expf(s - lse_i);
    const float dp = block_sum(dod * __bfloat162float(v[at]));
    const float ds = p * (dp - delta);
    dq_acc += ds * kd * scale;
    atomicAdd(&dk[at], ds * qd * scale);
    atomicAdd(&dv[at], p * dod);
  };
  for (int j = 0; j < min(sink, i + 1); ++j) visit(j);
  for (int j = max(i - window + 1, sink); j <= i; ++j) visit(j);
  for (int s = 0; s < TOPK; ++s) {
    const int j = ids[s];
    if (j >= 0) visit(j);
  }
  dq[row] = dq_acc;
}

__global__ void delta_k(const bf16* __restrict__ O, const bf16* __restrict__ DO, const float* __restrict__ LSE,
                        float2* __restrict__ LD, float* __restrict__ dq32, float* __restrict__ dk32, float* __restrict__ dv32,
                        long nkv8, int B, int T, int HQ, int HKV, int G, int D, int nqt, float l2e, float lsh) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int tpr = D / 8;
  const long row = idx / tpr;
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
    float4* qz = reinterpret_cast<float4*>(dq32 + idx * 8);
    qz[0] = z; qz[1] = z;
  }
  if (idx < nkv8) {
    float4* kz = reinterpret_cast<float4*>(dk32 + idx * 8);
    float4* vz = reinterpret_cast<float4*>(dv32 + idx * 8);
    kz[0] = z; kz[1] = z; vz[0] = z; vz[1] = z;
  }
  for (int off = tpr / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  if (ok && (idx % tpr) == 0) {
    const int h = row % HQ;
    const int pos = (row / HQ) % T;
    const int b = row / ((long)HQ * T);
    const int hk = h / G, g = h % G;
    LD[(size_t)(b * HKV + hk) * nqt * 64 + (size_t)pos * G + g] = make_float2(LSE[((size_t)b * HQ + h) * T + pos] * l2e - lsh, s);
  }
}

__global__ void cvt_k(const float* __restrict__ a, bf16* __restrict__ ao, long na8, const float* __restrict__ b,
                      bf16* __restrict__ bo, const float* __restrict__ c, bf16* __restrict__ co, long nb8) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  auto cv = [&](const float* src, bf16* dst) {
    const float4 x = reinterpret_cast<const float4*>(src + idx * 8)[0];
    const float4 y = reinterpret_cast<const float4*>(src + idx * 8)[1];
    uint4 r;
    r.x = pack_bf16(x.x, x.y); r.y = pack_bf16(x.z, x.w); r.z = pack_bf16(y.x, y.y); r.w = pack_bf16(y.z, y.w);
    *reinterpret_cast<uint4*>(dst + idx * 8) = r;
  };
  if (idx < na8) cv(a, ao);
  if (idx < nb8) { cv(b, bo); cv(c, co); }
}

__global__ void __launch_bounds__(256) build_k(const int* __restrict__ sel, unsigned* __restrict__ ST, unsigned* __restrict__ OCC,
                                               int T, int TOPK, int nw, int nkb) {
  extern __shared__ unsigned sm[];
  const int w = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  const int P0 = w * 32, np = min(32, T - P0);
  const int Kmax = min(T, ((P0 + 32 + 127) / 128) * 128);
  for (int i = tid; i < Kmax; i += 256) sm[i] = 0u;
  __syncthreads();
  const int* sb = sel + ((size_t)bh * T + P0) * TOPK;
  const int n = np * TOPK;
  for (int x = tid; x < n; x += 256) {
    const int j = __ldg(sb + x);
    if (j >= 0 && j < T) atomicOr(&sm[j], 1u << (x / TOPK));
  }
  __syncthreads();
  unsigned* out = ST + ((size_t)bh * nw + w) * T;
  for (int i = tid; i < Kmax; i += 256) out[i] = sm[i];
  const int warp = tid >> 5, lane = tid & 31;
  for (int kb = warp; kb * 128 < Kmax; kb += 8) {
    unsigned v = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = kb * 128 + e * 32 + lane;
      if (j < Kmax) v |= sm[j];
    }
    v = __reduce_or_sync(0xffffffff, v);
    if (lane == 0) OCC[((size_t)bh * nkb + kb) * nw + w] = v;
  }
}

__device__ __forceinline__ int block_scan256(int v, int& total, int* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) { int y = __shfl_up_sync(0xffffffff, x, off); if (lane >= off) x += y; }
  if (lane == 31) red[warp] = x;
  __syncthreads();
  int pre = 0, tot = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) { int r = red[k]; if (k < warp) pre += r; tot += r; }
  __syncthreads();
  total = tot;
  return pre + x - v;
}

__global__ void __launch_bounds__(256) list_k(const unsigned* __restrict__ OCC, int* __restrict__ LIST, int* __restrict__ CNT,
                                              int T, int NPq, int nqt, int nw, int nkb, int window, int sink) {
  __shared__ int red[8];
  const int kb = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  const int K0 = kb * 128;
  const unsigned* OCCb = OCC + ((size_t)bh * nkb + kb) * nw;
  const unsigned pmsk = NPq >= 32 ? ~0u : ((1u << NPq) - 1u);
  int* out = LIST + ((size_t)bh * nkb + kb) * nqt;
  int carry = 0;
  for (int bs = K0 / NPq; bs < nqt; bs += 256) {
    const int qt = bs + tid;
    int f = 0;
    if (qt < nqt) {
      const int P0 = qt * NPq, P1 = min(P0 + NPq, T) - 1;
      if (P1 >= K0) {
        if (window > 0 && P0 <= K0 + 126 + window) f = 1;
        if (K0 < sink) f = 1;
        if (!f) {
          const int w0 = P0 >> 5;
          unsigned x = (__ldg(OCCb + w0) >> (P0 & 31)) & pmsk;
          if (NPq == 64 && w0 + 1 < nw) x |= __ldg(OCCb + w0 + 1);
          f = x != 0u;
        }
      }
    }
    int tot;
    const int pre = block_scan256(f, tot, red);
    if (f) out[carry + pre] = qt;
    carry += tot;
  }
  if (tid == 0) CNT[bh * nkb + kb] = carry;
}

__global__ void __launch_bounds__(256) items_k(const int* __restrict__ CNT, int4* __restrict__ ITEMS, int* __restrict__ NIT,
                                               int n, int nkb, int minL, int target) {
  __shared__ int red[8];
  __shared__ long long sred[8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  long long tot = 0;
  for (int i = tid; i < n; i += 256) tot += CNT[i];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffff, tot, off);
  if (lane == 0) sred[warp] = tot;
  __syncthreads();
  tot = 0;
  for (int k = 0; k < 8; ++k) tot += sred[k];
  const int L = max((long long)minL, (tot + target - 1) / target);
  int carryF = 0;
  for (int bs = 0; bs < n; bs += 256) {
    const int i = bs + tid;
    const int c = i < n ? CNT[i] : 0;
    int t1;
    const int pf = block_scan256(c / L, t1, red);
    for (int m = 0; m < c / L; ++m) ITEMS[carryF + pf + m] = make_int4(i / nkb, i % nkb, m * L, (m + 1) * L);
    carryF += t1;
  }
  int ntail = 0;
  for (int i = tid; i < n; i += 256) {
    const int c = CNT[i], r = c % L;
    if (r == 0) continue;
    int rank = 0;
    for (int jx = 0; jx < n; ++jx) {
      const int r2 = CNT[jx] % L;
      rank += (r2 > r) || (r2 == r && jx < i);
    }
    ITEMS[carryF + rank] = make_int4(i / nkb, i % nkb, (c / L) * L, c);
    ntail++;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ntail += __shfl_xor_sync(0xffffffff, ntail, off);
  __syncthreads();
  if (lane == 0) red[warp] = ntail;
  __syncthreads();
  if (tid == 0) {
    int s = 0;
    for (int k = 0; k < 8; ++k) s += red[k];
    NIT[0] = carryF + s;
  }
}

__device__ __forceinline__ unsigned long long rmask64(int a, int b) {
  const int n = b - a + 1;
  return (n >= 64 ? ~0ull : ((1ull << n) - 1ull)) << a;
}

// ---- v2 planning: chunked bitmap build (any T), band-major work items ----
__global__ void __launch_bounds__(1024) build2_k(const int* __restrict__ sel, unsigned* __restrict__ ST, unsigned* __restrict__ OCC,
                                                 int T, int TOPK, int nw, int nkb, int CH) {
  extern __shared__ unsigned sm[];
  const int w = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  const int P0 = w * 32, np = min(32, T - P0);
  const int Kmax = min(T, ((P0 + 32 + 127) / 128) * 128);
  const int* sb = sel + ((size_t)bh * T + P0) * TOPK;
  const int n = np * TOPK;
  unsigned* out = ST + ((size_t)bh * nw + w) * T;
  const int warp = tid >> 5, lane = tid & 31;
  for (int c0 = 0; c0 < Kmax; c0 += CH) {
    const int cl = min(CH, Kmax - c0);
    for (int i = tid; i < cl; i += 1024) sm[i] = 0u;
    __syncthreads();
    for (int x = tid; x < n; x += 1024) {
      const int j = __ldg(sb + x) - c0;
      if (j >= 0 && j < cl) atomicOr(&sm[j], 1u << (x / TOPK));
    }
    __syncthreads();
    for (int i = tid; i < cl; i += 1024) out[c0 + i] = sm[i];
    for (int kl = warp; kl * 64 < cl; kl += 32) {
      unsigned v = 0;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int jl = kl * 64 + e * 32 + lane;
        if (jl < cl) v |= sm[jl];
      }
      v = __reduce_or_sync(0xffffffff, v);
      if (lane == 0) OCC[((size_t)bh * 2 * nkb + c0 / 64 + kl) * nw + w] = v;
    }
    __syncthreads();
  }
}

// per (key tile, kv head): ascending list of active query tiles, and the list offset at each band start
__global__ void __launch_bounds__(256) list2_k(const unsigned* __restrict__ OCC, int* __restrict__ LIST, int* __restrict__ BST,
                                               int T, int NPq, int nqt, int nw, int nkb, int nb, int BQ, int window, int sink,
                                               int* __restrict__ BSK) {
  __shared__ int red[8];
  const int kb = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  const int K0 = kb * 128;
  const unsigned* OCCb = OCC + ((size_t)bh * 2 * nkb + 2 * kb) * nw;
  int* bsk = BSK + ((size_t)bh * nkb + kb) * (nb + 1);
  const unsigned pmsk = NPq >= 32 ? ~0u : ((1u << NPq) - 1u);
  int* out = LIST + ((size_t)bh * nkb + kb) * nqt;
  int* bst = BST + ((size_t)bh * nkb + kb) * (nb + 1);
  const int b0 = (K0 / NPq) / BQ;
  if (tid == 0) for (int be = 0; be < b0; ++be) { bst[be] = 0; bsk[be] = 0; }
  int carry = 0;
  for (int be = b0; be < nb; ++be) {
    if (tid == 0) bst[be] = carry;
    int skc = 0;
    for (int sub = 0; sub < BQ; sub += 256) {
      const int qt = be * BQ + sub + tid;
      int f = 0, sk = 0;
      if (qt < nqt && sub + tid < BQ) {
        const int P0 = qt * NPq, P1 = min(P0 + NPq, T) - 1;
        if (P1 >= K0) {
          const int w0 = P0 >> 5;
          int a[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int h0 = K0 + 64 * h, h1 = h0 + 63;
            int ah = 0;
            if (h0 < T && h0 <= P1) {
              if (window > 0 && h1 > P0 - window) ah = 1;
              if (h0 < sink) ah = 1;
              if (!ah) ah = ((__ldg(OCCb + (size_t)h * nw + w0) >> (P0 & 31)) & pmsk) != 0u;
            }
            a[h] = ah;
          }
          f = a[0] | a[1];
          sk = f && !(a[0] && a[1]);
        }
      }
      int tot;
      const int pre = block_scan256(f, tot, red);
      if (f) out[carry + pre] = qt;
      carry += tot;
      int tsk;
      block_scan256(sk, tsk, red);
      skc += tsk;
    }
    if (tid == 0) bsk[be] = skc;
  }
  if (tid == 0) { bst[nb] = carry; bsk[nb] = 0; }
}

__global__ void __launch_bounds__(1024) items2_k(const int* __restrict__ BST, int4* __restrict__ ITEMS, int* __restrict__ NIT,
                                                 int BH, int nkb, int nb, int rot, const int* __restrict__ BSK) {
  __shared__ int red[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nseg = nb * BH * nkb;
  int carry = 0;
  for (int bs = 0; bs < nseg; bs += 1024) {
    const int i = bs + tid;
    int a = 0, e = 0, bh = 0, kb = 0, skf = 0;
    if (i < nseg) {
      const int be = i / (BH * nkb), r = i % (BH * nkb);
      bh = r / nkb; kb = r % nkb;
      const int* bst = BST + ((size_t)bh * nkb + kb) * (nb + 1);
      a = bst[be]; e = bst[be + 1];
      const int nsk = BSK[((size_t)bh * nkb + kb) * (nb + 1) + be];
      skf = 8 * nsk >= (e - a) ? 1 : 0;      // skip-capable loop when >= 1/8 of the entries have an empty half
    }
    const int f = e > a;
    int x = f;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) { int y = __shfl_up_sync(0xffffffff, x, off); if (lane >= off) x += y; }
    if (lane == 31) red[warp] = x;
    __syncthreads();
    int pre = 0, tot = 0;
    for (int k = 0; k < 32; ++k) { int rr = red[k]; if (k < warp) pre += rr; tot += rr; }
    __syncthreads();
    if (f) ITEMS[carry + pre + x - f] = make_int4(bh | (skf << 30), kb, a, e);
    carry += tot;
  }
  if (tid == 0) NIT[0] = carry;
}

// ---- v1 main kernel: D=128, G=8 (NPq=8), 128-key tile, 2 consumer WGs (64 keys each) ----
// changes vs bwd_r1: double-buffered dS (no pre-store barrier), dQ GEMM issued before waiting on dK/dV,
// dQ(g) stored to smem while S/dP(g+1) run, dedicated dQ-writer warp, single loader warp.
// ---- v7: v6b ping-pong schedule + instruction diet ----
// * WG index is a compile-time constant (generic lambda), descriptors = precomputed base + constant
// * dQ fp32 accumulator kept in MMA-fragment order in global memory: STS.128 staging, one 1-D bulk
//   reduce per entry, unpermuted (and converted) by cvt_dq_k
// * softmax scale folded into P (LD.x carries -log2(scale)): dS, dK, dQ come out scaled; dV / scale
// * dS stored with stmatrix
struct V7Cfg {
  static constexpr int D = 128, NS = 2;
  static constexpr int KB = 128 * D * 2;            // 32 KB
  static constexpr int QB = 64 * D * 2;             // 16 KB
  static constexpr int DSB = 128 * 64 * 2;          // 16 KB
  static constexpr int DQB = 64 * 128 * 4;          // 32 KB per entry (both WGs)
  static constexpr int OFF_K = 0, OFF_V = KB, OFF_Q = 2 * KB, OFF_DO = 2 * KB + NS * QB;
  static constexpr int OFF_DS = 2 * KB + 2 * NS * QB;
  static constexpr int OFF_DQ = OFF_DS + 2 * DSB;
  static constexpr int OFF_MASK = OFF_DQ + 2 * DQB;
  static constexpr int OFF_LD = OFF_MASK + NS * 128;
  static constexpr int SMEM = OFF_LD + NS * 512;
};
#ifndef PROF
#define PROF 0
#endif
#if PROF
__device__ long long g_prof[8][2][256][16];
#endif
__device__ __forceinline__ unsigned long long dadd(unsigned long long d, unsigned off) { return d + (unsigned long long)(off >> 4); }
__device__ __forceinline__ void bulk_reduce_f32(float* gdst, unsigned ssrc, unsigned bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
               :: "l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void stsm_x4(unsigned addr, unsigned a, unsigned b, unsigned c, unsigned d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void st_shared_f4(unsigned addr, float a, float b, float c, float d) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

__global__ void __launch_bounds__(384, 1) bwd_main(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV,
    const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmDO,
    const int4* __restrict__ items, const int* __restrict__ NIT, const int* __restrict__ LIST, const unsigned* __restrict__ ST,
    const float2* __restrict__ LD, float* __restrict__ dqp, float* __restrict__ dk32, float* __restrict__ dv32,
    int T, int HKV, int nqt, int nw, int nkb, int window, int sink, float sl2, float vscale) {
  using C = V7Cfg;
  constexpr int D = 128, NS = 2, G = 8, NPq = 8, NO = 64;
  constexpr int KPANEL = 128 * 128, QPANEL = 64 * 128;
  if ((int)blockIdx.x >= __ldg(NIT)) return;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[7 + 2 * NS];
  __shared__ int sFlag[NS];
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem_raw);
  if (sbase & 1023u) __trap();
  unsigned char* gbase = smem_raw;
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned kvfull = bar0;
  auto dqfull = [&](int i) { return bar0 + 8 * (1 + i); };
  auto dqempty = [&](int i) { return bar0 + 8 * (3 + i); };
  auto dsfree = [&](int i) { return bar0 + 8 * (5 + i); };
  auto full = [&](int s) { return bar0 + 8 * (7 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (7 + NS + s); };
  const unsigned sK = sbase + C::OFF_K, sV = sbase + C::OFF_V;
  auto sQ = [&](int s) { return sbase + C::OFF_Q + s * C::QB; };
  auto sDO = [&](int s) { return sbase + C::OFF_DO + s * C::QB; };
  unsigned char* sMask = gbase + C::OFF_MASK;
  const float* sLDg = reinterpret_cast<const float*>(gbase + C::OFF_LD);
  const unsigned sLD = sbase + C::OFF_LD;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    mbar_init(kvfull, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(dqfull(i), 256); mbar_init(dqempty(i), 1); mbar_init(dsfree(i), 256); }
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 32); mbar_init(empty(s), 256); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int4 itm = items[blockIdx.x];
  const int bh = itm.x & 0x3fffffff, kb = itm.y, ls0 = itm.z, ls1 = itm.w;
  const bool useSkip = (itm.x >> 30) & 1;
  const int nent = ls1 - ls0;
  const int b = bh / HKV, hk = bh % HKV;
  const int K0 = kb * 128;
  const int* lst = LIST + ((size_t)bh * nkb + kb) * nqt + ls0;

  if (warp >= 8) {
    reg_dealloc<40>();
    if (warp == 8) {
      if (lane == 0) {
        mbar_expect_tx(kvfull, 2 * C::KB);
#pragma unroll
        for (int p = 0; p < D / 64; ++p) {
          tma_load_4d(sK + p * KPANEL, &tmK, p * 64, hk, K0, b, kvfull);
          tma_load_4d(sV + p * KPANEL, &tmV, p * 64, hk, K0, b, kvfull);
        }
      }
      const unsigned* STb = ST + (size_t)bh * nw * T;
      unsigned nx[4];
      auto ldst = [&](int qtx, unsigned* a) {
        const unsigned* sp = STb + (size_t)((qtx * NPq) >> 5) * T;
#pragma unroll
        for (int r = 0; r < 4; ++r) { const int j = K0 + lane + 32 * r; a[r] = j < T ? __ldg(sp + j) : 0u; }
      };
      int qn = nent > 0 ? __ldg(lst) : 0;
      if (nent > 0) ldst(qn, nx);
      for (int e = 0; e < nent; ++e) {
        const int qt2 = qn;
        unsigned cw[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) cw[r] = nx[r];
        if (e + 1 < nent) { qn = __ldg(lst + e + 1); ldst(qn, nx); }
        const int s = e % NS, u = e / NS;
        const int P0 = qt2 * NPq;
        const int Pl = min(P0 + NPq, T) - 1;
        if (e >= NS) mbar_wait(empty(s), (u - 1) & 1);
        unsigned anyb[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int j = K0 + lane + 32 * r;
          unsigned long long pm = 0ull;
          if (j < T) {
            pm = (cw[r] >> (P0 & 31)) & 0xffu;
            if (window > 0) {
              const int lo = max(j, P0), hi = min(j + window - 1, Pl);
              if (lo <= hi) pm |= rmask64(lo - P0, hi - P0);
            }
            if (j < sink) {
              const int lo = max(j, P0), hi = Pl;
              if (lo <= hi) pm |= rmask64(lo - P0, hi - P0);
            }
          }
          sMask[s * 128 + lane + 32 * r] = (unsigned char)(pm & 0xffull);
          anyb[r] = (unsigned)(pm & 0xffull);
        }
        {
          const unsigned h0 = __ballot_sync(0xffffffffu, (anyb[0] | anyb[1]) != 0u);
          const unsigned h1 = __ballot_sync(0xffffffffu, (anyb[2] | anyb[3]) != 0u);
          if (lane == 0) sFlag[s] = (h0 ? 1 : 0) | (h1 ? 2 : 0);
        }
        if (lane == 0) {
          mbar_expect_tx(full(s), 2 * C::QB + 512);
#pragma unroll
          for (int p = 0; p < D / 64; ++p) {
            tma_load_4d(sQ(s) + p * QPANEL, &tmQ, p * 64, hk * G, P0, b, full(s));
            tma_load_4d(sDO(s) + p * QPANEL, &tmDO, p * 64, hk * G, P0, b, full(s));
          }
          bulk_load(sLD + s * 512, LD + (size_t)bh * nqt * 64 + (size_t)qt2 * 64, 512, full(s));
        } else {
          mbar_arrive(full(s));
        }
      }
    } else if (warp == 9 && lane == 0) {
      // dQ writer: one 32 KB fragment-ordered tile per entry, 1-D bulk reduce-add
      float* dqbase = dqp + (size_t)bh * nqt * 8192;
      for (int h = 0; h < nent; ++h) {
        const int qq = __ldg(lst + h);
        mbar_wait(dqfull(h & 1), (h >> 1) & 1);
        bulk_reduce_f32(dqbase + (size_t)qq * 8192, sbase + C::OFF_DQ + (h & 1) * C::DQB, C::DQB);
        bulk_commit();
        bulk_wait_read1();
        if (h >= 1) mbar_arrive(dqempty((h - 1) & 1));
      }
      bulk_wait0();
    }
    return;
  }

  reg_alloc<232>();
  const int wq = warp & 3, qd = lane & 3;
  const int t128 = tid & 127;
  auto consumer = [&](auto CWc) {
    constexpr int c = decltype(CWc)::value;
#if PROF
    const int pnit = __ldg(NIT), pb = (int)blockIdx.x - pnit / 2;
#define ST_(k) do { if (pb >= 0 && pb < 8 && lane == 0 && wq == 0 && g < 256) g_prof[pb][c][g][k] = clock64(); } while (0)
#else
#define ST_(k) do {} while (0)
#endif
    const int k0 = c * 64 + wq * 16 + (lane >> 2);
    float dk[NO], dv[NO];
#pragma unroll
    for (int i = 0; i < NO; ++i) { dk[i] = 0.f; dv[i] = 0.f; }
    constexpr int bar_mine = 2 + c, bar_other = 3 - c;
    bool first = (c == 0);
    // descriptor bases
    const unsigned long long dK_S = make_desc(sK + c * 64 * 128, 16, 1024);
    const unsigned long long dV_S = make_desc(sV + c * 64 * 128, 16, 1024);
    const unsigned long long dKt = make_desc(sK + c * KPANEL, KPANEL, 1024);
    const unsigned long long dQ0 = make_desc(sQ(0), 16, 1024), dO0 = make_desc(sDO(0), 16, 1024);
    const unsigned long long dQt0 = make_desc(sQ(0), QPANEL, 1024), dOt0 = make_desc(sDO(0), QPANEL, 1024);
    const unsigned long long dS0 = make_desc(sbase + C::OFF_DS, 8192, 1024);
    constexpr unsigned QSTEP = C::QB >> 4, DSSTEP = C::DSB >> 4;
    // stmatrix row address pieces: matrix i = lane / 8 (h = i & 1, n-offset = i >> 1), row j = lane % 8
    const int sm_i = lane >> 3, sm_j = lane & 7;
    const unsigned sm_row = (unsigned)((c * 64 + wq * 16 + (sm_i & 1) * 8 + sm_j) * 128);
    // dQ staging: thread's 32 floats as 8 x 16 B chunks, chunk j at (j ^ (t & 7))
    const unsigned dq_off = (unsigned)(c * 16384 + t128 * 128);
    const int dq_x = t128 & 7;
    if (nent > 0) mbar_wait(kvfull, 0);
    auto store_dq = [&](int h, const float* dqa) {
      if (h >= 2) mbar_wait(dqempty(h & 1), ((h >> 1) - 1) & 1);
      const unsigned sDQ = sbase + C::OFF_DQ + (h & 1) * C::DQB + dq_off;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        st_shared_f4(sDQ + ((j ^ dq_x) << 4), dqa[4 * j], dqa[4 * j + 1], dqa[4 * j + 2], dqa[4 * j + 3]);
      fence_proxy_async();
      mbar_arrive(dqfull(h & 1));
    };
    auto body_full = [&](int g) {
      ST_(0);
      const int s = g % NS;
      mbar_wait(full(s), (g / NS) & 1);
      ST_(1);
      const unsigned m0 = sMask[s * 128 + k0], m1 = sMask[s * 128 + k0 + 8];
      const bool isfull = __all_sync(0xffffffff, (m0 & m1) == 0xffu);
      const unsigned long long dQs = dQ0 + s * QSTEP, dOs = dO0 + s * QSTEP;
      const unsigned long long dQts = dQt0 + s * QSTEP, dOts = dOt0 + s * QSTEP;
      float sacc[32], pacc[32];
      // B1: S(g), dP(g)
      if (!first) bar_sync(bar_mine, 256);
      first = false;
      ST_(2);
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk)
        wg_ss_n64(sacc, dadd(dK_S, (kk >> 2) * KPANEL + (kk & 3) * 32), dadd(dQs, (kk >> 2) * QPANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
      wg_commit();
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk)
        wg_ss_n64(pacc, dadd(dV_S, (kk >> 2) * KPANEL + (kk & 3) * 32), dadd(dOs, (kk >> 2) * QPANEL + (kk & 3) * 32), kk > 0 ? 1 : 0);
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      ST_(3);
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
      ST_(4);
      unsigned pa[4][4], dsa[4][4];
      const float* LDs = sLDg + s * 128;
      float pv[32];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float4 L = *reinterpret_cast<const float4*>(LDs + 2 * (8 * n + 2 * qd));
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
      ST_(5);
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
      ST_(6);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float4 L = *reinterpret_cast<const float4*>(LDs + 2 * (8 * n + 2 * qd));
        const float d0 = pv[4 * n] * (pacc[4 * n] - L.y), d1 = pv[4 * n + 1] * (pacc[4 * n + 1] - L.w);
        const float d2 = pv[4 * n + 2] * (pacc[4 * n + 2] - L.y), d3 = pv[4 * n + 3] * (pacc[4 * n + 3] - L.w);
        dsa[n >> 1][(n & 1) * 2] = pack_bf16(d0, d1);
        dsa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(d2, d3);
      }
      if (g >= 2) mbar_wait(dsfree(g & 1), ((g >> 1) - 1) & 1);
      {
        const unsigned sd = sbase + C::OFF_DS + (g & 1) * C::DSB + sm_row;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int n = 2 * q + (sm_i >> 1);
          stsm_x4(sd + ((n ^ sm_j) << 4), dsa[q][0], dsa[q][1], dsa[q][2], dsa[q][3]);
        }
      }
      fence_proxy_async();
      ST_(7);
      // B2: dQ(g-1) (dummy at g = 0), dV(g), dK(g)
      bar_sync(bar_mine, 256);
      ST_(8);
      float dqa[32];
      wg_fence();
      {
        const unsigned long long dSp = dS0 + ((g + 1) & 1) * DSSTEP;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) wg_ss_n64_tt(dqa, dadd(dSp, kk * 2048), dadd(dKt, kk * 2048), kk > 0 ? 1 : 0);
      }
      wg_commit();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg_rs_nD(dv, pa[kk], dadd(dOts, kk * 2048), 1);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wg_rs_nD(dk, dsa[kk], dadd(dQts, kk * 2048), 1);
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      ST_(9);
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(dqa[i]);
      if (g > 0) {
        mbar_arrive(dsfree((g - 1) & 1));
        store_dq(g - 1, dqa);
      }
      ST_(10);
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { fence_ureg(pa[i][jj]); fence_ureg(dsa[i][jj]); }
#pragma unroll
      for (int i = 0; i < NO; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
      mbar_arrive(empty(s));
      ST_(12);
    };
    auto body_skip = [&](int g) {
      const int s = g % NS;
      mbar_wait(full(s), (g / NS) & 1);
      if (!first) bar_sync(bar_mine, 256);
      first = false;
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      if (g >= 2) mbar_wait(dsfree(g & 1), ((g >> 1) - 1) & 1);
      {
        const unsigned sd = sbase + C::OFF_DS + (g & 1) * C::DSB + sm_row;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int n = 2 * q + (sm_i >> 1);
          stsm_x4(sd + ((n ^ sm_j) << 4), 0u, 0u, 0u, 0u);
        }
      }
      fence_proxy_async();
      bar_sync(bar_mine, 256);
      float dqa[32];
      wg_fence();
      {
        const unsigned long long dSp = dS0 + ((g + 1) & 1) * DSSTEP;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) wg_ss_n64_tt(dqa, dadd(dSp, kk * 2048), dadd(dKt, kk * 2048), kk > 0 ? 1 : 0);
      }
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(dqa[i]);
      if (g > 0) {
        mbar_arrive(dsfree((g - 1) & 1));
        store_dq(g - 1, dqa);
      }
      mbar_arrive(empty(s));
    };
    if (useSkip) {
      for (int g = 0; g < nent; ++g) {
        mbar_wait(full(g % NS), (g / NS) & 1);
        const bool act = __any_sync(0xffffffffu, (sFlag[g % NS] >> c) & 1);
        if (act) body_full(g);
        else body_skip(g);
      }
    } else {
      for (int g = 0; g < nent; ++g) body_full(g);
    }
    if (nent > 0) {
      float dqa[32];
      bar_sync(bar_mine, 256);
      wg_fence();
      {
        const unsigned long long dSp = dS0 + ((nent - 1) & 1) * DSSTEP;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) wg_ss_n64_tt(dqa, dadd(dSp, kk * 2048), dadd(dKt, kk * 2048), kk > 0 ? 1 : 0);
      }
      wg_commit();
      asm volatile("bar.arrive %0, %1;\n" :: "n"(bar_other), "n"(256) : "memory");
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(dqa[i]);
      store_dq(nent - 1, dqa);
      if (c == 0) bar_sync(2, 256);
      const long kvb = ((long)b * T * HKV + hk) * D;
      const int j0 = K0 + k0, j1 = j0 + 8;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const int col = 8 * n + 2 * qd;
        if (j0 < T) {
          atomicAdd(reinterpret_cast<float2*>(dk32 + kvb + (long)j0 * HKV * D + col), make_float2(dk[4 * n], dk[4 * n + 1]));
          atomicAdd(reinterpret_cast<float2*>(dv32 + kvb + (long)j0 * HKV * D + col), make_float2(dv[4 * n] * vscale, dv[4 * n + 1] * vscale));
        }
        if (j1 < T) {
          atomicAdd(reinterpret_cast<float2*>(dk32 + kvb + (long)j1 * HKV * D + col), make_float2(dk[4 * n + 2], dk[4 * n + 3]));
          atomicAdd(reinterpret_cast<float2*>(dv32 + kvb + (long)j1 * HKV * D + col), make_float2(dv[4 * n + 2] * vscale, dv[4 * n + 3] * vscale));
        }
      }
    }
  };
  if (warp < 4) consumer(std::integral_constant<int, 0>{});
  else consumer(std::integral_constant<int, 1>{});
}

// dQ: fragment-ordered fp32 tile (64 rows x 128 cols per (bh, qt)) -> bf16 (b, t, hq, d), via smem
__global__ void __launch_bounds__(256) cvt_dq2_k(const float* __restrict__ dqp, bf16* __restrict__ dq, int T, int HKV, int nqt) {
  __shared__ float tile[64][129];
  const int qt = blockIdx.x, bh = blockIdx.y;
  const int b = bh / HKV, hk = bh % HKV;
  const int tid = threadIdx.x, c = tid >> 7, t128 = tid & 127;
  const float* src = dqp + ((size_t)bh * nqt + qt) * 8192 + c * 4096 + t128 * 32;
  const int wq = t128 >> 5, lane = t128 & 31, qd = lane & 3, dq_x = t128 & 7;
  const int r0 = wq * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 v = *reinterpret_cast<const float4*>(src + ((j ^ dq_x) << 2));
    const int col = c * 64 + 8 * j + 2 * qd;
    tile[r0][col] = v.x; tile[r0][col + 1] = v.y; tile[r1][col] = v.z; tile[r1][col + 1] = v.w;
  }
  __syncthreads();
  const int HQ = HKV * 8;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int ch = tid + 256 * k;            // 1024 chunks of 8 bf16
    const int r = ch >> 4, c8 = (ch & 15) * 8;
    const int q = qt * 8 + (r >> 3), h = hk * 8 + (r & 7);
    if (q < T) {
      uint4 o;
      o.x = pack_bf16(tile[r][c8], tile[r][c8 + 1]); o.y = pack_bf16(tile[r][c8 + 2], tile[r][c8 + 3]);
      o.z = pack_bf16(tile[r][c8 + 4], tile[r][c8 + 5]); o.w = pack_bf16(tile[r][c8 + 6], tile[r][c8 + 7]);
      *reinterpret_cast<uint4*>(dq + (((size_t)b * T + q) * HQ + h) * 128 + c8) = o;
    }
  }
}

// dQ: fragment-ordered fp32 accumulator -> bf16 (b, t, hq, d). One thread per (bh, qt, wg c, t128).
__global__ void __launch_bounds__(256) cvt_dq_k(const float* __restrict__ dqp, bf16* __restrict__ dq, int T, int HKV, int nqt, long nthreads) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nthreads) return;
  const int t128 = idx & 127, c = (idx >> 7) & 1;
  const long rest = idx >> 8;
  const int qt = rest % nqt;
  const long bh = rest / nqt;
  const int b = bh / HKV, hk = bh % HKV;
  const float* src = dqp + (size_t)bh * nqt * 8192 + (size_t)qt * 8192 + c * 4096 + t128 * 32;
  const int wq = t128 >> 5, lane = t128 & 31, qd = lane & 3, dq_x = t128 & 7;
  const int r0 = wq * 16 + (lane >> 2), r1 = r0 + 8;
  const int HQ = HKV * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 v = *reinterpret_cast<const float4*>(src + ((j ^ dq_x) << 2));
    const int col = c * 64 + 8 * j + 2 * qd;
    const int q0 = qt * 8 + (r0 >> 3), h0 = hk * 8 + (r0 & 7);
    const int q1 = qt * 8 + (r1 >> 3), h1 = hk * 8 + (r1 & 7);
    if (q0 < T) *reinterpret_cast<unsigned*>(dq + (((size_t)b * T + q0) * HQ + h0) * 128 + col) = pack_bf16(v.x, v.y);
    if (q1 < T) *reinterpret_cast<unsigned*>(dq + (((size_t)b * T + q1) * HQ + h1) * 128 + col) = pack_bf16(v.z, v.w);
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
struct MapCache { void* p = nullptr; int B = 0, T = 0, H = 0, D = 0, bh = 0, bt = 0; bool f = false; CUtensorMap m; };
static const CUtensorMap& get_map(MapCache& c, void* ptr, int B, int T, int H, int D, int bh, int bt, bool f32 = false) {
  if (c.p == ptr && c.B == B && c.T == T && c.H == H && c.D == D && c.bh == bh && c.bt == bt && c.f == f32) return c.m;
  const int es_ = f32 ? 4 : 2;
  cuuint64_t dims[4] = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T, (cuuint64_t)B};
  cuuint64_t strides[3] = {(cuuint64_t)D * es_, (cuuint64_t)H * D * es_, (cuuint64_t)T * H * D * es_};
  cuuint32_t box[4] = {(cuuint32_t)(f32 ? 32 : 64), (cuuint32_t)bh, (cuuint32_t)bt, 1};
  cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult r = get_encode()(&c.m, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "tensor map encode failed ", (int)r);
  c.p = ptr; c.B = B; c.T = T; c.H = H; c.D = D; c.bh = bh; c.bt = bt; c.f = f32;
  return c.m;
}
static void launch_main(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& dout,
                        const int4* items, const int* nit, const int* LIST, const unsigned* ST, const float2* LD,
                        float* dqp, float* dk32, float* dv32, int grid, int nqt, int nw, int nkb, int window, int sink,
                        float sl2, float vscale, cudaStream_t stream) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), HKV = k.size(2), G = HQ / HKV, D = 128;
  constexpr int SM = V7Cfg::SMEM;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(bwd_main, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); init = true; }
  static MapCache ck, cv, cq, cdo;
  const CUtensorMap& tk = get_map(ck, k.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tv = get_map(cv, v.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tq = get_map(cq, q.data_ptr(), B, T, HQ, D, G, 8);
  const CUtensorMap& tdo = get_map(cdo, dout.data_ptr(), B, T, HQ, D, G, 8);
  bwd_main<<<grid, 384, SM, stream>>>(tk, tv, tq, tdo, items, nit, LIST, ST, LD, dqp, dk32, dv32,
      T, HKV, nqt, nw, nkb, window, sink, sl2, vscale);
}
std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t target, int64_t minL) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int HKV = k.size(2), TOPK = sel.size(3), G = HQ / HKV;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long nq = q.numel(), nkv = k.numel();
  const bool fast = D == 128 && G == 8;
  if (!fast) {
    auto dq32 = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat32));
    auto dk32 = torch::zeros(k.sizes(), q.options().dtype(torch::kFloat32));
    auto dv32 = torch::zeros(v.sizes(), q.options().dtype(torch::kFloat32));
    ref_bwd_kernel<<<dim3(T, HQ, B), D, 0, stream>>>(reinterpret_cast<const bf16*>(q.data_ptr()),
        reinterpret_cast<const bf16*>(k.data_ptr()), reinterpret_cast<const bf16*>(v.data_ptr()), sel.data_ptr<int>(),
        reinterpret_cast<const bf16*>(o.data_ptr()), lse.data_ptr<float>(), reinterpret_cast<const bf16*>(dout.data_ptr()),
        dq32.data_ptr<float>(), dk32.data_ptr<float>(), dv32.data_ptr<float>(), T, HQ, HKV, D, TOPK, (int)window, (int)sink,
        (float)scale);
    return {dq32.to(torch::kBFloat16), dk32.to(torch::kBFloat16), dv32.to(torch::kBFloat16)};
  }
  const int NPq = 64 / G, nqt = (T + NPq - 1) / NPq, nw = (T + 31) / 32, nkb = (T + 127) / 128;
  const int BH = B * HKV, n = BH * nkb;
  const long nld = (long)BH * nqt * 64 * 2;
  const long ndqp = (long)BH * nqt * 8192;
  auto fbuf = torch::empty({ndqp + 2 * nkv + nld}, q.options().dtype(torch::kFloat32));
  float* dq32 = fbuf.data_ptr<float>();
  if (ndqp != nq) cudaMemsetAsync(dq32, 0, ndqp * 4, stream);
  float* dk32 = dq32 + ndqp;
  float* dv32 = dk32 + nkv;
  float* LDp = dv32 + nkv;
  if (T % NPq != 0) cudaMemsetAsync(LDp, 0, nld * 4, stream);
  const int BQ = (int)target, nb = (nqt + BQ - 1) / BQ;
  const long nseg = (long)nb * n;
  const long nitm = (nseg + 4) * 4;
  const long nST = (long)BH * nw * T, nOCC = (long)2 * n * nw, nLIST = (long)n * nqt;
  auto ibuf = torch::empty({nitm + 4 + nST + nOCC + nLIST + 2 * n * (nb + 1)}, q.options().dtype(torch::kInt32));
  int* ib = ibuf.data_ptr<int>();
  int4* items = reinterpret_cast<int4*>(ib);
  int* nit = ib + nitm;
  unsigned* ST = reinterpret_cast<unsigned*>(nit + 4);
  unsigned* OCC = ST + nST;
  int* LIST = reinterpret_cast<int*>(OCC + nOCC);
  int* CNT = LIST + nLIST;
  const long nthr = (long)B * T * HQ * (D / 8);
  delta_k<<<(int)((nthr + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const bf16*>(o.data_ptr()),
      reinterpret_cast<const bf16*>(dout.data_ptr()), lse.data_ptr<float>(), reinterpret_cast<float2*>(LDp),
      dq32, dk32, dv32, nkv / 8, B, T, HQ, HKV, G, D, nqt, 1.4426950408889634f, (float)std::log2(scale));
  const int CH = 32768;
  const int bsm = std::min(T, CH) * 4;
  static int bset = 0;
  if (bsm > 48 * 1024 && bsm > bset) { cudaFuncSetAttribute(build2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, bsm); bset = bsm; }
  build2_k<<<dim3(nw, BH), 1024, bsm, stream>>>(sel.data_ptr<int>(), ST, OCC, T, TOPK, nw, nkb, CH);
  int* BST = CNT;
  int* BSK = BST + n * (nb + 1);
  list2_k<<<dim3(nkb, BH), 256, 0, stream>>>(OCC, LIST, BST, T, NPq, nqt, nw, nkb, nb, BQ, (int)window, (int)sink, BSK);
  items2_k<<<1, 1024, 0, stream>>>(BST, items, nit, BH, nkb, nb, (int)minL, BSK);
  const int grid = (int)nseg;
  const float sl2 = (float)(scale * 1.4426950408889634);
  TORCH_CHECK(D == 128 && G == 8, "v1 supports D=128, G=8 only");
  launch_main(q, k, v, dout, items, nit, LIST, ST, reinterpret_cast<const float2*>(LDp), dq32, dk32, dv32,
              grid, nqt, nw, nkb, (int)window, (int)sink, sl2, (float)(1.0 / scale), stream);
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  cvt_k<<<(int)((nkv / 8 + 255) / 256), 256, 0, stream>>>(dq32, reinterpret_cast<bf16*>(dq.data_ptr()), 0,
      dk32, reinterpret_cast<bf16*>(dk.data_ptr()), dv32, reinterpret_cast<bf16*>(dv.data_ptr()), nkv / 8);
  const long ncv = (long)BH * nqt * 256;
  (void)ncv;
  cvt_dq2_k<<<dim3(nqt, BH), 256, 0, stream>>>(dq32, reinterpret_cast<bf16*>(dq.data_ptr()), T, HKV, nqt);
  return {dq, dk, dv};
}
"""


_RC = int(__import__("os").environ.get("SA_RCONS", "232"))
_RP = 40 if _RC <= 232 else 24


def _build():
    src = HEADER + _gen("wg_ss_n64", 64, False) + _gen("wg_ss_n64_tt", 64, False, 1, 1)
    src += _gen("wg_rs_n128_t", 128, True, 0, 1) + _gen("wg_rs_n64_t", 64, True, 0, 1)
    src += r"""
template <int NO> struct RsSel;
template <> struct RsSel<64> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n128_t(d, a, db, sd); } };
template <> struct RsSel<32> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n64_t(d, a, db, sd); } };
#define wg_rs_nD(d, a, db, sd) RsSel<NO>::run(d, a, db, sd)
"""
    return f"#define RCONS {_RC}\n#define RPROD {_RP}\n" + src + KERN


_ext = load_inline(
    name="lsa_bwd_v14",
    cpp_sources="std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, "
    "torch::Tensor o, torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale, "
    "int64_t target, int64_t minL);",
    cuda_sources=_build() + HOST,
    functions=["bwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
    verbose=bool(int(__import__("os").environ.get("SA_VERBOSE", "0"))),
)

import os as _os
_BQ_FORCED = "SA_BQ" in _os.environ
TARGET = int(_os.environ.get("SA_BQ", "256"))  # query tiles per band (default: 128 for t <= 16K, else 256)
MINL = 0


def attention_backward(q, k, v, selected, o, lse, do, window=0, sink=0, scale=None):
    D = q.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()
    if not o.is_contiguous(): o = o.contiguous()
    if not do.is_contiguous(): do = do.contiguous()
    if lse.dtype != torch.float32 or not lse.is_contiguous(): lse = lse.contiguous().float()
    if selected.dtype != torch.int32: selected = selected.to(torch.int32)
    if not selected.is_contiguous(): selected = selected.contiguous()
    bq = TARGET if _BQ_FORCED else (128 if q.shape[1] <= 16384 else 256)
    dq, dk, dv = _ext.bwd(q, k, v, selected, o, lse, do, int(window), int(sink), float(scale), bq, MINL)
    return dq, dk, dv
