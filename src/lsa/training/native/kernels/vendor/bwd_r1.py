# Vendored verbatim from sparse_attn/starts/bwd_r1.py
# (upstream commit fabbc3ef, md5 fea4ee68d32d83e118f8ab10c7b636e0); see __init__.py. Not edited here.
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
                        long nkv8, int B, int T, int HQ, int HKV, int G, int D, int nqt, float l2e) {
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
    LD[(size_t)(b * HKV + hk) * nqt * 64 + (size_t)pos * G + g] = make_float2(LSE[((size_t)b * HQ + h) * T + pos] * l2e, s);
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

template <int D, int NS>
struct BCfg {
  static constexpr int KB = 128 * D * 2;
  static constexpr int QB = 64 * D * 2;
  static constexpr int DSB = 128 * 64 * 2;
  static constexpr int DQB = 64 * 64 * 4;
  static constexpr int OFF_K = 0, OFF_V = KB, OFF_Q = 2 * KB, OFF_DO = 2 * KB + NS * QB, OFF_DS = 2 * KB + 2 * NS * QB;
  static constexpr int OFF_DQ = OFF_DS + DSB;
  static constexpr int OFF_MASK = OFF_DQ + 4 * DQB;
  static constexpr int OFF_LD = OFF_MASK + NS * 1024;
  static constexpr int SMEM = OFF_LD + NS * 512 + 1024;
};

template <int D, int NS>
__global__ void __launch_bounds__(384, 1) bwd_kv(
    const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV,
    const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmDO,
    const __grid_constant__ CUtensorMap tmDQ,
    const int4* __restrict__ items, const int* __restrict__ NIT, const int* __restrict__ LIST, const unsigned* __restrict__ ST,
    const float2* __restrict__ LD, float* __restrict__ dk32, float* __restrict__ dv32,
    int T, int HQ, int HKV, int G, int NPq, int nqt, int nw, int nkb, int window, int sink, float sl2, float scale) {
  using C = BCfg<D, NS>;
  constexpr int KPANEL = 128 * 128, QPANEL = 64 * 128, NO = D / 2;
  if ((int)blockIdx.x >= __ldg(NIT)) return;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) unsigned long long bars[5 + 2 * NS];
  __shared__ int sInfo[NS];
  __shared__ int sList[128];
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem_raw);
  const unsigned sbase = (base + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - base);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  const unsigned kvfull = bar0;
  auto dqfull = [&](int i) { return bar0 + 8 * (1 + i); };
  auto dqempty = [&](int i) { return bar0 + 8 * (3 + i); };
  auto full = [&](int s) { return bar0 + 8 * (5 + s); };
  auto empty = [&](int s) { return bar0 + 8 * (5 + NS + s); };
  auto sDQb = [&](int i, int c) { return sbase + C::OFF_DQ + (i * 2 + c) * C::DQB; };
  const unsigned sK = sbase + C::OFF_K, sV = sbase + C::OFF_V;
  auto sQ = [&](int s) { return sbase + C::OFF_Q + s * C::QB; };
  auto sDO = [&](int s) { return sbase + C::OFF_DO + s * C::QB; };
  unsigned long long* sMask = reinterpret_cast<unsigned long long*>(gbase + C::OFF_MASK);
  const float* sLDg = reinterpret_cast<const float*>(gbase + C::OFF_LD);
  const unsigned sLD = sbase + C::OFF_LD;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    mbar_init(kvfull, 1);
    for (int i = 0; i < 2; ++i) { mbar_init(dqfull(i), D == 128 ? 256 : 128); mbar_init(dqempty(i), 1); }
    for (int s = 0; s < NS; ++s) { mbar_init(full(s), 128); mbar_init(empty(s), 256); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int4 itm = items[blockIdx.x];
  const int bh = itm.x, kb = itm.y, ls0 = itm.z, ls1 = itm.w;
  const int b = bh / HKV, hk = bh % HKV;
  const int K0 = kb * 128;

  if (warp >= 8) {
    reg_dealloc<40>();
    const int ptid = tid - 256;
    if (ptid == 0) {
      mbar_expect_tx(kvfull, 2 * C::KB);
#pragma unroll
      for (int p = 0; p < D / 64; ++p) {
        tma_load_4d(sK + p * KPANEL, &tmK, p * 64, hk, K0, b, kvfull);
        tma_load_4d(sV + p * KPANEL, &tmV, p * 64, hk, K0, b, kvfull);
      }
    }
    const int j = K0 + ptid;
    const bool jv = j < T;
    const unsigned* STb = ST + (size_t)bh * nw * T + (jv ? j : 0);
    const int* lst = LIST + ((size_t)bh * nkb + kb) * nqt;
    const unsigned pmsk = NPq >= 32 ? ~0u : ((1u << NPq) - 1u);
    int qh0 = 0, qh1 = 0, qh2 = 0, qh3 = 0;
    auto handle = [&](int h) {
      const int i = h & 1;
      const int qq = (h & 3) == 0 ? qh0 : (h & 3) == 1 ? qh1 : (h & 3) == 2 ? qh2 : qh3;
      mbar_wait(dqfull(i), (h >> 1) & 1);
      if (D == 128) {
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          tma_reduce_4d(&tmDQ, cc * 64, hk * G, qq * NPq, b, sDQb(i, cc));
          tma_reduce_4d(&tmDQ, cc * 64 + 32, hk * G, qq * NPq, b, sDQb(i, cc) + 8192);
        }
      } else {
        tma_reduce_4d(&tmDQ, 0, hk * G, qq * NPq, b, sDQb(i, i));
        tma_reduce_4d(&tmDQ, 32, hk * G, qq * NPq, b, sDQb(i, i) + 8192);
      }
      bulk_commit();
      bulk_wait_read1();
      if (h >= 1) mbar_arrive(dqempty((h - 1) & 1));
    };
    auto ldst = [&](int qtx, unsigned& a0, unsigned& a1) {
      const int w0 = (qtx * NPq) >> 5;
      const unsigned* sp = STb + (size_t)w0 * T;
      a0 = __ldg(sp);
      a1 = (NPq == 64 && w0 + 1 < nw) ? __ldg(sp + T) : 0u;
    };
    int g = 0;
    for (int bs = ls0; bs < ls1; bs += 128) {
      const int nb = min(128, ls1 - bs);
      if (ptid < nb) sList[ptid] = __ldg(lst + bs + ptid);
      bar_sync(2, 128);
      unsigned nx0 = 0u, nx1 = 0u;
      ldst(sList[0], nx0, nx1);
      for (int e = 0; e < nb; ++e) {
        const int qt2 = sList[e];
        const unsigned cw0 = nx0, cw1 = nx1;
        if (e + 1 < nb) ldst(sList[e + 1], nx0, nx1);
        const int s = g % NS, u = g / NS;
        const int P0 = qt2 * NPq;
        unsigned long long pm = 0ull;
        if (jv) {
          if (NPq == 64) pm = (unsigned long long)cw0 | ((unsigned long long)cw1 << 32);
          else pm = (cw0 >> (P0 & 31)) & pmsk;
          const int Pl = min(P0 + NPq, T) - 1;
          if (window > 0) {
            const int lo = max(j, P0), hi = min(j + window - 1, Pl);
            if (lo <= hi) pm |= rmask64(lo - P0, hi - P0);
          }
          if (j < sink) {
            const int lo = max(j, P0), hi = Pl;
            if (lo <= hi) pm |= rmask64(lo - P0, hi - P0);
          }
        }
        unsigned long long rm;
        if (G == 1) rm = pm;
        else {
          rm = 0ull;
          const unsigned long long gm = G >= 64 ? ~0ull : ((1ull << G) - 1ull);
          for (int p = 0; p < NPq; ++p)
            if ((pm >> p) & 1ull) rm |= gm << (p * G);
        }
        if (g >= NS) mbar_wait(empty(s), (u - 1) & 1);
        sMask[s * 128 + ptid] = rm;
        if (ptid == 0) {
          *((volatile int*)&sInfo[s]) = qt2;
          mbar_expect_tx(full(s), 2 * C::QB + 512);
#pragma unroll
          for (int p = 0; p < D / 64; ++p) {
            tma_load_4d(sQ(s) + p * QPANEL, &tmQ, p * 64, hk * G, P0, b, full(s));
            tma_load_4d(sDO(s) + p * QPANEL, &tmDO, p * 64, hk * G, P0, b, full(s));
          }
          bulk_load(sLD + s * 512, LD + (size_t)bh * nqt * 64 + (size_t)qt2 * 64, 512, full(s));
          switch (g & 3) { case 0: qh0 = qt2; break; case 1: qh1 = qt2; break; case 2: qh2 = qt2; break; default: qh3 = qt2; }
          if (g >= 2) handle(g - 2);
        } else mbar_arrive(full(s));
        ++g;
      }
      bar_sync(2, 128);
    }
    {
      const int s = g % NS, u = g / NS;
      if (g >= NS) mbar_wait(empty(s), (u - 1) & 1);
      if (ptid == 0) *((volatile int*)&sInfo[s]) = -1;
      mbar_arrive(full(s));
    }
    if (ptid == 0) {
      for (int h = max(0, g - 2); h < g; ++h) handle(h);
      bulk_wait0();
    }
    return;
  }

  reg_alloc<232>();
  const int c = warp >> 2, wq = warp & 3, qd = lane & 3;
  const int k0 = c * 64 + wq * 16 + (lane >> 2);
  float dk[NO], dv[NO];
#pragma unroll
  for (int i = 0; i < NO; ++i) { dk[i] = 0.f; dv[i] = 0.f; }
  const int r0 = wq * 16 + (lane >> 2), r1 = r0 + 8;
  const unsigned sd = sbase + C::OFF_DS;
  bool started = false;
  for (int g = 0;; ++g) {
    const int s = g % NS;
    mbar_wait(full(s), (g / NS) & 1);
    const int qt = *((volatile int*)&sInfo[s]);
    if (qt < 0) break;
    if (!started) { mbar_wait(kvfull, 0); started = true; }
    const unsigned long long w0 = sMask[s * 128 + k0], w1 = sMask[s * 128 + k0 + 8];
    const bool isfull = __all_sync(0xffffffff, w0 == ~0ull && w1 == ~0ull);
    const unsigned long long m0 = w0 >> (2 * qd), m1 = w1 >> (2 * qd);
    float sacc[32], pacc[32];
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      unsigned long long da = make_desc(sK + (kk >> 2) * KPANEL + c * 64 * 128 + (kk & 3) * 32, 16, 1024);
      unsigned long long db = make_desc(sQ(s) + (kk >> 2) * QPANEL + (kk & 3) * 32, 16, 1024);
      wg_ss_n64(sacc, da, db, kk > 0 ? 1 : 0);
    }
    wg_commit();
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      unsigned long long da = make_desc(sV + (kk >> 2) * KPANEL + c * 64 * 128 + (kk & 3) * 32, 16, 1024);
      unsigned long long db = make_desc(sDO(s) + (kk >> 2) * QPANEL + (kk & 3) * 32, 16, 1024);
      wg_ss_n64(pacc, da, db, kk > 0 ? 1 : 0);
    }
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sacc[i]);
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
        if (!((m0 >> (8 * n)) & 1ull)) p0 = 0.f;
        if (!((m0 >> (8 * n + 1)) & 1ull)) p1 = 0.f;
        if (!((m1 >> (8 * n)) & 1ull)) p2 = 0.f;
        if (!((m1 >> (8 * n + 1)) & 1ull)) p3 = 0.f;
      }
      pv[4 * n] = p0; pv[4 * n + 1] = p1; pv[4 * n + 2] = p2; pv[4 * n + 3] = p3;
      pa[n >> 1][(n & 1) * 2] = pack_bf16(p0, p1);
      pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(pacc[i]);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      unsigned long long db = make_desc(sDO(s) + kk * 16 * 128, QPANEL, 1024);
      wg_rs_nD(dv, pa[kk], db, 1);
    }
    wg_commit();
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float2 Ld = *reinterpret_cast<const float2*>(LDs + 2 * (8 * n + 2 * qd));
      const float2 Le = *reinterpret_cast<const float2*>(LDs + 2 * (8 * n + 2 * qd) + 2);
      const float d0 = pv[4 * n] * (pacc[4 * n] - Ld.y), d1 = pv[4 * n + 1] * (pacc[4 * n + 1] - Le.y);
      const float d2 = pv[4 * n + 2] * (pacc[4 * n + 2] - Ld.y), d3 = pv[4 * n + 3] * (pacc[4 * n + 3] - Le.y);
      dsa[n >> 1][(n & 1) * 2] = pack_bf16(d0, d1);
      dsa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(d2, d3);
    }
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      unsigned long long db = make_desc(sQ(s) + kk * 16 * 128, QPANEL, 1024);
      wg_rs_nD(dk, dsa[kk], db, 1);
    }
    wg_commit();
    bar_sync(4, 256);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const unsigned off = ((n ^ (k0 & 7)) << 4) + qd * 4;
      st_shared_u32(sd + k0 * 128 + off, dsa[n >> 1][(n & 1) * 2]);
      st_shared_u32(sd + (k0 + 8) * 128 + off, dsa[n >> 1][(n & 1) * 2 + 1]);
    }
    fence_proxy_async();
    bar_sync(1, 256);
    float dqa[32];
    const bool doq = (D == 128) || ((g & 1) == c);
    const int cp = D == 128 ? c : 0;
    if (doq) {
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        unsigned long long da = make_desc(sd + kk * 16 * 128, 8192, 1024);
        unsigned long long db = make_desc(sK + cp * KPANEL + kk * 16 * 128, KPANEL, 1024);
        wg_ss_n64_tt(dqa, da, db, kk > 0 ? 1 : 0);
      }
      wg_commit();
      wg_wait<1>();
    } else {
      wg_wait<0>();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) { fence_ureg(pa[i][jj]); fence_ureg(dsa[i][jj]); }
    mbar_arrive(empty(s));
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < NO; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(dqa[i]);
    if (doq) {
      const int bi = g & 1;
      if (g >= 2) mbar_wait(dqempty(bi), ((g >> 1) - 1) & 1);
      const unsigned sDQ = sDQb(bi, c);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int lc = 8 * n + 2 * qd;
        const int bx = lc >> 5, wi = lc & 31, ch = wi >> 2, of = (wi & 3) * 4;
        st_shared_f2(sDQ + bx * 8192 + r0 * 128 + ((ch ^ (r0 & 7)) << 4) + of, dqa[4 * n] * scale, dqa[4 * n + 1] * scale);
        st_shared_f2(sDQ + bx * 8192 + r1 * 128 + ((ch ^ (r1 & 7)) << 4) + of, dqa[4 * n + 2] * scale, dqa[4 * n + 3] * scale);
      }
      fence_proxy_async();
      mbar_arrive(dqfull(bi));
    }
  }
  if (started) {
    const long kvb = ((long)b * T * HKV + hk) * D;
    const int j0 = K0 + k0, j1 = j0 + 8;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      const int col = 8 * n + 2 * qd;
      if (j0 < T) {
        atomicAdd(reinterpret_cast<float2*>(dk32 + kvb + (long)j0 * HKV * D + col), make_float2(dk[4 * n] * scale, dk[4 * n + 1] * scale));
        atomicAdd(reinterpret_cast<float2*>(dv32 + kvb + (long)j0 * HKV * D + col), make_float2(dv[4 * n], dv[4 * n + 1]));
      }
      if (j1 < T) {
        atomicAdd(reinterpret_cast<float2*>(dk32 + kvb + (long)j1 * HKV * D + col), make_float2(dk[4 * n + 2] * scale, dk[4 * n + 3] * scale));
        atomicAdd(reinterpret_cast<float2*>(dv32 + kvb + (long)j1 * HKV * D + col), make_float2(dv[4 * n + 2], dv[4 * n + 3]));
      }
    }
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
template <int D, int NS>
static void launch_kv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& dout,
                      const int4* items, const int* nit, const int* LIST, const unsigned* ST, const float2* LD,
                      float* dq32, float* dk32, float* dv32, int grid, int NPq, int nqt, int nw, int nkb, int window, int sink,
                      float sl2, float scale, cudaStream_t stream) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), HKV = k.size(2), G = HQ / HKV;
  constexpr int SM = BCfg<D, NS>::SMEM;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(bwd_kv<D, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); init = true; }
  static MapCache ck, cv, cq, cdo, cdq;
  const CUtensorMap& tk = get_map(ck, k.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tv = get_map(cv, v.data_ptr(), B, T, HKV, D, 1, 128);
  const CUtensorMap& tq = get_map(cq, q.data_ptr(), B, T, HQ, D, G, NPq);
  const CUtensorMap& tdo = get_map(cdo, dout.data_ptr(), B, T, HQ, D, G, NPq);
  const CUtensorMap& tdq = get_map(cdq, dq32, B, T, HQ, D, G, NPq, true);
  bwd_kv<D, NS><<<grid, 384, SM, stream>>>(tk, tv, tq, tdo, tdq, items, nit, LIST, ST, LD, dk32, dv32,
      T, HQ, HKV, G, NPq, nqt, nw, nkb, window, sink, sl2, scale);
}
std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t target, int64_t minL) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int HKV = k.size(2), TOPK = sel.size(3), G = HQ / HKV;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long nq = q.numel(), nkv = k.numel();
  const bool fast = (D == 64 || D == 128) && G >= 1 && G <= 64 && (64 % G) == 0 && (long)T * 4 <= 200 * 1024;
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
  auto fbuf = torch::empty({nq + 2 * nkv + nld}, q.options().dtype(torch::kFloat32));
  float* dq32 = fbuf.data_ptr<float>();
  float* dk32 = dq32 + nq;
  float* dv32 = dk32 + nkv;
  float* LDp = dv32 + nkv;
  if (T % NPq != 0) cudaMemsetAsync(LDp, 0, nld * 4, stream);
  const long nitm = ((long)target + n + 4) * 4;
  const long nST = (long)BH * nw * T, nOCC = (long)n * nw, nLIST = (long)n * nqt;
  auto ibuf = torch::empty({nitm + 4 + nST + nOCC + nLIST + n}, q.options().dtype(torch::kInt32));
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
      dq32, dk32, dv32, nkv / 8, B, T, HQ, HKV, G, D, nqt, 1.4426950408889634f);
  const int bsm = T * 4;
  static int bset = 0;
  if (bsm > 48 * 1024 && bsm > bset) { cudaFuncSetAttribute(build_k, cudaFuncAttributeMaxDynamicSharedMemorySize, bsm); bset = bsm; }
  build_k<<<dim3(nw, BH), 256, bsm, stream>>>(sel.data_ptr<int>(), ST, OCC, T, TOPK, nw, nkb);
  list_k<<<dim3(nkb, BH), 256, 0, stream>>>(OCC, LIST, CNT, T, NPq, nqt, nw, nkb, (int)window, (int)sink);
  items_k<<<1, 256, 0, stream>>>(CNT, items, nit, n, nkb, (int)minL, (int)target);
  const int grid = (int)target + n;
  const float sl2 = (float)(scale * 1.4426950408889634);
  if (D == 128) launch_kv<128, 2>(q, k, v, dout, items, nit, LIST, ST, reinterpret_cast<const float2*>(LDp), dq32, dk32, dv32,
                                  grid, NPq, nqt, nw, nkb, (int)window, (int)sink, sl2, (float)scale, stream);
  else launch_kv<64, 2>(q, k, v, dout, items, nit, LIST, ST, reinterpret_cast<const float2*>(LDp), dq32, dk32, dv32,
                        grid, NPq, nqt, nw, nkb, (int)window, (int)sink, sl2, (float)scale, stream);
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  cvt_k<<<(int)((nq / 8 + 255) / 256), 256, 0, stream>>>(dq32, reinterpret_cast<bf16*>(dq.data_ptr()), nq / 8,
      dk32, reinterpret_cast<bf16*>(dk.data_ptr()), dv32, reinterpret_cast<bf16*>(dv.data_ptr()), nkv / 8);
  return {dq, dk, dv};
}
"""


def _build():
    src = HEADER + _gen("wg_ss_n64", 64, False) + _gen("wg_ss_n64_tt", 64, False, 1, 1)
    src += _gen("wg_rs_n128_t", 128, True, 0, 1) + _gen("wg_rs_n64_t", 64, True, 0, 1)
    src += r"""
template <int NO> struct RsSel;
template <> struct RsSel<64> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n128_t(d, a, db, sd); } };
template <> struct RsSel<32> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n64_t(d, a, db, sd); } };
#define wg_rs_nD(d, a, db, sd) RsSel<NO>::run(d, a, db, sd)
"""
    return src + KERN


_ext = load_inline(
    name="lsa_bwd_r1",
    cpp_sources="std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, "
    "torch::Tensor o, torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale, "
    "int64_t target, int64_t minL);",
    cuda_sources=_build() + HOST,
    functions=["bwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"],
)

TARGET = 3 * 132
MINL = 24


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
    dq, dk, dv = _ext.bwd(q, k, v, selected, o, lse, do, int(window), int(sink), float(scale), TARGET, MINL)
    return dq, dk, dv
