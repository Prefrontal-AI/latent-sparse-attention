"""LSA's bucket ranking and key gather (lsa_select.select_indices) by FA3-style CUDA kernels.

lsa_select scores a tile of queries against the visible centroids into a (b, rows, m) fp32
buffer, then a program a row reloads its row and bisects it for the top_p-th score: at 32,768
tokens the second kernel is 80% of the selection (1.6 of 2.0 ms), at 65,536 74% (7.0 of 9.5).
Here the scores never leave the chip:

- kern, persistent, 384 threads a CTA, takes 128 query rows (one head) or 16 (eight heads,
  whose rows score max_h q_h . c), rows of one LSA block, so all of them see the same visible
  buckets, the first (i // block) * (block / bucket_size). Its producer warp loads the rows'
  queries and then each 64-bucket tile of the centroids, split in three bf16 parts as
  lsa_select's 'bf16x3' splits them, by TMA into a ring guarded by mbarriers; two consumer
  warpgroups score a tile by wgmma (A the queries, B the parts, lo then mid then hi into one fp32
  accumulator), so with bf16 queries a score is fp32-exact up to its summation order. Eight
  heads' max is a reduce-scatter over the accumulator's lanes.
- Pass 1 scores the hi part alone and histograms each row's scores in 256 bins over [-bound,
  bound], bound the row's largest query norm (centroids are unit or zero); the hi part is within
  half a bin of the full score, so the bin below the highest one with top_p scores at or above it
  holds every top_p score. Pass 2 scores all three parts and lists each score in a bin at or
  above that one, (score, bucket) as one 64-bit key, in the row's slot of a global buffer.
- select_kern, a warp a row at full occupancy, takes the listed buckets best first, by score
  and an exact tie to the lower bucket as lsa_select orders them, 128 at a time (a radix select
  of the chunk's last key, a warp bitonic sort of the chunk) while top_k is not full, and writes
  each pick's members, a bucket being pools of consecutive keys, in slot order and filtered to
  [sink_size, i - window_size].

A row whose listed candidates would pass CAP is flagged and redone by lsa_select (none on random
or trained index vectors at 32,768 tokens), so the selection is lsa_select's up to the scores'
last bits: every row identical on random and trained index vectors at 32,768 tokens.
lsa_select.redo_flagged redoes them on the device, so nothing
waits for the host; past lsa_select's MAX_BUCKETS the reference redoes them after one host sync.
The positions are int32, each row's distinct keys in [sink_size, i - window_size] first and -1 after, what
cuda_sparse.attention(sanitized=True) reads as they are. sm_90a, bf16 index queries 64 or 128
wide, one head or eight, block a multiple of 128, top_p up to MAX_CAP / 2, buckets of at most 8
pools and 16 keys, up to MAX_BUCKETS of them, past lsa_select's 8,192: a row's histogram packs
two 16-bit bins a word below WIDE_BUCKETS buckets and takes 32-bit bins from there, so the
fast path runs past 65,536 tokens, to a million and beyond.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from lsa.training.native.kernels import lsa_select
from lsa.training.native.kernels.vendor.fa3idx_common import HDR, wgmma_ss_fn

# Candidates a row lists, and so the most top_p (half of it) and a threshold bin's slack
CAP = 512
# A larger top_p lists twice it, rounded up to a power of two, up to MAX_CAP (see cap)
MAX_CAP = 2048
# The sequence length from which a fast-path LSA layer ranks here rather than by lsa_select
# (faster at 8 x 4,096: 0.53 against 0.64 ms one head, 0.65 against 0.86 eight)
RANK_TOKENS = 16384
HEADS = (1, 8)
HEAD_DIMS = (64, 128)
MAX_BUCKETS = 2**20
# Buckets from which a row's histogram takes 32-bit bins, past the 16-bit ones' count
WIDE_BUCKETS = 2**16
# Pools a bucket, whose first positions the selection loads for each pick
MAX_POOLS = 8

BODY = r"""
#define BN 64
#define NKC (D / 64)
#define NR (128 / H)
#define TILE_ELEMS (64 * 64)
#define TILE_BYTES (TILE_ELEMS * 2)
#define STAGE_BYTES (3 * NKC * TILE_BYTES)
#define Q_BYTES (2 * NKC * TILE_BYTES)
#if WIDE
// 32-bit bins, for rows that see 65,536 buckets or more
#define HW 256
#define HADD(r, bin) atomicAdd(&hist[(r) * HW + (bin)], 1u)
#define HGET(r, bin) hist[(r) * HW + (bin)]
#else
// Two 16-bit bins a word: exact while a row sees fewer than 65,536 buckets
#define HW 128
#define HADD(r, bin) atomicAdd(&hist[(r) * HW + ((bin) >> 1)], 1u << (((bin) & 1) * 16))
#define HGET(r, bin) ((hist[(r) * HW + ((bin) >> 1)] >> (((bin) & 1) * 16)) & 0xFFFFu)
#endif
#define CHUNK 128

struct Params {
  const float* rowa;     // (B, T) 128 / the row's score bound
  const long* members;   // (B, M, CAPM)
  const int* starts;     // (B, M, NS) a bucket's pools' first positions, -1 past its size
  int* out;              // (B, T, TOPK) positions, -1 past the taken keys
  unsigned long long* buf;  // (grid, NR, CAP) candidates: ordered score << 32 | ~bucket
  int* flag;             // (B, T) rows lsa_select redoes
  int* cnts;             // (B, T) a row's listed candidates, -1 where lsa_select redoes it
  int B, T, M, CAPM, TOPK, WINDOW, SINK, BLOCK, PERB, PICKS, NQT;
};

static CUtensorMap make_map(const void* ptr, long rows, int cols) {
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    void* fn = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    g_encode = (PFN_cuTensorMapEncodeTiled_v12000)fn;
  }
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, 64};
  cuuint32_t es[2] = {1, 1};
  CUresult r = g_encode(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
                        strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  return tm;
}

__device__ __forceinline__ unsigned ordered(float s) {
  unsigned u = __float_as_uint(s);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct Tile { int b, i0, V, nkt; };
__device__ __forceinline__ Tile tile_at(const Params& p, int L) {
  Tile w;
  w.b = L % p.B;
  w.i0 = (p.NQT - 1 - L / p.B) * NR;   // the widest rows first
  w.V = min((w.i0 / p.BLOCK) * p.PERB, p.M);
  w.nkt = (w.V + BN - 1) / BN;
  return w;
}

// One pass over a tile's centroid tiles: score, then hist (MODE 0) or list (MODE 1)
template <int MODE>
__device__ __forceinline__ void pass(const Params& p, const Tile& w, int& c, const bf16* ring,
                                     const bf16* qs, uint64_t* full, uint64_t* empty,
                                     unsigned* hist, int* cnt, const int* thr, const float* ra,
                                     unsigned long long* slot) {
  const int tid = threadIdx.x, cw = tid >> 7, wq = (tid >> 5) & 3, lane = tid & 31;
  const int g = lane >> 2, cq = lane & 3;
  // The thread's two accumulator rows, as CTA rows (H 8: the query rows its head max lands in)
  int R[2];
  if (H == 1) { R[0] = cw * 64 + wq * 16 + g; R[1] = R[0] + 8; }
  else { R[0] = cw * 8 + wq * 2; R[1] = R[0] + 1; }
  float A[2];
  int T0[2];
#pragma unroll
  for (int x = 0; x < 2; x++) { A[x] = ra[R[x]]; T0[x] = MODE ? thr[R[x]] : 0; }
  auto issue = [&](float (&acc)[BN / 2], int cc) {
    const int s = cc % NST;
    mbar_wait(&full[s], (cc / NST) & 1);
    wg_fence();
    // MODE 0 histograms the hi part's scores alone, within half a bin of the full score
#pragma unroll
    for (int part = MODE ? 2 : 0; part >= 0; part--)
#pragma unroll
      for (int kc = 0; kc < NKC; kc++) {
        const uint64_t qd = desc_sw128(qs + (cw * NKC + kc) * TILE_ELEMS);
        const uint64_t cd =
            desc_sw128(ring + (size_t)s * (STAGE_BYTES / 2) + (part * NKC + kc) * TILE_ELEMS);
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
          wgmma_ss_m64n64(acc, qd + 2 * kk, cd + 2 * kk,
                          (part == (MODE ? 2 : 0) && kc == 0 && kk == 0) ? 0 : 1);
      }
    wg_commit();
  };
  auto finish = [&](float (&acc)[BN / 2], int cc, int kt) {
    fence_regs(acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[cc % NST]);
    if (ABL & (1 << MODE)) return;
    const int jb = kt * BN;
    auto take = [&](float sc, int x, int j) {
      if (j >= w.V) return;
      const int bin = (int)fminf(fmaxf(fmaf(sc, A[x], 128.f), 0.f), 255.f);
      if (MODE == 0) {
        HADD(R[x], bin);
      } else if (bin >= T0[x]) {
        const int at = atomicAdd(&cnt[R[x]], 1);
        if (at < CAP)
          slot[R[x] * CAP + at] =
              ((unsigned long long)ordered(sc) << 32) | (0xFFFFFFFFu - (unsigned)j);
      }
    };
    if (H == 1) {
#pragma unroll
      for (int a = 0; a < BN / 8; a++)
#pragma unroll
        for (int x = 0; x < 2; x++)
#pragma unroll
          for (int e = 0; e < 2; e++) take(acc[4 * a + 2 * x + e], x, jb + a * 8 + 2 * cq + e);
    } else {
      // The max over the 8 heads (lane bits 2-4) as a reduce-scatter: each step halves the
      // column blocks a lane keeps, so lane g ends with block a = g, 4 values, in 28 shuffles
      float k4[16], k2[8], k1[4];
      const bool b2 = (g >> 2) & 1, b1 = (g >> 1) & 1, b0 = g & 1;
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float lo = acc[4 * a + u], hi = acc[4 * (a + 4) + u];
          k4[4 * a + u] = fmaxf(b2 ? hi : lo, __shfl_xor_sync(FULLM, b2 ? lo : hi, 16));
        }
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float lo = k4[4 * a + u], hi = k4[4 * (a + 2) + u];
          k2[4 * a + u] = fmaxf(b1 ? hi : lo, __shfl_xor_sync(FULLM, b1 ? lo : hi, 8));
        }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const float lo = k2[u], hi = k2[4 + u];
        k1[u] = fmaxf(b0 ? hi : lo, __shfl_xor_sync(FULLM, b0 ? lo : hi, 4));
      }
#pragma unroll
      for (int x = 0; x < 2; x++)
#pragma unroll
        for (int e = 0; e < 2; e++) take(k1[2 * x + e], x, jb + g * 8 + 2 * cq + e);
    }
  };
  // One accumulator: a second (the next tile's wgmma under this one's scores) and FA3's
  // ping-pong of the two warpgroups' issue both measured no faster
  for (int kt = 0; kt < w.nkt; kt++) {
    float acc[BN / 2];
    issue(acc, c + kt);
    wg_wait<0>();
    finish(acc, c + kt, kt);
  }
  c += w.nkt;
}

// Bitonic sort of a warp's 32 * KPL keys, descending; element e = lane * KPL + r
template <int KPL>
__device__ __forceinline__ void warp_sort_desc(unsigned long long (&v)[KPL], int lane) {
  constexpr int N = 32 * KPL;
#pragma unroll
  for (int k = 2; k <= N; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      if (j < KPL) {
#pragma unroll
        for (int r = 0; r < KPL; r++) {
          const int pr = r ^ j;
          if (pr > r) {
            const bool desc = ((lane * KPL + r) & k) == 0;
            const unsigned long long a = v[r], b = v[pr];
            if (desc ? (a < b) : (a > b)) { v[r] = b; v[pr] = a; }
          }
        }
      } else {
        const int lj = j / KPL;
        const bool lower = (lane & lj) == 0;
#pragma unroll
        for (int r = 0; r < KPL; r++) {
          const bool desc = ((lane * KPL + r) & k) == 0;
          const unsigned long long o = __shfl_xor_sync(FULLM, v[r], lj);
          const unsigned long long mx = v[r] > o ? v[r] : o, mn = v[r] > o ? o : v[r];
          v[r] = (lower == desc) ? mx : mn;
        }
      }
    }
  }
}
__global__ void __launch_bounds__(384, 1)
    kern(const __grid_constant__ CUtensorMap cm, const __grid_constant__ CUtensorMap qm, Params p) {
  extern __shared__ unsigned char smraw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  bf16* ring = (bf16*)sm;
  bf16* qs = (bf16*)(sm + NST * STAGE_BYTES);
  uint64_t* full = (uint64_t*)(sm + NST * STAGE_BYTES + Q_BYTES);
  uint64_t* empty = full + NST;
  uint64_t* qfull = empty + NST;
  uint64_t* qempty = qfull + 1;
  unsigned* hist = (unsigned*)(qempty + 1);
  int* cnt = (int*)(hist + NR * HW);
  int* thr = cnt + NR;
  float* ra = (float*)(thr + NR);
  const int tid = threadIdx.x, lane = tid & 31;
  const int ntiles = p.B * p.NQT;
  if (tid == 0) {
    for (int s = 0; s < NST; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8); }
    mbar_init(qfull, 1);
    mbar_init(qempty, 8);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (tid >= 256) {   // ---------------- producer ----------------
    if (tid != 256) return;
    int c = 0, tq = 0;
    for (int L = blockIdx.x; L < ntiles; L += gridDim.x, tq++) {
      const Tile w = tile_at(p, L);
      if (tq > 0) mbar_wait(qempty, (tq - 1) & 1);
      mbar_expect_tx(qfull, Q_BYTES);
      const int qrow = (w.b * p.T + w.i0) * H;
      for (int cw = 0; cw < 2; cw++)
        for (int kc = 0; kc < NKC; kc++)
          tma_load_2d(qs + (cw * NKC + kc) * TILE_ELEMS, &qm, kc * 64, qrow + cw * 64, qfull);
      for (int ps = 0; ps < 2; ps++)
        for (int kt = 0; kt < w.nkt; kt++, c++) {
          const int s = c % NST;
          if (c >= NST) mbar_wait(&empty[s], ((c / NST) - 1) & 1);
          mbar_expect_tx(&full[s], ps ? STAGE_BYTES : STAGE_BYTES / 3);
          for (int part = 0; part < (ps ? 3 : 1); part++)
            for (int kc = 0; kc < NKC; kc++)
              tma_load_2d(ring + (size_t)s * (STAGE_BYTES / 2) + (part * NKC + kc) * TILE_ELEMS,
                          &cm, kc * 64, (part * p.B + w.b) * p.M + kt * BN, &full[s]);
        }
    }
    return;
  }

  // ---------------- consumers ----------------
  const int gw = tid >> 5;
  int c = 0, tq = 0;
  for (int L = blockIdx.x; L < ntiles; L += gridDim.x, tq++) {
    const Tile w = tile_at(p, L);
    // Row r's candidates at buf[(b * T + i0 + r) * CAP]: rows past T list none
    unsigned long long* slot = p.buf + ((size_t)w.b * p.T + w.i0) * CAP;
    for (int x = tid; x < NR * HW; x += 256) hist[x] = 0u;
    if (tid < NR) {
      const int i = w.i0 + tid;
      cnt[tid] = 0;
      ra[tid] = i < p.T ? p.rowa[(size_t)w.b * p.T + i] : 0.f;
    }
    bar_sync(1, 256);
    mbar_wait(qfull, tq & 1);
    pass<0>(p, w, c, ring, qs, full, empty, hist, cnt, thr, ra, slot);
    bar_sync(1, 256);
    if (tid < NR) {
      // The highest bin with PICKS scores at or above it; every visible bucket if there are fewer
      const int i = w.i0 + tid, live = min(w.V, p.PICKS);
      int t0 = 0, cum = 0;
      if (w.V > p.PICKS) {
        for (int bin = 255; bin >= 0; bin--) {
          cum += HGET(tid, bin);
          if (cum >= live) { t0 = bin; break; }
        }
        // The hi part's score is within half a bin of the full one: a bin lower holds them all
        t0 = max(t0 - 1, 0);
        cum = 0;
        for (int bin = 255; bin >= t0; bin--)
          cum += HGET(tid, bin);
      } else cum = w.V;
      if (cum > CAP) {
        t0 = 256;
        if (i < p.T) p.flag[(size_t)w.b * p.T + i] = 1;
      }
      thr[tid] = i < p.T ? t0 : 256;
    }
    bar_sync(1, 256);
    pass<1>(p, w, c, ring, qs, full, empty, hist, cnt, thr, ra, slot);
    __syncwarp();
    if (lane == 0) mbar_arrive(qempty);
    bar_sync(1, 256);
    if (tid < NR) {
      const int i = w.i0 + tid;
      if (i < p.T) {
        // A row the threshold left to lsa_select, or whose full scores crossed past the cap
        const bool left = thr[tid] == 256 || cnt[tid] > CAP;
        if (left && thr[tid] != 256) p.flag[(size_t)w.b * p.T + i] = 1;
        p.cnts[(size_t)w.b * p.T + i] = left ? -1 : cnt[tid];
      }
    }
    bar_sync(1, 256);
  }
}


// A warp a row, as many warps as fit an SM: the row's listed buckets best first, their members
// in slot order up to TOPK keys. The picks come 128 at a time: a radix select of the chunk's last
// key over the listed ones left, a sort of the chunk, its keys written, and the next chunk only
// while TOPK is not full (about 128 picks fill 1,024 keys at 8 a bucket)
__global__ void __launch_bounds__(256) select_kern(Params p) {
  __shared__ unsigned sorted_all[8][256];
  __shared__ int bins_all[8][256];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * 8 + warp;
  if (row >= (long)p.B * p.T) return;
  const int b = (int)(row / p.T), i = (int)(row % p.T);
  const int V = min((i / p.BLOCK) * p.PERB, p.M);
  const unsigned long long* slot = p.buf + row * CAP;
    {
      const int n = p.cnts[row];
      if (n < 0 || (ABL & 4)) return;
      int* orow = p.out + row * p.TOPK;
      const int live = min(min(V, p.PICKS), n);
      // The chunk's buckets best first
      unsigned* sorted = sorted_all[warp];
      int* bins = bins_all[warp];
      unsigned long long v[CAP / 32];
#pragma unroll
      for (int q = 0; q < CAP / 32; q++) {
        const int at = lane * (CAP / 32) + q;
        v[q] = at < n ? slot[at] : 0ull;
      }
      const long lim = (long)i - p.WINDOW;
      const int lo = p.SINK;
      const int* srow = p.starts + (size_t)b * p.M * NS;
      int taken = 0;
      for (int base = 0; base < live && taken < p.TOPK; base += CHUNK) {
        const int need0 = min(CHUNK, live - base);
        // The need0-th largest key left, 8 bits a round from the top, until its digit holds
        // exactly the keys still needed (keys are distinct: the bucket is their low word)
        unsigned long long prefix = 0ull;
        int need = need0, sh = 64;
        while (true) {
          sh -= 8;
#pragma unroll
          for (int q = 0; q < 8; q++) bins[lane * 8 + q] = 0;
          __syncwarp();
#pragma unroll
          for (int q = 0; q < CAP / 32; q++)
            if (v[q] != 0ull && (sh == 56 || (v[q] >> (sh + 8)) == prefix))
              atomicAdd(&bins[(v[q] >> sh) & 255], 1);
          __syncwarp();
          int c8[8], ls = 0;
#pragma unroll
          for (int q = 0; q < 8; q++) { c8[q] = bins[255 - 8 * lane - q]; ls += c8[q]; }
          int inc = ls;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            const int o = __shfl_up_sync(FULLM, inc, off);
            if (lane >= off) inc += o;
          }
          const int exc = inc - ls;
          const int src = __ffs(__ballot_sync(FULLM, exc < need && need <= inc)) - 1;
          int dig = 0, above = 0, cntd = 0;
          if (lane == src) {
            int cum = exc;
#pragma unroll
            for (int q = 0; q < 8; q++) {
              if (cntd == 0 && cum + c8[q] >= need) {
                dig = 255 - 8 * lane - q;
                above = cum;
                cntd = c8[q];
              }
              cum += c8[q];
            }
          }
          dig = __shfl_sync(FULLM, dig, src);
          above = __shfl_sync(FULLM, above, src);
          cntd = __shfl_sync(FULLM, cntd, src);
          prefix = (prefix << 8) | (unsigned long long)dig;
          need -= above;
          __syncwarp();
          if (cntd == need || sh == 0) break;
        }
        const unsigned long long lowest = prefix << sh;
        // The chunk's keys to a scratch list (taken from v), sorted 4 a lane
        int mine = 0;
#pragma unroll
        for (int q = 0; q < CAP / 32; q++) mine += v[q] != 0ull && v[q] >= lowest;
        int inc = mine;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const int o = __shfl_up_sync(FULLM, inc, off);
          if (lane >= off) inc += o;
        }
        unsigned long long* list = reinterpret_cast<unsigned long long*>(bins);
        int o = inc - mine;
#pragma unroll
        for (int q = 0; q < CAP / 32; q++)
          if (v[q] != 0ull && v[q] >= lowest) { list[o++] = v[q]; v[q] = 0ull; }
        __syncwarp();
        unsigned long long c4[CHUNK / 32];
#pragma unroll
        for (int q = 0; q < CHUNK / 32; q++) {
          const int e = lane * (CHUNK / 32) + q;
          c4[q] = e < need0 ? list[e] : 0ull;
        }
        __syncwarp();
        warp_sort_desc<CHUNK / 32>(c4, lane);
#pragma unroll
        for (int q = 0; q < CHUNK / 32; q++) {
          const int e = lane * (CHUNK / 32) + q;
          if (e < need0) sorted[e] = 0xFFFFFFFFu - (unsigned)(c4[q] & 0xFFFFFFFFull);
        }
        __syncwarp();
        if (ABL & 8) continue;
        // The chunk's picks, 8 a step: lane l holds slots u * 32 + l of the step's 8 x 16, a
        // pool's first position each (a bucket is NS pools of POOL consecutive keys), loaded
        // three steps ahead; a ballot's prefix orders the kept keys, as lsa_select writes them
        const int steps = (need0 + 7) / 8;
        auto load = [&](int g, int (&a)[4]) {
#pragma unroll
          for (int u = 0; u < 4; u++) {
            const int e = u * 32 + lane, rk = g * 8 + (e >> 4), sl = e & 15;
            const bool on = g < steps && rk < need0 && sl < NS * POOL;
            a[u] = on ? __ldg(srow + (size_t)sorted[rk] * NS + sl / POOL) : -1;
          }
        };
        int buf4[4][4];
        load(0, buf4[0]);
        load(1, buf4[1]);
        load(2, buf4[2]);
        for (int g0 = 0; g0 < steps && taken < p.TOPK; g0 += 4) {
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            const int g = g0 + q4;
            if (g >= steps || taken >= p.TOPK) break;
            load(g + 3, buf4[(q4 + 3) & 3]);
#pragma unroll
            for (int u = 0; u < 4; u++) {
              const int a = buf4[q4][u];
              const long id = (long)a + ((u * 32 + lane) & 15) % POOL;
              const bool ok = a >= 0 && id >= lo && id <= lim;
              const unsigned bal = __ballot_sync(FULLM, ok);
              const int at = taken + __popc(bal & ((1u << lane) - 1u));
              if (ok && at < p.TOPK) orow[at] = (int)id;
              taken += __popc(bal);
            }
          }
        }
        __syncwarp();
      }
      if (ABL & 8) return;
      for (int x = min(taken, p.TOPK) + lane; x < p.TOPK; x += 32) orow[x] = -1;
    }
}

void run(torch::Tensor q, torch::Tensor parts, torch::Tensor rowa, torch::Tensor members,
         torch::Tensor starts, torch::Tensor out, torch::Tensor buf, torch::Tensor flag,
         torch::Tensor cnts, int64_t block, int64_t perb, int64_t picks, int64_t window_size,
         int64_t sink_size, int64_t grid) {
  Params p;
  p.B = q.size(0); p.T = q.size(1); p.M = members.size(1); p.CAPM = members.size(2);
  p.TOPK = out.size(2);
  p.WINDOW = window_size; p.SINK = sink_size; p.BLOCK = block; p.PERB = perb; p.PICKS = picks;
  p.NQT = (p.T + NR - 1) / NR;
  p.rowa = rowa.data_ptr<float>(); p.members = members.data_ptr<long>();
  p.starts = starts.data_ptr<int>(); p.out = out.data_ptr<int>();
  p.buf = (unsigned long long*)buf.data_ptr(); p.flag = flag.data_ptr<int>();
  p.cnts = cnts.data_ptr<int>();
  CUtensorMap cm = make_map(parts.data_ptr(), 3L * p.B * p.M, D);
  CUtensorMap qm = make_map(q.data_ptr(), (long)p.B * p.T * H, D);
  size_t smem = 1024 + NST * STAGE_BYTES + Q_BYTES + (2 * NST + 2) * 8 + NR * HW * 4 + 3 * NR * 4;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<grid, 384, smem, at::cuda::getCurrentCUDAStream()>>>(cm, qm, p);
  const long rows = (long)p.B * p.T;
  select_kern<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}
"""
CPP = (
    "void run(torch::Tensor q, torch::Tensor parts, torch::Tensor rowa, torch::Tensor members, "
    "torch::Tensor starts, torch::Tensor out, torch::Tensor buf, torch::Tensor flag, "
    "torch::Tensor cnts, int64_t block, int64_t perb, int64_t picks, int64_t window_size, int64_t sink_size, "
    "int64_t grid);"
)
_built: dict[tuple[int, ...], Any] = {}
_sms: dict[torch.device, int] = {}


def _stages(d: int, h: int, wide: bool = False) -> int:
    """Ring stages that fit the 227 KiB of shared memory beside the queries and histograms."""
    stage = 3 * (d // 64) * 8192
    rest = 2 * (d // 64) * 8192 + (128 // h) * (1024 if wide else 512) + 4096
    return min(4, (227 * 1024 - rest) // stage)


# Ablation bits: 1 skips pass 1's histogram, 2 pass 2's list, 4 the select kernel, 8 its writes
ABL = 0


def cap(top_p: int) -> int:
    """Return the candidates a row lists for top_p picks: CAP, or twice top_p up to a power of two."""
    return max(CAP, 1 << (2 * top_p - 1).bit_length())


def _ext(d: int, h: int, ns: int, pool: int, wide: bool = False, cap: int = CAP) -> Any:
    """Build the CUDA extension for one shape and cache it.

    Args:
        d: Index dim.
        h: Index heads.
        ns: Buckets scanned a pass.
        pool: Keys a bucket pools.
        wide: Whether the kernel is the wide variant.
        cap: Candidates a row lists.
    """
    key = (d, h, ns, pool, ABL, wide, cap)
    if key not in _built:
        defs = (
            f"#define D {d}\n#define H {h}\n#define NST {_stages(d, h, wide)}\n#define CAP {cap}\n"
            f"#define NS {ns}\n#define POOL {pool}\n#define ABL {ABL}\n#define WIDE {int(wide)}\n"
        )
        _built[key] = load_inline(
            name=f"lsa_lsa_rank_d{d}_h{h}_s{ns}_p{pool}_c{cap}_a{ABL}_w{int(wide)}",
            cpp_sources=CPP,
            cuda_sources=HDR + wgmma_ss_fn(64) + defs + BODY,
            functions=["run"],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--expt-relaxed-constexpr",
            ],
        )
    return _built[key]


def supported(
    index_queries: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor] | None,
    block: int,
    top_p: int,
    pool: int = 1,
) -> bool:
    """Check whether the kernel takes these index queries, buckets of pools of pool keys, and top_p."""
    if buckets is None or not index_queries.is_cuda:
        return False
    if torch.cuda.get_device_capability(index_queries.device) != (9, 0):
        return False
    heads = index_queries.shape[2] if index_queries.dim() == 4 else 1
    m, cap = buckets[1].shape[1:]
    return (
        index_queries.dtype == torch.bfloat16
        and index_queries.shape[-1] in HEAD_DIMS
        and heads in HEADS
        and block % 128 == 0
        and cap % pool == 0
        and cap // pool <= MAX_POOLS
        and cap <= 16
        and m <= MAX_BUCKETS
        and min(top_p, m) <= MAX_CAP // 2
        and buckets[1].dtype == torch.long
    )


def select_indices(
    index_queries: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor],
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    pool: int = 1,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Select a whole sequence's long-range keys as lsa_select.select_indices does (query i at position i), as supported takes.

    Args:
        pool: The pool size cluster made the buckets with: a bucket's members are runs of pool consecutive positions, which the kernel reads as their first positions.
        dtype: The positions' dtype, int64 or int32.
            In int32 they are what cuda_sparse.attention(sanitized=True) reads: each row's distinct keys in [sink_size, i - window_size] first, -1 after.
        Others: lsa_select.select_indices'.

    Returns:
        Positions [b, t, top_k], PAD past the taken keys.
    """
    centroids, members = buckets
    out, flag, buf = _launch(
        index_queries, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, pool
    )
    if lsa_select.supported(index_queries, buckets):
        # Rows whose threshold bin lists too many candidates: lsa_select's search, on the device
        lsa_select.redo_flagged(
            index_queries, flag, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, out, buf
        )  # fmt: skip
    elif flag.any():
        # Past lsa_select's kernels the reference redoes them, after one host sync
        for bi in range(out.shape[0]):
            redo = flag[bi].nonzero()[:, 0]
            if redo.numel():
                out[bi, redo] = lsa_select.select_indices(
                    index_queries[bi : bi + 1, redo], redo, (centroids[bi : bi + 1], members[bi : bi + 1]),
                    block, bucket_size, top_p, top_k, window_size, sink_size, dtype=out.dtype,
                )[0]  # fmt: skip
    return out if dtype == torch.int32 else out.to(dtype)


def flagged(
    index_queries: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor],
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    pool: int = 1,
) -> int:
    """Return how many rows the kernel leaves to lsa_select, for the benchmarks.

    Args:
        pool: The pool size cluster made the buckets with.
        Others: select_indices'.
    """
    flag = _launch(index_queries, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, pool)[1]
    return int(flag.sum())


@triton.jit
def _scale_kernel(
    Q: tl.tensor, OUT: tl.tensor, rows: tl.tensor, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr
) -> None:
    """128 / a row's score bound, its largest query norm with a 2^-10 margin, or 0 for none."""
    row = tl.program_id(0) * R + tl.arange(0, R)
    held = row < rows
    at = (row[:, None, None] * H + tl.arange(0, H)[None, :, None]) * D
    q = tl.load(Q + at + tl.arange(0, D)[None, None, :], mask=held[:, None, None], other=0.0)
    q = q.to(tl.float32)
    bound = tl.max(tl.sqrt(tl.sum(q * q, axis=2)), axis=1) * (1 + 2**-10)
    tl.store(OUT + row, tl.where(bound > 0, 128 / bound, 0.0), mask=held)


def _launch(
    index_queries: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor],
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    pool: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's (b, t, top_k) int32 positions, (b, t) flags of the rows it left, its buffer."""
    centroids, members = buckets
    b, t = index_queries.shape[:2]
    heads = index_queries.shape[2] if index_queries.dim() == 4 else 1
    d = index_queries.shape[-1]
    m = members.shape[1]
    q = index_queries.reshape(b, t, heads, d).contiguous()
    parts = lsa_select.split3(centroids)
    # Unit or zero centroids keep a score within the row's largest query norm
    # The bound only sets the histogram's bins, whose slack and margin cover its last bits
    # So the order its sum runs in leaves the selection as it is
    rowa = torch.empty(b, t, dtype=torch.float32, device=q.device)
    rows_at = max(1, 4096 // (heads * d))
    _scale_kernel[(triton.cdiv(b * t, rows_at),)](
        q, rowa, b * t, H=heads, D=d, R=rows_at, num_warps=4
    )  # fmt: skip
    ns = -(-(members.shape[2] // pool) // 4) * 4
    starts = members[..., ::pool].to(torch.int32)
    starts = torch.nn.functional.pad(starts, (0, ns - starts.shape[2]), value=-1).contiguous()
    out = torch.empty(b, t, top_k, dtype=torch.int32, device=q.device)
    flag = torch.zeros(b, t, dtype=torch.int32, device=q.device)
    dev = q.device
    if dev not in _sms:
        _sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    rows = 128 // heads
    grid = min(_sms[dev], b * -(-t // rows))
    # Every row's candidates, 8-byte keys: 128 MiB at 32,768 tokens and the CAP of 512
    listed = cap(min(top_p, m))
    buf = torch.empty(b * t * listed, dtype=torch.long, device=dev)
    cnts = torch.empty(b, t, dtype=torch.int32, device=dev)
    _ext(d, heads, ns, pool, m >= WIDE_BUCKETS, listed).run(
        q, parts, rowa, members, starts, out, buf, flag, cnts, block, block // bucket_size,
        min(top_p, m), window_size, sink_size, grid,
    )  # fmt: skip
    return out, flag, buf
