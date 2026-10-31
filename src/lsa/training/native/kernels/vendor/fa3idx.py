# Vendored from fa3idx.py (md5 2ca5f8f40c660141e5821b20ce680b47); see
# __init__.py. Edits are marked 'lsa:'.
"""fa3idx: an FA3-style top-k indexer for hq 16, hk 2, d 64 on H200 (sm_90a), with Opus v8d's
two-threshold selection run on the wgmma accumulators.

One CTA = one (batch, query-head pair of one KV group, 64-query tile); 384 threads, 2 CTAs per SM:
- warpgroup 2 is the producer: setmaxnreg drops it to 24 registers; its warp 8 lane 0 loads the
  two heads' Q tiles once and streams K tiles (BN keys x 64 dims bf16, 128-byte swizzle) with TMA
  (cp.async.bulk.tensor via a CUtensorMap) into an NST-stage ring guarded by full/empty
  mbarriers; warps 9-11 exit.
- warpgroups 0 and 1 are consumers (setmaxnreg raises them to 104 registers), one query head each.
  Both heads read the same K head, so each loaded tile serves 128 query rows. Each runs wgmma
  m64nBNk16 with A = its Q tile and B = the K tile, both from swizzled shared memory, and
  classifies the scores in the accumulator registers: two masks per row and lane (>= lo, >= hi)
  from set.ge + LOP3, one vote, one packed quad scan for all four streams (winners and band, two
  rows), then winners (>= hi) and band entries ([lo, hi), 16-bit band key << 16 | 16-bit id) go
  through per-row shared rings and leave for global memory as aligned 16-byte stores.
- The passes are v8d's: a sample pass (NSAMP tiles spread over the row's range, 16-bit packed
  histograms) sets each row's band [lo, hi); the main pass; a final 256-bin radix select per row
  over its buffer (staged in shared memory), stopping at the tolerance; rows whose band failed are
  redone by the multi-pass fallback (stats pass, up to MAXIT histogram refinements, a final
  classify pass) with idx_pp_start.py's fix for v8d's bug (more than top_k above the band: move
  the band up instead of selecting every candidate). When the sample cannot place an upper edge
  (expected rank of the k-th below ~1) the row has no hi: everything >= lo goes to the band.
- The producer follows the same pass sequence; it learns whether a fallback runs, and whether
  each refinement continues, through named barrier 2 (consumers arrive, producer warp syncs).
  Named barrier 1 syncs the 256 consumer threads.
"""
import math, os, sys
import torch
from torch.utils.cpp_extension import load_inline
from lsa.training.native.kernels.vendor.fa3idx_common import HDR, wgmma_ss_fn

BODY = r"""
#define NB 128
#define HW 66                 // 32-bit words per row of a 16-bit packed histogram (bins 0..NB)
#define MAXIT 6
#define MAGF 12582912.0f
#define MAGI 0x4B400000
#define NROW 128
#define WR 32
#define BR 32
#define RMASK 31
#define REGION 45056
// final-select staging entries per consumer warp: warps 0-3 stage in the (then idle) K ring,
// warps 4-7 in the region after the 256-bin histograms
#define STG_RING (NST * BN * 128 / 16)
#define STG_HIST ((REGION - 8192) / 16)
#define STG (STG_RING < STG_HIST ? STG_RING : STG_HIST)
__device__ int g_lock[8192];
__device__ unsigned long long g_dbg[16];
__device__ unsigned long long g_prof[8];

struct Params {
  int* out; unsigned* buf;
  int B, T, topk, window, sink, nqt, cap, capb, nsamp, tm, uni; float scale;
  int H, HK, G, RS, NPAIR, QT;   // lsa: query heads, key heads, group, consumer 1's row shift (0: a head pair, 64: a second q tile), CTAs per (b, q tile), rows per q tile
  int* uout; int* gcnt;   // group union: (B, 2, T, 8 topk) output, per (b, group, q tile) arrival counters   // cap: per-row stride = capb band + top_k winner entries
};
struct Smem {
  int lim[NROW], n[NROW];
  int segc[NROW][4], wcnt[NROW][4];
  int hinf[NROW], state[NROW], need[NROW], outcnt[NROW], bndcnt[NROW], bufcnt[NROW], sv[NROW], pend[NROW];
  float bA[NROW], bC[NROW], mA[NROW], mC[NROW], lor[NROW], hir[NROW], blo[NROW], bhi[NROW];
  float tmv[NROW];
  float fA[NROW], fC[NROW], fsel[NROW], fbnd[NROW], mx[NROW], mn[NROW], ssq[NROW], rms[NROW];
  int flag2, slot, iflag[MAXIT + 2], ulast, ucnt;
};

__device__ __forceinline__ int tile_of(int kt0, int span, int nt, int it) {
  return kt0 + (int)(((unsigned)(2 * it + 1) * (unsigned)span) / (unsigned)(2 * nt));
}
struct PassState {
  float A[2], C[2], Fs[2], Fb[2], lo[2], hi[2];
  float mx[2], mn[2], sq[2];
  int cnt[2];
  float tl[2][TMR], th[2];          // MODE 6: the lane's sorted top-TMR scores per row, and its rn-th
  int wv[2], wf[2], bv[2], bf[2];   // MODE 5: wv = the lane's segment fill count
  uint4 q4[2];                      // MODE 5: the lane's pending 4 entries per row
};
// Append one entry to the lane's segment: entries gather in a uint4 register and leave as one
// 16-byte store per 4 (aligned: segments start on 32-byte boundaries).
__device__ __forceinline__ void put4(PassState& ps, int x, unsigned* seg, int segcap, unsigned e) {
  const int sp = ps.wv[x], k = sp & 3;
  if (k == 0) ps.q4[x].x = e;
  if (k == 1) ps.q4[x].y = e;
  if (k == 2) ps.q4[x].z = e;
  if (k == 3) {
    ps.q4[x].w = e;
    if (sp < segcap && !(ABL & 1)) *reinterpret_cast<uint4*>(seg + sp - 3) = ps.q4[x];
  }
  ps.wv[x] = sp + 1;
}
__device__ __forceinline__ unsigned setge(float a, float b) {
  unsigned r; asm("set.ge.u32.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ void hadd(unsigned* hrow, int bin) { atomicAdd(hrow + (bin >> 1), 1u << ((bin & 1) << 4)); }
__device__ __forceinline__ int hget(const unsigned* hrow, int bin) { return (hrow[bin >> 1] >> ((bin & 1) << 4)) & 0xffff; }
__device__ __forceinline__ int quad_reserve(int* ctr, int c, int cq) {
  int incl = c;
  int v = __shfl_up_sync(FULLM, incl, 1, 4); if (cq >= 1) incl += v;
  v = __shfl_up_sync(FULLM, incl, 2, 4); if (cq >= 2) incl += v;
  int tot = __shfl_sync(FULLM, incl, 3, 4);
  int base = 0;
  if (cq == 0 && tot > 0) base = atomicAdd(ctr, tot);
  base = __shfl_sync(FULLM, base, 0, 4);
  return base + incl - c;
}
// exclusive prefix over the 4 lanes of a quad (packed byte fields, each total <= 64), and the total
__device__ __forceinline__ unsigned quad_scan(unsigned c, int cq, unsigned& tot) {
  unsigned incl = c;
  unsigned v = __shfl_up_sync(FULLM, incl, 1, 4); if (cq >= 1) incl += v;
  v = __shfl_up_sync(FULLM, incl, 2, 4); if (cq >= 2) incl += v;
  tot = __shfl_sync(FULLM, incl, 3, 4);
  return incl - c;
}
// Write ring entries of positions [f, f2) to ob[] (positions < lim only); the quad's 4 lanes take
// 4-entry groups, an aligned whole group as one 16-byte store. Returns max(f, f2).
__device__ __forceinline__ int flush_ring(const unsigned* ring, int* ob, int f, int f2, int lim, int cq) {
  if (f2 <= f) return f;
  for (int q = (f & ~3) + 4 * cq; q < f2; q += 16) {
    if (q >= f && q + 4 <= f2 && q + 4 <= lim) {
      if (!(ABL & 1)) *reinterpret_cast<uint4*>(ob + q) = *reinterpret_cast<const uint4*>(ring + (q & RMASK));
    } else {
#pragma unroll
      for (int j = 0; j < 4; j++) { const int e = q + j; if (e >= f && e < f2 && e < lim && !(ABL & 1)) ob[e] = (int)ring[e & RMASK]; }
    }
  }
  return f2;
}

struct Ctx {
  const Params* p; Smem* S; unsigned* hist; const bf16* ring; const bf16* qs; uint64_t* full; uint64_t* empty;
  int ct, cw, w, gw, lane, g, cq, i0;
  int* op; unsigned* bufp; unsigned* wring; unsigned* bring;
};

// Every mode but 5: v8d's per-row processing (MODE 4 sample stats + histogram, 0 stats, 1 fallback
// histogram, 3 fallback classify).
template <int MODE, bool MASK>
__device__ __forceinline__ void process(float (&acc)[BN / 2], PassState& ps, const Ctx& X, const int (&R)[2],
                                        const int (&lim)[2], int jb) {
  const Params& p = *X.p; Smem& S = *X.S;
  const int cq = X.cq;
#pragma unroll
  for (int x = 0; x < 2; x++) {
    const int r = R[x] & 63;
    unsigned selm = 0, bndm = 0;
    unsigned* hrow = X.hist + R[x] * HW;
#pragma unroll
    for (int a = 0; a < BN / 8; a++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        float v = acc[4 * a + 2 * x + e];
        const int idx = 2 * a + e;
        if (MASK) {
          int j = jb + a * 8 + 2 * cq + e;
          if (!(j >= p.sink && j <= lim[x])) v = -INFINITY;
        }
        if (MODE == 0 || MODE == 4 || MODE == 6) {
          float s = v * p.scale;
          if (v != -INFINITY) { ps.mx[x] = fmaxf(ps.mx[x], s); ps.mn[x] = fminf(ps.mn[x], s); ps.sq[x] = fmaf(s, s, ps.sq[x]); ps.cnt[x]++; }
          if (MODE == 4) {
            const float f = fmaf(v, ps.A[x], ps.C[x]);
            if (f >= MAGF && ps.A[x] > 0.f) hadd(hrow, __float_as_int(fminf(f, MAGF + (float)(NB - 1))) - MAGI);
          }
        } else if (MODE == 1) {
          float f = fmaf(v, ps.A[x], ps.C[x]);
          if (f >= MAGF) hadd(hrow, __float_as_int(fminf(f, MAGF + (float)NB)) - MAGI);
        } else if (MODE == 3) {
          float f = fmaf(v, ps.A[x], ps.C[x]);
          bool sel = f >= ps.Fs[x];
          bool bnd = (f >= ps.Fb[x]) && !sel;
          selm |= (sel ? 1u : 0u) << idx;
          bndm |= (bnd ? 1u : 0u) << idx;
        }
      }
    }
    if (MODE == 6) {
      // the lane's own top rn scores of the row, sorted in registers; values at or below the
      // lane's current rn-th skip (warp-uniform skip on the lane maxima first)
      float lm = -INFINITY;
#pragma unroll
      for (int a = 0; a < BN / 8; a++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float v = acc[4 * a + 2 * x + e];
          if (MASK) { const int j = jb + a * 8 + 2 * cq + e; if (!(j >= p.sink && j <= lim[x])) v = -INFINITY; }
          lm = fmaxf(lm, v);
        }
      if (__any_sync(FULLM, lm > ps.th[x])) {
        const int rn = (p.topk + 3) >> 2;
#pragma unroll
        for (int a = 0; a < BN / 8; a++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            float v = acc[4 * a + 2 * x + e];
            if (MASK) { const int j = jb + a * 8 + 2 * cq + e; if (!(j >= p.sink && j <= lim[x])) v = -INFINITY; }
            if (v > ps.th[x]) {
#pragma unroll
              for (int i = 0; i < TMR; i++) { const float hi_ = fmaxf(ps.tl[x][i], v); v = fminf(ps.tl[x][i], v); ps.tl[x][i] = hi_; }
              float t_ = ps.tl[x][0];
#pragma unroll
              for (int i = 1; i < TMR; i++) if (i == rn - 1) t_ = ps.tl[x][i];
              ps.th[x] = t_;
            }
          }
      }
    }
    if (MODE == 3) {
      while (bndm) {
        int idx = __ffs(bndm) - 1;
        bndm &= bndm - 1;
        if (atomicAdd(&S.bndcnt[R[x]], 1) < S.need[R[x]]) selm |= 1u << idx;
      }
      if (__any_sync(FULLM, selm != 0)) {
        int pos = quad_reserve(&S.outcnt[R[x]], __popc(selm), cq);
        int* orow = X.op + (size_t)(X.i0 + r) * p.topk;
        while (selm) {
          int idx = __ffs(selm) - 1;
          selm &= selm - 1;
          if (pos < p.topk) orow[pos] = jb + (idx >> 1) * 8 + 2 * cq + (idx & 1);
          pos++;
        }
      }
    }
  }
}

__device__ __forceinline__ int* out_base(const Ctx& X, int R, int& wo) {
  int* orow = X.op + (size_t)(X.i0 + (R & 63)) * X.p->topk;
  wo = (int)((reinterpret_cast<uintptr_t>(orow) >> 2) & 7);
  return orow - wo;   // 32-byte aligned; position v of the virtual row is orow[v - wo]
}

// MODE 5, the main pass. Per row and lane: two masks (>= lo, >= hi) from set.ge + LOP3. Each of
// the 4 lanes of a row's quad owns a private segment (SEGCAP entries) of the row's buffer and keeps
// its fill count in a register: no atomics, shuffles, votes or warp syncs. A winner (>= hi) is
// stored as 0xFFFF << 16 | id, a band entry as (16-bit band key, at most 0xFFFE) << 16 | id; the
// final select writes each output row once, contiguously.
template <bool MASK>
__device__ __forceinline__ void process5(float (&acc)[BN / 2], PassState& ps, const Ctx& X, const int (&R)[2],
                                         const int (&lim)[2], int jb, unsigned* const (&seg)[2]) {
  const Params& p = *X.p;
  const int cq = X.cq;
  unsigned hm[2] = {0u, 0u};
#pragma unroll
  for (int a = 0; a < BN / 8; a++)
#pragma unroll
    for (int x = 0; x < 2; x++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        float v = acc[4 * a + 2 * x + e];
        if (MASK) { const int j = jb + a * 8 + 2 * cq + e; if (!(j >= p.sink && j <= lim[x])) v = -INFINITY; }
        hm[x] |= setge(v, ps.lo[x]) & (1u << (2 * a + e));
      }
  if (ABL & 8) { if ((hm[0] ^ hm[1]) == 0xFFFFFFFFu) X.hist[0] = hm[1]; return; }
  if (!(hm[0] | hm[1])) return;
  // stage this lane's scores of the rows with hits; then one loop over the hits of both rows
  // (interleaved for ILP): a hit >= hi is a winner (0xFFFF << 16 | id), else a band entry
  float4* st = reinterpret_cast<float4*>(X.hist) + X.ct;
#pragma unroll
  for (int x = 0; x < 2; x++)
    if (hm[x]) {
#pragma unroll
      for (int k4 = 0; k4 < BN / 16; k4++)
        st[(x * (BN / 16) + k4) * 256] = make_float4(acc[8 * k4 + 2 * x], acc[8 * k4 + 2 * x + 1], acc[8 * k4 + 4 + 2 * x], acc[8 * k4 + 5 + 2 * x]);
    }
  const float* sv = reinterpret_cast<const float*>(X.hist) + X.ct * 4;
  const int jq = jb + 2 * cq;
  const int segcap = p.cap >> 2;
  unsigned m0 = hm[0], m1 = hm[1];
  while (m0 | m1) {
#pragma unroll
    for (int x = 0; x < 2; x++) {
      unsigned& m = x ? m1 : m0;
      if (m) {
        const int idx = __ffs(m) - 1;
        m &= m - 1;
        const float v = sv[((x * (BN / 16) + (idx >> 2)) * 1024) + (idx & 3)];
        const unsigned id = (unsigned)(jq + (idx >> 1) * 8 + (idx & 1));
        int key = __float_as_int(fmaf(v, ps.A[x], ps.C[x])) - MAGI;
        key = min(max(key, 0), 65534);
        const bool win = v >= ps.hi[x];
        ps.cnt[x] += win;
        put4(ps, x, seg[x], segcap, (win ? 0xFFFF0000u : ((unsigned)key << 16)) | id);
      }
    }
  }
}

// One pass over nt K tiles (tile_of spreads them over [kt0, kt0 + span)); every consumer warp
// arrives once per tile on the stage's empty barrier. Ends with a consumer barrier.
template <int MODE>
__device__ __forceinline__ void run_pass(const Ctx& X, int& c, int kt0, int span, int nt) {
  const Params& p = *X.p; Smem& S = *X.S;
  const int cq = X.cq;
  const int R[2] = {X.gw * 16 + X.g, X.gw * 16 + X.g + 8};
  const int lim[2] = {S.lim[R[0]], S.lim[R[1]]};
  PassState ps;
#pragma unroll
  for (int x = 0; x < 2; x++) {
    ps.mx[x] = -INFINITY; ps.mn[x] = INFINITY; ps.sq[x] = 0.f; ps.cnt[x] = 0;
    ps.A[x] = 0.f; ps.C[x] = 0.f; ps.Fs[x] = 0.f; ps.Fb[x] = 0.f; ps.lo[x] = 0.f; ps.hi[x] = 0.f;
    ps.wv[x] = 0; ps.wf[x] = 0; ps.bv[x] = 0; ps.bf[x] = 0;
    ps.th[x] = -INFINITY;
#pragma unroll
    for (int i = 0; i < TMR; i++) ps.tl[x][i] = -INFINITY;
    if (MODE == 1) { ps.A[x] = S.bA[R[x]]; ps.C[x] = S.bC[R[x]]; }
    if (MODE == 5) { ps.A[x] = S.mA[R[x]]; ps.C[x] = S.mC[R[x]]; ps.lo[x] = S.lor[R[x]]; ps.hi[x] = S.hir[R[x]]; }
    if (MODE == 3) { ps.A[x] = S.fA[R[x]]; ps.C[x] = S.fC[R[x]]; ps.Fs[x] = S.fsel[R[x]]; ps.Fb[x] = S.fbnd[R[x]]; }
  }
  int wlim = 0x7fffffff;
#pragma unroll
  for (int q = 0; q < 16; q++) wlim = min(wlim, S.lim[X.cw * 64 + X.w * 16 + q]);
  const uint64_t qd = desc_sw128(X.qs + X.cw * 64 * 64);
  unsigned* const seg[2] = {X.bufp + (size_t)R[0] * p.cap + cq * (p.cap >> 2), X.bufp + (size_t)R[1] * p.cap + cq * (p.cap >> 2)};
  const int segcap5 = p.cap >> 2;
  ps.q4[0] = make_uint4(0u, 0u, 0u, 0u); ps.q4[1] = ps.q4[0];
  if (MODE == 5 && (OVL || PP)) {
    // the main pass with FA3's tricks: PP orders the two consumer warpgroups' wgmma issue with
    // named barriers 3/4 (WG0 issues tile i, then WG1 issues tile i, then WG0 tile i+1, ...), so
    // one warpgroup's selection runs under the other's MMA; OVL issues this warpgroup's next tile
    // into a second accumulator before it selects on the current one (wgmma.wait_group 1).
    if (PP && X.cw == 1) bar_arrive(3, 256);
    auto issue = [&](float (&a)[BN / 2], int cc) {
      const int s = cc % NST;
      mbar_wait(&X.full[s], (cc / NST) & 1);
      if (PP) { if (X.cw == 0) bar_sync(3, 256); else bar_sync(4, 256); }
      const uint64_t kd = desc_sw128(X.ring + s * BN * 64);
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; kk++) WGMMA(a, qd + 2 * kk, kd + 2 * kk, kk);
      wg_commit();
      if (PP) { if (X.cw == 0) bar_arrive(4, 256); else bar_arrive(3, 256); }
    };
    auto finish = [&](float (&a)[BN / 2], int cc, int it) {
      fence_regs(a);
      if (X.lane == 0) mbar_arrive(&X.empty[cc % NST]);
      const int jb = tile_of(kt0, span, nt, it) * BN;
      if ((jb >= p.sink) && (jb + BN - 1 <= wlim)) process5<false>(a, ps, X, R, lim, jb, seg);
      else process5<true>(a, ps, X, R, lim, jb, seg);
    };
    if (!OVL) {
      for (int it = 0; it < nt; it++) {
        float acc[BN / 2];
        issue(acc, c);
        wg_wait<0>();
        finish(acc, c, it);
        c++;
      }
    } else {
      float acc0[BN / 2], acc1[BN / 2];
      issue(acc0, c);
      for (int it = 0; it < nt; it += 2) {
        if (it + 1 < nt) { issue(acc1, c + 1); wg_wait<1>(); } else wg_wait<0>();
        finish(acc0, c, it);
        if (it + 1 < nt) {
          if (it + 2 < nt) { issue(acc0, c + 2); wg_wait<1>(); } else wg_wait<0>();
          finish(acc1, c + 1, it + 1);
        }
        c += 2;
      }
      c -= (nt & 1);
    }
    if (PP && X.cw == 0) bar_sync(3, 256);
  } else
  for (int it = 0; it < nt; it++) {
    const int tile = tile_of(kt0, span, nt, it);
    const int jb = tile * BN;
    const int s = c % NST;
    mbar_wait(&X.full[s], (c / NST) & 1);
    float acc[BN / 2];
    {
      const uint64_t kd = desc_sw128(X.ring + s * BN * 64);
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; kk++) WGMMA(acc, qd + 2 * kk, kd + 2 * kk, kk);
      wg_commit();
      wg_wait<0>();
      fence_regs(acc);
    }
    if (X.lane == 0) mbar_arrive(&X.empty[s]);
    c++;
    if (MODE == 4 && it == 0) {
#pragma unroll
      for (int x = 0; x < 2; x++) {
        float m1 = -INFINITY, m0 = INFINITY, sm = 0.f, cn = 0.f;
#pragma unroll
        for (int a = 0; a < BN / 8; a++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            int j = jb + a * 8 + 2 * cq + e;
            if (j >= p.sink && j <= lim[x]) { float sc = acc[4 * a + 2 * x + e] * p.scale; m1 = fmaxf(m1, sc); m0 = fminf(m0, sc); sm += sc; cn += 1.f; }
          }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
          m1 = fmaxf(m1, __shfl_xor_sync(FULLM, m1, o));
          m0 = fminf(m0, __shfl_xor_sync(FULLM, m0, o));
          sm += __shfl_xor_sync(FULLM, sm, o);
          cn += __shfl_xor_sync(FULLM, cn, o);
        }
        // The histogram's range: v8d's [min - 0.6 w, max + 0.6 w] of the first sampled tile, with the
        // upper margin UPM w (a skewed upper tail would pile into the clamped top bin); and when the
        // band edges reach only the top fraction (< 1/4) of the sample, from the tile's mean up:
        // values below the range are not counted at all (only ranks from the top are read).
        float lo, hi;
        if (m1 >= m0) {
          const float wd = fmaxf(m1 - m0, 1e-20f + 1e-3f * fabsf(m1));
          const float nrow = fmaxf((float)S.n[R[x]], 1.f);
          const float sve = fmaxf(fminf((float)(p.nsamp * BN), nrow), 1.f);
          const float rr = (float)p.topk * sve / nrow;
          const float fr = (rr + 4.f * sqrtf(rr) + 2.f) / sve;
          const float mean = sm / fmaxf(cn, 1.f);
          if (MEANCUT && fr < 0.25f && mean < m1) { lo = mean; hi = m1 + UPM * (m1 - mean); }
          else { lo = m0 - 0.6f * wd; hi = m1 + UPM * wd; }
        }
        else { lo = 0.f; hi = 0.f; }
        float iv = (hi > lo) ? (float)NB / (hi - lo) : 0.f;
        float A = p.scale * iv;
        if (!(A < 1e30f) || !(A > 0.f)) { ps.A[x] = 0.f; ps.C[x] = MAGF; }
        else { ps.A[x] = A; ps.C[x] = MAGF - 0.5f - lo * iv; }
        if (cq == 0) { S.bA[R[x]] = ps.A[x]; S.bC[R[x]] = ps.C[x]; }
      }
    }
    const bool wfull = (jb >= p.sink) && (jb + BN - 1 <= wlim);
    if (MODE == 5) {
      if (wfull) process5<false>(acc, ps, X, R, lim, jb, seg);
      else process5<true>(acc, ps, X, R, lim, jb, seg);
    } else {
      if (wfull) process<MODE, false>(acc, ps, X, R, lim, jb);
      else process<MODE, true>(acc, ps, X, R, lim, jb);
    }
  }
  if (MODE == 0 || MODE == 4 || MODE == 6) {
#pragma unroll
    for (int x = 0; x < 2; x++) {
      // MODE 6: tau = min over the quad of the lanes' rn-th: >= k distinct scores are >= tau
      float tau = ps.th[x];
#pragma unroll
      for (int o = 1; o < 4; o <<= 1) {
        if (MODE == 6) tau = fminf(tau, __shfl_xor_sync(FULLM, tau, o));
        ps.cnt[x] += __shfl_xor_sync(FULLM, ps.cnt[x], o);
        ps.mx[x] = fmaxf(ps.mx[x], __shfl_xor_sync(FULLM, ps.mx[x], o));
        ps.mn[x] = fminf(ps.mn[x], __shfl_xor_sync(FULLM, ps.mn[x], o));
        ps.sq[x] += __shfl_xor_sync(FULLM, ps.sq[x], o);
      }
      if (cq == 0) { S.mx[R[x]] = ps.mx[x]; S.mn[R[x]] = ps.mn[x]; S.sv[R[x]] = ps.cnt[x]; S.ssq[R[x]] = ps.sq[x]; if (MODE == 6) S.tmv[R[x]] = tau; }
    }
  }
  if (MODE == 5) {
#pragma unroll
    for (int x = 0; x < 2; x++) {
      const int sp = ps.wv[x], k = sp & 3, base = sp - k;   // the last partial group
      if (k > 0 && !(ABL & 1)) {
        if (base + 0 < segcap5) seg[x][base] = ps.q4[x].x;
        if (k > 1 && base + 1 < segcap5) seg[x][base + 1] = ps.q4[x].y;
        if (k > 2 && base + 2 < segcap5) seg[x][base + 2] = ps.q4[x].z;
      }
      S.segc[R[x]][cq] = sp; S.wcnt[R[x]][cq] = ps.cnt[x];
    }
  }
  bar_sync(1, 256);
}

// rank search over a 16-bit packed NB-bin histogram row (4 bins per lane), from the top
__device__ __forceinline__ int find_rank_h(const unsigned* hrow, int chi, int Rk, int lane, int& bsel, int& cab) {
  int hv[4]; int sl = 0;
  const unsigned w0 = hrow[lane * 2], w1 = hrow[lane * 2 + 1];
  hv[0] = w0 & 0xffff; hv[1] = w0 >> 16; hv[2] = w1 & 0xffff; hv[3] = w1 >> 16;
#pragma unroll
  for (int q = 0; q < 4; q++) sl += hv[q];
  int x = sl;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int y = __shfl_down_sync(FULLM, x, o); if (lane + o < 32) x += y; }
  int total = chi + __shfl_sync(FULLM, x, 0);
  bsel = 0; cab = 0;
  if (Rk <= chi) return 1;
  if (total < Rk) return 2;
  int above = chi + x - sl;
  bool mine = above < Rk && Rk <= above + sl;
  unsigned bal = __ballot_sync(FULLM, mine);
  int src = __ffs(bal) - 1;
  int b = 0, cc = above;
  bool found = false;
#pragma unroll
  for (int q = 3; q >= 0; q--) {
    if (!found) { if (cc + hv[q] >= Rk) { b = lane * 4 + q; found = true; } else cc += hv[q]; }
  }
  bsel = __shfl_sync(FULLM, b, src);
  cab = __shfl_sync(FULLM, cc, src);
  return 0;
}
// rank search over a 16-bit packed NB-bin histogram row by a group of 8 lanes (16 bins per lane,
// 4 rows per warp at once); every lane of the warp must call it (no early exits)
__device__ __forceinline__ int find_rank_hg(const unsigned* hrow, int chi, int Rk, int gl, int grp, int& bsel, int& cab) {
  int hv[16]; int sl = 0;
#pragma unroll
  for (int q = 0; q < 8; q++) { const unsigned w = hrow[gl * 8 + q]; hv[2 * q] = w & 0xffff; hv[2 * q + 1] = w >> 16; }
#pragma unroll
  for (int q = 0; q < 16; q++) sl += hv[q];
  int x = sl;
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) { int y = __shfl_down_sync(FULLM, x, o, 8); if (gl + o < 8) x += y; }
  const int total = chi + __shfl_sync(FULLM, x, 0, 8);
  const int above = chi + x - sl;
  const bool mine = above < Rk && Rk <= above + sl;
  const unsigned bal = (__ballot_sync(FULLM, mine) >> (grp * 8)) & 0xffu;
  const int src = bal ? __ffs(bal) - 1 : 0;
  int b = 0, cc = above;
  bool found = false;
#pragma unroll
  for (int q = 15; q >= 0; q--) {
    if (!found) { if (cc + hv[q] >= Rk) { b = gl * 16 + q; found = true; } else cc += hv[q]; }
  }
  bsel = __shfl_sync(FULLM, b, src, 8);
  cab = __shfl_sync(FULLM, cc, src, 8);
  if (Rk <= chi) { bsel = 0; cab = 0; return 1; }
  if (total < Rk) { bsel = 0; cab = 0; return 2; }
  return 0;
}
// rank search over a 32-bit 256-bin histogram (8 bins per lane), from the top
__device__ __forceinline__ int find_rank8(const int* hrow, int Rk, int lane, int& bsel, int& cab) {
  int hv[8]; int sl = 0;
#pragma unroll
  for (int q = 0; q < 8; q++) { hv[q] = hrow[lane * 8 + q]; sl += hv[q]; }
  int x = sl;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int y = __shfl_down_sync(FULLM, x, o); if (lane + o < 32) x += y; }
  int total = __shfl_sync(FULLM, x, 0);
  bsel = 0; cab = 0;
  if (total < Rk) return 2;
  int above = x - sl;
  bool mine = above < Rk && Rk <= above + sl;
  unsigned bal = __ballot_sync(FULLM, mine);
  int src = __ffs(bal) - 1;
  int b = 0, cc = above;
  bool found = false;
#pragma unroll
  for (int q = 7; q >= 0; q--) {
    if (!found) { if (cc + hv[q] >= Rk) { b = lane * 8 + q; found = true; } else cc += hv[q]; }
  }
  bsel = __shfl_sync(FULLM, b, src);
  cab = __shfl_sync(FULLM, cc, src);
  return 0;
}
__device__ __forceinline__ void set_band(Smem& S, int r, float lo, float hi, float scale) {
  float wd = hi - lo;
  float iv = (wd > 0.f) ? (float)NB / wd : 0.f;
  float A = scale * iv;
  if (!(A < 1e30f) || !(A > 0.f)) { S.bA[r] = 0.f; S.bC[r] = MAGF; return; }
  S.bA[r] = A;
  S.bC[r] = MAGF - 0.5f - lo * iv;
}
__device__ __forceinline__ float edge(float A, float C, float scale, float b) {
  return scale * (((MAGF - C) + (b - 0.5f)) / A);
}

__global__ void __launch_bounds__(384, MINB) kern(const __grid_constant__ CUtensorMap km, const __grid_constant__ CUtensorMap qm, Params p) {
  extern __shared__ unsigned char smraw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  bf16* ring = (bf16*)sm;
  bf16* qs = (bf16*)(sm + NST * BN * 128);
  uint64_t* full = (uint64_t*)(sm + NST * BN * 128 + 2 * 8192);
  uint64_t* empty = full + NST;
  uint64_t* qbar = empty + NST;
  unsigned* hist = (unsigned*)(qbar + 2);
  Smem& S = *(Smem*)((unsigned char*)hist + REGION);
  const int tid = threadIdx.x, lane = tid & 31;
  const int nbp = p.B * p.NPAIR, L = blockIdx.x;
  const int qt = p.nqt - 1 - L / nbp, rest = L % nbp, b = rest / p.NPAIR, pair = rest % p.NPAIR;
  const int i0 = qt * p.QT, T = p.T, topk = p.topk, sink = p.sink, window = p.window;
  const float scale = p.scale;
  const int ilast = min(i0 + p.QT - 1, T - 1), maxlim = ilast - window;
  // lsa: consumer c's query head and first row; both read key head kvh
  const int qh0 = p.RS ? pair : 2 * pair, qh1 = p.RS ? pair : 2 * pair + 1, kvh = qh0 / p.G;
  const bool act = ilast - window - sink + 1 > topk;
  if (tid < NROW) {
    const int R = tid, r = R & 63, i = i0 + r + (R >> 6) * p.RS;
    int n = 0;
    if (i < T) { n = i - window - sink + 1; if (n < 0) n = 0; }
    const bool ra = (i < T) && (n > topk);
    S.lim[R] = ra ? i - window : -1; S.n[R] = n;
    S.state[R] = ra ? 1 : 0;
    S.outcnt[R] = 0; S.bndcnt[R] = 0; S.bufcnt[R] = 0; S.pend[R] = 0; S.need[R] = 0;
    S.fA[R] = 0.f; S.fC[R] = 1.f; S.fsel[R] = INFINITY; S.fbnd[R] = INFINITY;
    S.bA[R] = 0.f; S.bC[R] = -INFINITY; S.mA[R] = 0.f; S.mC[R] = -INFINITY; S.lor[R] = INFINITY; S.hir[R] = INFINITY;
    S.blo[R] = 0.f; S.bhi[R] = 0.f; S.rms[R] = 0.f; S.hinf[R] = 0;
  }
  if (tid < MAXIT + 2) S.iflag[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < NST; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], 8); }
    mbar_init(qbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    S.flag2 = 0;
    int slot = -1;
    if (act && p.cap > 0) {
      unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      for (int tries = 0; tries < 64 && slot < 0; tries++)
        for (int s = 0; s < 4; s++) if (atomicCAS(&g_lock[smid * 4 + s], 0, 1) == 0) { slot = smid * 4 + s; break; }
      if (slot < 0) atomicAdd(&g_dbg[3], 1ull);
    }
    S.slot = slot;
  }
  for (int x = tid; x < NROW * HW; x += 384) hist[x] = 0;
  __syncthreads();
  const int kt0 = sink / BN, kt1 = maxlim / BN + 1, span = kt1 - kt0;
  const int ns = min(span, p.nsamp);

  if (tid >= 256) {   // ---------------- producer warpgroup ----------------
    if (PREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(PREG));
    if (!act || tid >= 288) return;
    const int bk = b * p.HK + kvh;
    const int rowbase = bk * T;
    if (lane == 0) {
      mbar_expect_tx(qbar, 2 * 8192);
      tma_load_2d(qs, &qm, 0, (b * p.H + qh0) * T + i0, qbar);
      tma_load_2d(qs + 64 * 64, &qm, 0, (b * p.H + qh1) * T + i0 + p.RS, qbar);
    }
    int c = 0;
    auto produce = [&](int nt) {
      if (lane == 0) {
        for (int it = 0; it < nt; it++, c++) {
          const int s = c % NST;
          if (c >= NST) mbar_wait(&empty[s], ((c / NST) - 1) & 1);
          mbar_expect_tx(&full[s], BN * 128);
          tma_load_2d(ring + s * BN * 64, &km, 0, rowbase + tile_of(kt0, span, nt, it) * BN, &full[s]);
        }
      }
      __syncwarp();
    };
    produce(p.tm ? span : ns);
    if (S.slot >= 0) produce(span);
    bar_sync(2, 288);
    if (S.flag2) {
      produce(span);
      for (int it = 0; it < MAXIT; it++) {
        produce(span);
        bar_sync(2, 288);
        if (!S.iflag[it]) break;
      }
      produce(span);
    }
    return;
  }

  // ---------------- consumers ----------------
  if (CREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(CREG));
  Ctx X;
  X.p = &p; X.S = &S; X.hist = hist; X.ring = ring; X.qs = qs; X.full = full; X.empty = empty;
  X.ct = tid; X.cw = tid >> 7; X.w = (tid >> 5) & 3; X.gw = tid >> 5; X.lane = lane; X.g = lane >> 2; X.cq = lane & 3;
  X.i0 = i0;
  const int h = X.cw ? qh1 : qh0, bh = b * p.H + h;
  // lsa: consumer 1's rows start RS rows on, so row r of either consumer is X.op[i0 + r]
  X.op = p.out + (size_t)bh * T * topk + (size_t)X.cw * p.RS * topk;
  X.bufp = nullptr;
  X.wring = hist + 4096; X.bring = X.wring + NROW * WR;
  const int wt = tid & 127;
  for (int r = 0; r < 64; r++) {
    int i = i0 + r + X.cw * p.RS;
    if (i >= T) break;
    if (S.lim[X.cw * 64 + r] < 0) {
      int n = S.n[X.cw * 64 + r];
      int* o = X.op + (size_t)(i0 + r) * topk;
      for (int x = wt; x < topk; x += 128) o[x] = x < n ? sink + x : -1;
    }
  }
  if (act) {
  mbar_wait(qbar, 0);
  int c = 0;
  long long tp0 = PROF ? clock64() : 0;
  if (p.tm) run_pass<6>(X, c, kt0, span, span);
  else run_pass<4>(X, c, kt0, span, ns);
  long long tp1 = PROF ? clock64() : 0;
  if (p.tm) {
    // small-k path: the band is [tau, row max] with no upper edge; tau (raw score units) is an
    // exact lower bound on the row's k-th score, so every top-k key is a band entry
    if (X.ct < NROW && S.state[X.ct]) {
      const int R = X.ct;
      const float tau = S.tmv[R], smax = S.mx[R], smin = S.mn[R];
      const float lor = tau > -INFINITY ? tau : (smin / scale) - fabsf(smin / scale) * 1e-6f - 1e-30f;
      const float nlo = lor * scale;
      const float qhi = fmaxf(nextafterf(smax, INFINITY) + 1e-6f * fabsf(smax), nextafterf(nlo, INFINITY)) + 1e-30f;
      const float iv = 65536.f / (qhi - nlo), A2 = scale * iv;
      S.blo[R] = nlo; S.bhi[R] = qhi; S.hinf[R] = 0; S.hir[R] = INFINITY; S.lor[R] = lor;
      if (!(A2 < 1e30f) || !(A2 > 0.f)) { S.mA[R] = 0.f; S.mC[R] = -INFINITY; S.blo[R] = 0.f; S.bhi[R] = INFINITY; S.lor[R] = INFINITY; }
      else { S.mA[R] = A2; S.mC[R] = MAGF - 0.5f - nlo * iv; }
      const int sv = S.sv[R];
      S.rms[R] = sv > 0 ? sqrtf(S.ssq[R] / (float)sv) : 0.f;
    }
  } else
  // band setup: each warp its 16 rows, 4 at a time (8 lanes per row)
  {
    const int grp = lane >> 3, gl = lane & 7;
    for (int rq = 0; rq < 4; rq++) {
      const int R = X.gw * 16 + rq * 4 + grp, r = R & 63;
      const bool on = S.state[R] != 0;
      const int sv = S.sv[R];
      const float A = S.bA[R], C = S.bC[R], smin = S.mn[R], smax = S.mx[R];
      float rr_ = 0.f, sd = 0.f;
      int Rhi = 1, Rlo = 1;
      if (sv > 0 && S.n[R] > 0) {
        rr_ = (float)topk * (float)sv / (float)S.n[R];
        sd = sqrtf(rr_);
        Rhi = max(1, (int)floorf(rr_ - 4.f * sd - 1.f));
        Rlo = (int)ceilf(rr_ + 4.f * sd + 2.f);
      }
      int b1, cc1, b2, cc2;
      const int st1 = find_rank_hg(hist + R * HW, 0, Rhi, gl, grp, b1, cc1);
      const int st2 = find_rank_hg(hist + R * HW, 0, Rlo, gl, grp, b2, cc2);
      if (on && gl == 0) {
        float nlo, nhi, qhi;
        bool hinf = false;
        if (sv == 0) { nlo = -1e30f; nhi = 1e30f; }
        else {
          if (A > 0.f) {
            nhi = (st1 == 0 && b1 < NB - 1) ? edge(A, C, scale, (float)(b1 + 1)) : nextafterf(smax, INFINITY);
            if (Rlo > sv) nlo = smin - 0.5f * (smax - smin) - 1e-6f * fabsf(smin) - 1e-30f;
            else nlo = (st2 == 0 && b2 > 0) ? edge(A, C, scale, (float)b2) : smin;
          } else {
            nhi = nextafterf(smax, INFINITY);
            nlo = smin - 0.5f * fabsf(smin) - 1e-30f;
          }
          if (!(nhi > nlo)) nhi = nextafterf(nlo, INFINITY);
          // HIINF: the sample cannot place an upper edge with room (expected rank of the k-th < ~1):
          // everything >= lo goes to the band, keyed over [lo, smax + QHM (smax - lo)), the top key clamped.
          hinf = HINF && (rr_ - 4.f * sd - 1.f < 1.f);
        }
        qhi = nhi;
        if (hinf) qhi = fmaxf(nhi, smax + QHM * fmaxf(smax - nlo, 1e-30f));
        S.blo[R] = nlo; S.bhi[R] = qhi; S.hinf[R] = hinf ? 1 : 0;
        float iv = 65536.f / (qhi - nlo);
        float A2 = scale * iv;
        if (!(A2 < 1e30f) || !(A2 > 0.f)) { S.mA[R] = 0.f; S.mC[R] = -INFINITY; S.blo[R] = 0.f; S.bhi[R] = INFINITY; S.lor[R] = INFINITY; S.hir[R] = INFINITY; }
        else { S.mA[R] = A2; S.mC[R] = MAGF - 0.5f - nlo * iv; S.lor[R] = nlo / scale; S.hir[R] = hinf ? INFINITY : nhi / scale; }
        S.rms[R] = sv > 0 ? sqrtf(S.ssq[R] / (float)sv) : 0.f;
      }
    }
  }
  bar_sync(1, 256);
  long long tp2 = PROF ? clock64() : 0, tp3 = tp2;
  const int slot = S.slot;
  if (slot >= 0) {
    X.bufp = p.buf + (size_t)slot * NROW * p.cap;
    run_pass<5>(X, c, kt0, span, span);
    if (PROF) tp3 = clock64();
    int* wh = reinterpret_cast<int*>(hist) + X.gw * 256;
    unsigned* stg = X.gw < 4 ? reinterpret_cast<unsigned*>(const_cast<bf16*>(X.ring)) + X.gw * STG : hist + 2048 + (X.gw - 4) * STG;
    const int segcap = p.cap >> 2;
    for (int rr = 0; rr < 16; rr++) {
      const int R = X.gw * 16 + rr, r = R & 63;
      if (!S.state[R]) continue;
      // lane l < 4 holds segment l's count; P = exclusive prefix of the (clamped) counts
      int cl = lane < 4 ? S.segc[R][lane] : 0;
      int wl = lane < 4 ? S.wcnt[R][lane] : 0;
      const bool ovf = __any_sync(FULLM, cl > segcap);
      cl = min(cl, segcap);
      int c0 = __shfl_sync(FULLM, cl, 0), c1 = __shfl_sync(FULLM, cl, 1), c2 = __shfl_sync(FULLM, cl, 2), c3 = __shfl_sync(FULLM, cl, 3);
      int ca = __shfl_sync(FULLM, wl, 0) + __shfl_sync(FULLM, wl, 1) + __shfl_sync(FULLM, wl, 2) + __shfl_sync(FULLM, wl, 3);
      const int P1 = c0, P2 = c0 + c1, P3 = c0 + c1 + c2, ntot = P3 + c3;
      const int nb = ntot - ca;
      float W = S.bhi[R] - S.blo[R];
      float tol = 0.015625f * 0.75f * S.rms[R];
      bool fail = ovf || nb > p.capb || ca > topk || ntot < topk || !(W < 1e30f) || W * (1.f / 65536.f) > 0.6f * tol;
      if (ABL & 2) continue;
      if (fail) {
        if (PROF >= 2 && lane == 0) { if (ovf) atomicAdd(&g_prof[5], 1ull); if (nb > p.capb) atomicAdd(&g_prof[6], 1ull); atomicMax(&g_prof[7], (unsigned long long)ntot); }
        if (!(ABL & 4) && lane == 0) { S.pend[R] = 1; S.flag2 = 1; atomicAdd(&g_dbg[1], 1ull);
          atomicAdd(&g_dbg[ca > topk ? 5 : (ntot < topk ? 6 : (ovf || nb > p.capb ? 7 : 8))], 1ull); }
        continue;
      }
      long long tq0 = PROF >= 2 ? clock64() : 0;
      const int need = topk - ca;
      const unsigned* brow = X.bufp + (size_t)R * p.cap;
      auto ent = [&](int e) -> unsigned {
        if (e < STG) return stg[e];
        const int l = (e >= P1) + (e >= P2) + (e >= P3);
        const int P = l == 0 ? 0 : (l == 1 ? P1 : (l == 2 ? P2 : P3));
        return __ldcg(brow + l * segcap + (e - P));
      };
      for (int q = lane; q < 256; q += 32) wh[q] = 0;
      __syncwarp();
      // stage the 4 segments contiguously in shared memory (up to STG entries; every segment's
      // 16-byte groups loaded in the same round) and histogram the band entries' top key byte
      {
        const int cs[4] = {c0, c1, c2, c3}, Ps[4] = {0, P1, P2, P3};
        const int m4 = (max(max(c0, c1), max(c2, c3)) + 3) >> 2;
        for (int q0 = 0; q0 < m4; q0 += 32 * SU) {
          uint4 v[4][SU];
#pragma unroll
          for (int l = 0; l < 4; l++)
#pragma unroll
            for (int u = 0; u < SU; u++) {
              const int q = q0 + u * 32 + lane;
              if (4 * q < cs[l]) v[l][u] = __ldcg(reinterpret_cast<const uint4*>(brow + l * segcap) + q);
            }
#pragma unroll
          for (int l = 0; l < 4; l++)
#pragma unroll
            for (int u = 0; u < SU; u++) {
              const int q = q0 + u * 32 + lane;
              if (4 * q < cs[l]) {
                const unsigned vv[4] = {v[l][u].x, v[l][u].y, v[l][u].z, v[l][u].w};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                  const int e = 4 * q + j;
                  if (e < cs[l]) {
                    if (Ps[l] + e < STG) stg[Ps[l] + e] = vv[j];
                    if ((vv[j] >> 16) != 0xFFFFu) atomicAdd(&wh[vv[j] >> 24], 1);
                  }
                }
              }
            }
        }
      }
      __syncwarp();
      long long tq1 = PROF >= 2 ? clock64() : 0;
      // entries 4e4 .. 4e4+3 of the staged row (shared memory up to STG, else the segments)
      auto ent4 = [&](int e4, unsigned (&en)[4]) {
        const int e = 4 * e4;
        if (e + 4 <= STG && e + 4 <= ntot) { const uint4 t4 = *reinterpret_cast<const uint4*>(stg + e); en[0] = t4.x; en[1] = t4.y; en[2] = t4.z; en[3] = t4.w; }
        else {
#pragma unroll
          for (int j = 0; j < 4; j++) en[j] = (e + j < ntot) ? ent(e + j) : 0u;
        }
      };
      const int n4 = (ntot + 3) >> 2;
      unsigned thr_hi = 0x10000u, thr_lo = 0x10000u; int bneed = 0;
      if (need > 0) {
        int b1, c1;
        find_rank8(wh, need, lane, b1, c1);
        float w1 = W * (1.f / 256.f);
        float e0 = S.blo[R] + b1 * w1, e1 = e0 + w1;
        float ak = (e0 <= 0.f && e1 >= 0.f) ? 0.f : fminf(fabsf(e0), fabsf(e1));
        float tol1 = 0.015625f * fmaxf(ak, 0.75f * S.rms[R]);
        const bool hinf = S.hinf[R] != 0;
        if (w1 <= 0.6f * tol1 && !(hinf && b1 == 255)) {
          thr_hi = (unsigned)(b1 + 1) << 8; thr_lo = (unsigned)b1 << 8; bneed = need - c1;
        } else {
          __syncwarp();
          for (int q = lane; q < 256; q += 32) wh[q] = 0;
          __syncwarp();
          for (int e4 = lane; e4 < n4; e4 += 32) {
            unsigned en[4]; ent4(e4, en);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const unsigned key = en[j] >> 16;
              if (4 * e4 + j < ntot && key != 0xFFFFu && (int)(key >> 8) == b1) atomicAdd(&wh[key & 255], 1);
            }
          }
          __syncwarp();
          int b2, c2;
          find_rank8(wh, need - c1, lane, b2, c2);
          thr_lo = ((unsigned)b1 << 8) | (unsigned)b2; thr_hi = thr_lo + 1; bneed = need - c1 - c2;
          if (lane == 0) atomicAdd(&g_dbg[2], 1ull);
          if (hinf && thr_lo == 65534u) {   // the k-th is in the clamped top key: unknown width
            __syncwarp();
            if (lane == 0) { S.pend[R] = 1; S.flag2 = 1; atomicAdd(&g_dbg[1], 1ull); atomicAdd(&g_dbg[9], 1ull); }
            continue;
          }
        }
      }
      __syncwarp();
      long long tq2 = PROF >= 2 ? clock64() : 0;
      // output: 4 entries per lane per step; bit-sliced ballots give the in-order prefixes
      int* orow = X.op + (size_t)(i0 + r) * topk;
      int wpos = 0, seen = 0;
      const unsigned lt = (1u << lane) - 1u;
      for (int s0 = 0; s0 < n4; s0 += 32) {
        const int e4 = s0 + lane;
        unsigned en[4] = {0u, 0u, 0u, 0u};
        if (e4 < n4) ent4(e4, en);
        bool sel[4], bnd[4];
        int nbd = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const bool v = 4 * e4 + j < ntot;
          const unsigned key = en[j] >> 16;
          sel[j] = v && (key == 0xFFFFu || key >= thr_hi);
          bnd[j] = v && !sel[j] && key >= thr_lo;
          nbd += bnd[j];
        }
        const unsigned B0 = __ballot_sync(FULLM, nbd & 1), B1 = __ballot_sync(FULLM, nbd & 2), B2 = __ballot_sync(FULLM, nbd & 4);
        int bpre = seen + __popc(B0 & lt) + 2 * __popc(B1 & lt) + 4 * __popc(B2 & lt);
        int nsel = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if (bnd[j]) { sel[j] = bpre < bneed; bpre++; }
          nsel += sel[j];
        }
        const unsigned S0 = __ballot_sync(FULLM, nsel & 1), S1 = __ballot_sync(FULLM, nsel & 2), S2 = __ballot_sync(FULLM, nsel & 4);
        int pp = wpos + __popc(S0 & lt) + 2 * __popc(S1 & lt) + 4 * __popc(S2 & lt);
#pragma unroll
        for (int j = 0; j < 4; j++) if (sel[j]) { if (pp < topk) orow[pp] = (int)(en[j] & 0xffffu); pp++; }
        wpos += __popc(S0) + 2 * __popc(S1) + 4 * __popc(S2);
        seen += __popc(B0) + 2 * __popc(B1) + 4 * __popc(B2);
      }
      __syncwarp();
      if (PROF >= 2 && lane == 0) {
        long long tq3 = clock64();
        atomicAdd(&g_prof[0], (unsigned long long)(tq1 - tq0)); atomicAdd(&g_prof[1], (unsigned long long)(tq2 - tq1));
        atomicAdd(&g_prof[2], (unsigned long long)(tq3 - tq2)); atomicAdd(&g_prof[3], 1ull); atomicAdd(&g_prof[4], (unsigned long long)ntot);
      }
    }
  } else {
    if (X.ct < NROW && S.state[X.ct]) { S.pend[X.ct] = 1; S.flag2 = 1; atomicAdd(&g_dbg[1], 1ull); }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the K ring was written by threads; TMA reuses it
  bar_sync(1, 256);
  if (PROF && X.ct == 0) {
    long long tp4 = clock64();
    atomicAdd(&g_dbg[10], (unsigned long long)(tp1 - tp0)); atomicAdd(&g_dbg[11], (unsigned long long)(tp2 - tp1));
    atomicAdd(&g_dbg[12], (unsigned long long)(tp3 - tp2)); atomicAdd(&g_dbg[13], (unsigned long long)(tp4 - tp3));
  }
  long long tp5 = PROF ? clock64() : 0;
  if (X.ct == 0 && slot >= 0) { __threadfence(); atomicExch(&g_lock[slot], 0); }
  bar_arrive(2, 288);
  if (S.flag2) {
  if (X.ct == 0) atomicAdd(&g_dbg[4], 1ull);
  run_pass<0>(X, c, kt0, span, span);
  if (X.ct < NROW) {
    const int R = X.ct;
    if (S.pend[R]) {
      S.state[R] = 1; S.outcnt[R] = 0; S.bndcnt[R] = 0;
      set_band(S, R, S.mn[R], nextafterf(S.mx[R], INFINITY), scale);
    } else { S.state[R] = 0; S.bA[R] = 0.f; S.bC[R] = -INFINITY; }
    S.pend[R] = 0;
  }
  for (int it = 0; it < MAXIT; it++) {
    bar_sync(1, 256);
    for (int x = X.ct; x < NROW * HW; x += 256) hist[x] = 0;
    bar_sync(1, 256);
    run_pass<1>(X, c, kt0, span, span);
    for (int rr = 0; rr < 16; rr++) {
      const int R = X.gw * 16 + rr;
      if (!S.state[R]) continue;
      float A = S.bA[R], C = S.bC[R];
      int chi = hget(hist + R * HW, NB);
      int bb = 0, cc = 0;
      int st = find_rank_h(hist + R * HW, chi, topk, lane, bb, cc);
      int done = 0;
      float nlo = 0.f, nhi = 0.f;
      if (st == 0 && A > 0.f) {
        float wd = scale / A;
        float e0 = edge(A, C, scale, (float)bb), e1 = e0 + wd;
        float ak = (e0 <= 0.f && e1 >= 0.f) ? 0.f : fminf(fabsf(e0), fabsf(e1));
        float tl = 0.015625f * fmaxf(ak, 0.75f * S.rms[R]);
        if (wd <= 0.6f * tl || it == MAXIT - 1) done = 1;
        else { nlo = e0 - 0.02f * wd; nhi = e1 + 0.02f * wd; }
      } else if (st == 1 && A > 0.f && it < MAXIT - 1) {
        // More than top_k scores above the band (Opus v8d's bug: it selected every candidate
        // here): move the band above its old top and refine again (idx_pp_start.py's fix).
        float top = edge(A, C, scale, (float)NB), bot = edge(A, C, scale, 0.f);
        nlo = top - 0.02f * (top - bot);
        nhi = top + 4.f * fmaxf(top - bot, fabsf(top)) + 1e-30f;
      } else done = 2;
      __syncwarp();
      if (lane == 0) {
        if (done == 1) {
          S.fA[R] = A; S.fC[R] = C; S.fsel[R] = MAGF + (float)(bb + 1); S.fbnd[R] = MAGF + (float)bb;
          S.need[R] = topk - cc; S.state[R] = 0; S.bA[R] = 0.f; S.bC[R] = -INFINITY;
        } else if (done == 2) {
          S.fA[R] = 0.f; S.fC[R] = 1.f; S.fsel[R] = 0.f; S.fbnd[R] = 0.f; S.need[R] = 0;
          S.state[R] = 0; S.bA[R] = 0.f; S.bC[R] = -INFINITY;
        } else {
          S.iflag[it] = 1;
          set_band(S, R, nlo, nhi, scale);
        }
      }
    }
    bar_sync(1, 256);
    bar_arrive(2, 288);
    if (!S.iflag[it]) break;
  }
  run_pass<3>(X, c, kt0, span, span);
  if (PROF && X.ct == 0) atomicAdd(&g_dbg[14], (unsigned long long)(clock64() - tp5));
  }   // flag2
  }   // act
  if (p.uni) {
    // Group union, fused: the last of the 4 head-pair CTAs of this (b, KV group, q tile) to finish
    // merges the 8 heads' rows (their per-head top_k lists, in global memory) into one
    // deduplicated list per row; a T-bit shared bitmap marks the ids seen.
    __threadfence();
    bar_sync(1, 256);
    const int grpk = pair >> 2, gidx = (b * 2 + grpk) * p.nqt + qt;
    if (tid == 0) {
      const int old = atomicAdd(&p.gcnt[gidx], 1);
      S.ulast = old == 3;
      if (old == 3) p.gcnt[gidx] = 0;
    }
    bar_sync(1, 256);
    const int K = 8 * topk, nw = (T + 31) >> 5;
    if (S.ulast && nw * 8 <= REGION / 4) {
      // one row per warp at a time, a bitmap per warp; positions from ballot prefixes
      __threadfence();
      unsigned* wb = hist + X.gw * nw;
      for (int w = lane; w < nw; w += 32) wb[w] = 0u;
      __syncwarp();
      const unsigned lt = (1u << lane) - 1u;
      for (int r = X.gw; r < 64; r += 8) {
        const int i = i0 + r;
        if (i >= T) break;
        int* urow = p.uout + ((size_t)(b * 2 + grpk) * T + i) * K;
        const int* src = p.out + ((size_t)(b * 16 + grpk * 8) * T + i) * topk;
        int cnt = 0;
        for (int e0 = 0; e0 < K; e0 += 32) {
          const int e = e0 + lane;
          const int id = e < K ? __ldcg(src + (size_t)(e / topk) * T * topk + (e % topk)) : -1;
          bool nw_ = false;
          if (id >= 0) { const unsigned bit = 1u << (id & 31); nw_ = !(atomicOr(&wb[id >> 5], bit) & bit); }
          const unsigned bal = __ballot_sync(FULLM, nw_);
          if (nw_) urow[cnt + __popc(bal & lt)] = id;
          cnt += __popc(bal);
        }
        for (int e = cnt + lane; e < K; e += 32) urow[e] = -1;
        __syncwarp();
        for (int e = lane; e < K; e += 32) {
          const int id = __ldcg(src + (size_t)(e / topk) * T * topk + (e % topk));
          if (id >= 0) wb[id >> 5] = 0u;
        }
        __syncwarp();
      }
    } else if (S.ulast) {
      __threadfence();
      for (int w = tid; w < nw; w += 256) hist[w] = 0u;
      for (int r = 0; r < 64; r++) {
        const int i = i0 + r;
        if (i >= T) break;
        if (tid == 0) S.ucnt = 0;
        bar_sync(1, 256);
        int* urow = p.uout + ((size_t)(b * 2 + grpk) * T + i) * K;
        for (int e = tid; e < K; e += 256) {
          const int hh = grpk * 8 + e / topk;
          const int id = __ldcg(p.out + ((size_t)(b * 16 + hh) * T + i) * topk + (e % topk));
          if (id >= 0) {
            const unsigned bit = 1u << (id & 31);
            if (!(atomicOr(&hist[id >> 5], bit) & bit)) urow[atomicAdd(&S.ucnt, 1)] = id;
          }
        }
        bar_sync(1, 256);
        for (int e = S.ucnt + tid; e < K; e += 256) urow[e] = -1;
        for (int e = tid; e < K; e += 256) {
          const int hh = grpk * 8 + e / topk;
          const int id = __ldcg(p.out + ((size_t)(b * 16 + hh) * T + i) * topk + (e % topk));
          if (id >= 0) hist[id >> 5] = 0u;
        }
        bar_sync(1, 256);
      }
    }
  }
}

void run(torch::Tensor q, torch::Tensor k, torch::Tensor out, torch::Tensor buf, int64_t topk, int64_t window,
         int64_t sink, double scale, int64_t cap, int64_t capb, int64_t nsamp, int64_t tm, torch::Tensor uout, torch::Tensor gcnt, int64_t uni) {
  Params p;
  p.out = out.data_ptr<int>(); p.buf = (unsigned*)buf.data_ptr();
  p.B = q.size(0); p.T = q.size(2);
  p.topk = topk; p.window = window; p.sink = sink; p.scale = (float)scale; p.cap = cap; p.capb = capb; p.nsamp = nsamp; p.tm = (int)tm; p.uni = (int)uni; p.uout = uout.data_ptr<int>(); p.gcnt = gcnt.data_ptr<int>();
  p.H = q.size(1); p.HK = k.size(1); p.G = p.H / p.HK;
  p.RS = (p.G % 2 == 0) ? 0 : 64; p.NPAIR = p.RS ? p.H : p.H / 2; p.QT = p.RS ? 128 : 64;
  p.nqt = (p.T + p.QT - 1) / p.QT;
  CUtensorMap km = make_kmap(k.data_ptr(), (long)p.B * p.HK * p.T, BN);
  CUtensorMap qm = make_kmap(q.data_ptr(), (long)p.B * p.H * p.T, 64);
  size_t smem = 1024 + NST * BN * 128 + 2 * 8192 + (2 * NST + 2) * 8 + REGION + sizeof(Smem);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<p.nqt * p.B * p.NPAIR, 384, smem, at::cuda::getCurrentCUDAStream()>>>(km, qm, p);
}
// Group union: in = (B, 16, T, kh) per-head top-kh ids, out = (B, 2, T, 8 kh): per (b, kv head, row)
// the deduplicated union of its 8 heads' ids, -1 padded. A CTA takes RPC rows in turn; a T-bit
// shared bitmap marks the ids seen (cleared again id by id), positions come from a shared counter.
#define RPC 16
__global__ void __launch_bounds__(256) merge_group(const int* __restrict__ in, int* __restrict__ out, int B, int T, int kh) {
  extern __shared__ unsigned bm[];
  __shared__ int cnt;
  const int K = 8 * kh, nw = (T + 31) >> 5, nrows = B * 2 * T;
  for (int w = threadIdx.x; w < nw; w += blockDim.x) bm[w] = 0u;
  __syncthreads();
  for (int rr = 0; rr < RPC; rr++) {
    const int gi = blockIdx.x * RPC + rr;
    if (gi >= nrows) break;
    const int b = gi / (2 * T), g = (gi / T) % 2, i = gi % T;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    int* orow = out + (size_t)gi * K;
    for (int e = threadIdx.x; e < K; e += blockDim.x) {
      const int h = g * 8 + e / kh;
      const int id = in[((size_t)(b * 16 + h) * T + i) * kh + (e % kh)];
      if (id >= 0) {
        const unsigned bit = 1u << (id & 31);
        if (!(atomicOr(&bm[id >> 5], bit) & bit)) orow[atomicAdd(&cnt, 1)] = id;
      }
    }
    __syncthreads();
    for (int e = cnt + threadIdx.x; e < K; e += blockDim.x) orow[e] = -1;
    for (int e = threadIdx.x; e < K; e += blockDim.x) {
      const int h = g * 8 + e / kh;
      const int id = in[((size_t)(b * 16 + h) * T + i) * kh + (e % kh)];
      if (id >= 0) bm[id >> 5] = 0u;
    }
    __syncthreads();
  }
}
void merge(torch::Tensor in, torch::Tensor out) {
  const int B = in.size(0), T = in.size(2), kh = in.size(3);
  const int nrows = B * 2 * T;
  merge_group<<<(nrows + RPC - 1) / RPC, 256, ((T + 31) / 32) * 4, at::cuda::getCurrentCUDAStream()>>>(
      in.data_ptr<int>(), out.data_ptr<int>(), B, T, kh);
}
std::vector<int64_t> dbg() {
  unsigned long long h[16];
  cudaMemcpyFromSymbol(h, g_dbg, sizeof(h));
  unsigned long long z[16] = {0};
  cudaMemcpyToSymbol(g_dbg, z, sizeof(z));
  return std::vector<int64_t>(h, h + 16);
}
std::vector<int64_t> prof() {
  unsigned long long h[8];
  cudaMemcpyFromSymbol(h, g_prof, sizeof(h));
  unsigned long long z[8] = {0};
  cudaMemcpyToSymbol(g_prof, z, sizeof(z));
  return std::vector<int64_t>(h, h + 8);
}
std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, kern);
  size_t smem = 1024 + NST * BN * 128 + 2 * 8192 + (2 * NST + 2) * 8 + REGION + sizeof(Smem);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 384, smem);
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)smem, nb};
}
"""
CPP = ("void run(torch::Tensor q, torch::Tensor k, torch::Tensor out, torch::Tensor buf, int64_t topk, int64_t window, "
       "int64_t sink, double scale, int64_t cap, int64_t capb, int64_t nsamp, int64_t tm, torch::Tensor uout, torch::Tensor gcnt, int64_t uni);\nstd::vector<int64_t> dbg();\nvoid merge(torch::Tensor in, torch::Tensor out);\nstd::vector<int64_t> prof();\nstd::vector<int64_t> attrs();")


def build(bn=64, nst=4, minb=2, su=2, abl=0, hinf=1, preg=24, creg=104, prof=0, qhm=8, ovl=0, pp=1, upm=0.6, meancut=0, tmr=4, tag=""):
    defs = (f"#define BN {bn}\n#define NST {nst}\n#define WGMMA wgmma_ss_m64n{bn}\n#define MINB {minb}\n#define SU {su}\n"
            f"#define ABL {abl}\n#define HINF {hinf}\n#define PREG {preg}\n#define CREG {creg}\n#define PROF {prof}\n#define QHM {float(qhm)}f\n#define OVL {ovl}\n#define PP {pp}\n#define UPM {float(upm)}f\n#define MEANCUT {meancut}\n#define TMR {tmr}\n")
    return load_inline(name=f"lsa_fa3idx12_bn{bn}_s{nst}_m{minb}_u{su}_a{abl}_h{hinf}_r{preg}_{creg}_p{prof}_q{qhm}_o{ovl}{pp}_u{int(round(upm * 10))}_{meancut}_t{tmr}{tag}", cpp_sources=CPP,
                       cuda_sources=HDR + wgmma_ss_fn(bn) + defs + BODY, functions=["run", "dbg", "merge", "prof", "attrs"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"])


class Indexer:
    def __init__(self, bn=64, nst=4, minb=2, su=2, abl=0, hinf=1, preg=24, creg=104, prof=0, qhm=8, ovl=0, pp=1, upm=0.6, meancut=0, nsamp_keys=None, tmr=4, tmk=0, tag="") -> None:
        # tmk: the largest top_k that takes the two-pass small-k path (at most 4 tmr)
        self.ext = build(bn, nst, minb, su, abl, hinf, preg, creg, prof, qhm, ovl, pp, upm, meancut, tmr, tag)
        self.tmk = min(tmk, 4 * tmr)
        self.bn = bn
        self.nsamp_keys = nsamp_keys
        self.bufs = {}
        self.gcnts = {}
        self.nsm = {}

    def debug_stats(self):
        v = self.ext.dbg()
        return {"fallback_rows": v[1], "level2_rows": v[2], "slot_fail": v[3], "fallback_ctas": v[4],
                "fail_over_topk": v[5], "fail_short": v[6], "fail_overflow": v[7], "fail_width": v[8], "fail_hinf_top": v[9],
                "cyc_sample": v[10], "cyc_band": v[11], "cyc_main": v[12], "cyc_final": v[13], "cyc_fallback": v[14]}

    def __call__(self, q, k, top_k, window=0, sink=0, scale=None, buffer=None, uout=None):
        b, hq, t, d = q.shape
        hk = k.shape[1]
        assert hq % hk == 0 and d == 64 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, \
            "fa3idx: bf16, d 64, query heads a multiple of key heads only"
        assert uout is None or (hq == 16 and hk == 2), "fa3idx: the group union takes hq 16, hk 2 only"
        # lsa: ids (at most t - 1 - window) and a row's candidates (t - window - sink) in 16 bits
        assert t - window <= 65536, "fa3idx: 16-bit key ids and histograms need t - window <= 65536"
        scale = d ** -0.5 if scale is None else scale
        q = q.contiguous(); k = k.contiguous()
        out = torch.empty((b, hq, t, top_k), dtype=torch.int32, device=q.device)
        dev = q.device
        if dev not in self.nsm:
            self.nsm[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
        nslots = self.nsm[dev] * 4
        if buffer is not None:   # the band buffer's entries per row
            cap = (min(int(buffer), t) + 31) // 32 * 32
        else:
            cap = int(3 * (8 * math.sqrt(top_k * t / 768.0) + 64)) + 256
            cap = max(cap, 2 * top_k + 256)
            cap = min(cap, t)
            cap = (cap + 31) // 32 * 32
        capb = cap
        cap = (cap + top_k + 31) // 32 * 32   # per row: capb band entries + top_k winners, in 4 lane segments
        # sample size (keys): 512 for t <= 4096, 1024 for t >= 16384 or top_k >= 1024, else 768
        # (tuned on the dev cases), doubled while the expected band at the row with the most
        # candidates, (8 sqrt(top_k sv / n) + 3) n / sv entries, would not fit in 0.6 capb
        nmax = max(1, t - window - sink)
        if self.nsamp_keys is not None:
            sv = self.nsamp_keys
        else:
            sv = 1024 if (t >= 16384 or top_k >= 1024) else (512 if t <= 4096 else 768)
        while sv < min(nmax, 8192) and (8 * math.sqrt(top_k * sv / nmax) + 3) * nmax / sv > 0.6 * capb:
            sv *= 2
        nsamp = max(1, sv // self.bn)
        tm = 1 if top_k <= self.tmk else 0
        need = nslots * 128 * max(cap, 1)
        buf = self.bufs.get(dev)
        if buf is None or buf.numel() < need:
            buf = torch.empty(need, dtype=torch.int32, device=dev)
            self.bufs[dev] = buf
        self.ext.run(q, k, out, buf, int(top_k), int(window), int(sink), float(scale), cap, capb, nsamp, tm,
                     uout if uout is not None else out, self._gcnt(dev, b * hk * ((t + 63) // 64)), 1 if uout is not None else 0)
        return out

    def _gcnt(self, dev, n):
        g = self.gcnts.get(dev)
        if g is None or g.numel() < n:
            g = torch.zeros(max(n, 1), dtype=torch.int32, device=dev)   # self-resetting arrival counters
            self.gcnts[dev] = g
        return g

    def group_union(self, q, k, top_k, window=0, sink=0, scale=None, per_head=None, fused=True):
        """(b, 2, t, 8 top_k): per (b, KV group, row) the deduplicated union of the 8 heads' own top_k.
        fused: one launch (the last head-pair CTA of each group and q tile merges); else the per-head
        kernel then the separate merge kernel."""
        b, hq, t, _ = q.shape
        out = torch.empty((b, 2, t, 8 * top_k), dtype=torch.int32, device=q.device)
        if fused and per_head is None:
            self(q, k, top_k, window=window, sink=sink, scale=scale, uout=out)
            return out
        ph = self(q, k, top_k, window=window, sink=sink, scale=scale) if per_head is None else per_head
        self.ext.merge(ph, out)
        return out


_default = None


def indexer(q, k, top_k, window=0, sink=0, scale=None, buffer=None):
    global _default
    if _default is None:
        _default = Indexer()
    return _default(q, k, top_k, window=window, sink=sink, scale=scale, buffer=buffer)


def indexer_group_union(q, k, top_k, window=0, sink=0, scale=None):
    """Group union: each of the 8 query heads of a KV group selects its own top_k (the per-head
    path above), then a merge kernel writes, per (b, kv head, row), the deduplicated union of the 8
    sets: (b, 2, t, 8 top_k) int32, -1 padded. Two launches."""
    global _default
    if _default is None:
        _default = Indexer()
    return _default.group_union(q, k, top_k, window=window, sink=sink, scale=scale)
