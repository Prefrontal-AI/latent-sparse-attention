# Vendored from frozen_v13/final.py (md5 a57ba7dd1cb065b1a3e350b32a26ac34), by a local Claude Opus 5.5 session, 2026-10-14;
# see __init__.py. Edits are marked 'lsa:', and load_inline's name "hyb_final_v3" is renamed.
"""Sparse-attention backward (H200, d128, G = 8), dispatching per call between
  * the hybrid kernel (hyb.py): key-major dK/dV over 64-key tiles with per-query skipping + query-major dQ, and
  * v14 (v14.py, copied in unchanged apart from a gate argument): key-major over consecutive 8-query tiles.
A statistics kernel samples 8-query groups of THIS call's selection and estimates both kernels' work:
  kvT = (query, 64-key tile) touches, dqT = 64-key tiles of each group's key union (hybrid),
  v14H = active (8-query tile, 64-key half) pairs (v14),
then writes a device-side gate; both pipelines are launched and the one not chosen exits at once.
No host synchronization and nothing cached across calls. Small t (< MIN_T) goes straight to v14.
"""
import os
import re
import torch
from torch.utils.cpp_extension import load_inline
from .bwd_hybrid_common import _gen, HEADER  # lsa: package import

HERE = os.path.dirname(os.path.abspath(__file__))
V14_PATH = os.path.join(HERE, "bwd_v14.py")  # lsa: the vendored v14
HYB_PATH = os.path.join(HERE, "bwd_hybrid_kernel.py")  # lsa: hyb.py


def _between(src, a, b):
    i = src.index(a) + len(a)
    return src[i:src.index(b, i)]


def _add_gate(src, names, want):
    for nm in names:
        m = re.search(r'__global__ void(?: __launch_bounds__\([^)]*\))? ' + nm + r'\(', src)
        assert m, nm
        i = m.end()
        src = src[:i] + "const int* __restrict__ gate_, " + src[i:]
        j = src.index(") {", i) + 3
        src = src[:j] + f"\n  if (gate_ != nullptr && __ldg(gate_) != {want}) return;" + src[j:]
    return src


def _gate_calls(src, names):
    for nm in names:
        src = re.sub(r'(\b' + nm + r'(?:<\d>)?<<<[^;]*?>>>\()', r'\1gate, ', src)
    return src


HYB_KERNELS = ("prep_k", "idx_k", "scan_k", "plan_k", "move_k", "kv_main", "kvp_main", "dq2_main", "dq3_main", "cvt_k")
V14_KERNELS = ("delta_k", "build2_k", "list2_k", "items2_k", "bwd_main", "cvt_k", "cvt_dq2_k")


def _hyb_parts():
    src = open(HYB_PATH).read()
    kern = _between(src, 'KERN = r"""', '"""\n\nHOST = r"""')
    host = _between(src, 'HOST = r"""', '"""\n')
    kern = _add_gate(kern, HYB_KERNELS, 1)
    # host: hyb() -> hyb_run() writing into given outputs, gate passed to every kernel
    host = host.replace("""std::vector<torch::Tensor> hyb(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t BQ, int64_t flags) {""", """static torch::Tensor hyb_run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t BQ, int64_t flags, const int* gate, torch::Tensor dq, torch::Tensor dk, torch::Tensor dv) {""")
    for t in ("dq", "dk", "dv"):
        host = host.replace(f"  auto {t} = torch::empty_like({t[1]});\n", "")
    host = host.replace("return {dq, dk, dv, ibuf};", "return ibuf;")
    host = _gate_calls(host, HYB_KERNELS)
    assert "hyb_run" in host and "empty_like" not in host
    return kern, host


def _v14_kern():
    src = open(V14_PATH).read()
    kern = _between(src, 'KERN = r"""', '"""\n\nHOST = r"""')
    kern = _add_gate(kern, V14_KERNELS, 0)
    return "namespace v14ns {\n" + kern + "\n}  // namespace v14ns\n"


STATS = r"""
// One CTA per sampled 8-query group (bh, g): its queries' key bitmaps (selected | window | sinks).
// acc[0] += kv touches, acc[1] += dQ union tiles, acc[2] += v14 active (8-query tile, 64-key half) pairs.
// The last CTA turns the sums into time estimates and writes gate = 1 (hybrid) or 0 (v14).
__global__ void __launch_bounds__(512) stats_k(const int* __restrict__ sel, unsigned long long* __restrict__ acc,
    int* __restrict__ gate, int T, int TOPK, int BH, int NG, int nsamp, int window, int sink,
    float a_kv, float a_dq, float a_v14, float fix_hyb, float fix_v14, float margin, float ratio) {
  extern __shared__ unsigned bm[];
  __shared__ int red[3][16];
  __shared__ bool last;
  const int s = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long total = (long)BH * NG;
  const long it = (long)(((double)s + 0.5) * (double)total / (double)nsamp);
  const int bh = (int)(it % BH), g = (int)(it / BH);
  const int i0 = 8 * g, tl = (i0 + 7) >> 6, NWc = 2 * (tl + 1), NWp = NWc + 1;
  for (int x = tid; x < 8 * NWp; x += 512) {
    const int p = x / NWp, w = x - p * NWp, i = i0 + p;
    unsigned v = 0u;
    if (window > 0) v |= range_bits(max(0, i - window + 1), i, w);
    if (sink > 0) v |= range_bits(0, min(sink, i + 1) - 1, w);
    bm[x] = v;
  }
  __syncthreads();
  const int* sb = sel + ((size_t)bh * T + i0) * TOPK;
  for (int x = tid; x < 8 * TOPK; x += 512) {
    const int j = __ldg(sb + x);
    const int p = x / TOPK;
    if ((unsigned)j <= (unsigned)(i0 + p)) atomicOr(bm + p * NWp + (j >> 5), 1u << (j & 31));
  }
  __syncthreads();
  int kv = 0, hv = 0, u = 0;
  for (int tt = tid; tt <= tl; tt += 512) {
    int any = 0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const unsigned a = bm[p * NWp + 2 * tt], b2 = bm[p * NWp + 2 * tt + 1];
      kv += (a | b2) != 0u;
      any |= (a | b2) != 0u;
    }
    hv += any;
  }
  for (int w = tid; w < NWc; w += 512) {
    unsigned x = 0u;
#pragma unroll
    for (int p = 0; p < 8; ++p) x |= bm[p * NWp + w];
    u += __popc(x);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    kv += __shfl_xor_sync(0xffffffffu, kv, o);
    hv += __shfl_xor_sync(0xffffffffu, hv, o);
    u += __shfl_xor_sync(0xffffffffu, u, o);
  }
  if (lane == 0) { red[0][warp] = kv; red[1][warp] = hv; red[2][warp] = u; }
  __syncthreads();
  if (tid == 0) {
    int K1 = 0, H1 = 0, U1 = 0;
    for (int w = 0; w < 16; ++w) { K1 += red[0][w]; H1 += red[1][w]; U1 += red[2][w]; }
    atomicAdd(acc, (unsigned long long)K1);
    atomicAdd(acc + 1, (unsigned long long)((U1 + 63) / 64));
    atomicAdd(acc + 2, (unsigned long long)H1);
    __threadfence();
    last = atomicAdd(acc + 3, 1ull) == (unsigned long long)(nsamp - 1);
  }
  __syncthreads();
  if (last && tid == 0) {
    __threadfence();
    const double f = (double)total / (double)nsamp;
    const double K = (double)atomicAdd(acc, 0ull), Q = (double)atomicAdd(acc + 1, 0ull), H = (double)atomicAdd(acc + 2, 0ull);
    const double th = (f * (a_kv * K + a_dq * Q) + fix_hyb) / margin;
    const double tv = f * a_v14 * H + fix_v14;
    // random-like selections (many 64-key touches per v14 half) go to v14 regardless of the time model
    if (gate) *gate = (th < tv && K < (double)ratio * H) ? 1 : 0;
  }
}
"""

V14_HOST = r"""
static void v14_run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o, torch::Tensor lse,
                    torch::Tensor dout, int64_t window, int64_t sink, double scale, int64_t target, const int* gate,
                    torch::Tensor dq, torch::Tensor dk, torch::Tensor dv) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int HKV = k.size(2), TOPK = sel.size(3), G = HQ / HKV;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long nq = q.numel(), nkv = k.numel();
  const int NPq = 64 / G, nqt = (T + NPq - 1) / NPq, nw = (T + 31) / 32, nkb = (T + 127) / 128;
  const int BH = B * HKV, n = BH * nkb;
  const long nld = (long)BH * nqt * 64 * 2;
  const long ndqp = (long)BH * nqt * 8192;
  auto fbuf = torch::empty({ndqp + 2 * nkv + nld}, q.options().dtype(torch::kFloat32));
  float* dq32 = fbuf.data_ptr<float>();
  TORCH_CHECK(ndqp == nq && T % NPq == 0, "v14 path: T % 8 == 0");
  float* dk32 = dq32 + ndqp;
  float* dv32 = dk32 + nkv;
  float* LDp = dv32 + nkv;
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
  v14ns::delta_k<<<(int)((nthr + 255) / 256), 256, 0, stream>>>(gate, reinterpret_cast<const bf16*>(o.data_ptr()),
      reinterpret_cast<const bf16*>(dout.data_ptr()), lse.data_ptr<float>(), reinterpret_cast<float2*>(LDp),
      dq32, dk32, dv32, nkv / 8, B, T, HQ, HKV, G, D, nqt, 1.4426950408889634f, (float)std::log2(scale));
  const int CH = 32768;
  const int bsm = std::min(T, CH) * 4;
  static int bset = 0;
  if (bsm > 48 * 1024 && bsm > bset) { cudaFuncSetAttribute(v14ns::build2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, bsm); bset = bsm; }
  v14ns::build2_k<<<dim3(nw, BH), 1024, bsm, stream>>>(gate, sel.data_ptr<int>(), ST, OCC, T, TOPK, nw, nkb, CH);
  int* BST = CNT;
  int* BSK = BST + n * (nb + 1);
  v14ns::list2_k<<<dim3(nkb, BH), 256, 0, stream>>>(gate, OCC, LIST, BST, T, NPq, nqt, nw, nkb, nb, BQ, (int)window, (int)sink, BSK);
  v14ns::items2_k<<<1, 1024, 0, stream>>>(gate, BST, items, nit, BH, nkb, nb, 0, BSK);
  const float sl2 = (float)(scale * 1.4426950408889634);
  constexpr int SM = v14ns::V7Cfg::SMEM;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(v14ns::bwd_main, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); init = true; }
  CUtensorMap tk = make_map(k.data_ptr(), B, T, HKV, D, 1, 128);
  CUtensorMap tv = make_map(v.data_ptr(), B, T, HKV, D, 1, 128);
  CUtensorMap tq = make_map(q.data_ptr(), B, T, HQ, D, G, 8);
  CUtensorMap tdo = make_map(dout.data_ptr(), B, T, HQ, D, G, 8);
  v14ns::bwd_main<<<(int)nseg, 384, SM, stream>>>(gate, tk, tv, tq, tdo, items, nit, LIST, ST, reinterpret_cast<const float2*>(LDp),
      dq32, dk32, dv32, T, HKV, nqt, nw, nkb, (int)window, (int)sink, sl2, (float)(1.0 / scale));
  v14ns::cvt_k<<<(int)((nkv / 8 + 255) / 256), 256, 0, stream>>>(gate, dq32, reinterpret_cast<bf16*>(dq.data_ptr()), 0,
      dk32, reinterpret_cast<bf16*>(dk.data_ptr()), dv32, reinterpret_cast<bf16*>(dv.data_ptr()), nkv / 8);
  v14ns::cvt_dq2_k<<<dim3(nqt, BH), 256, 0, stream>>>(gate, dq32, reinterpret_cast<bf16*>(dq.data_ptr()), T, HKV, nqt);
}

// mode 0: dispatch (stats + both pipelines gated); 1: hybrid only; 2: v14 only; 3: stats only (exact if nsamp = all)
std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, torch::Tensor o,
                               torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale,
                               int64_t mode, int64_t nsamp, int64_t BQ, int64_t BQ14, std::vector<double> cf) {
  const int B = q.size(0), T = q.size(1), HKV = k.size(2), TOPK = sel.size(3);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  auto st = torch::zeros({5}, q.options().dtype(torch::kInt64));
  int* gate = reinterpret_cast<int*>(st.data_ptr<int64_t>() + 4);
  if (mode == 0 || mode == 3) {
    const int BH = B * HKV, NG = T / 8;
    const int ns = (int)std::min<long>(nsamp, (long)BH * NG);
    const int NT = T / 64;
    const int smem = 8 * (2 * NT + 1) * 4;
    static int sset = 0;
    if (smem > 48 * 1024 && smem > sset) { cudaFuncSetAttribute(stats_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); sset = smem; }
    const double TT = (double)T / 32768.0;
    stats_k<<<ns, 512, smem, stream>>>(sel.data_ptr<int>(), reinterpret_cast<unsigned long long*>(st.data_ptr<int64_t>()),
        gate, T, TOPK, BH, NG, ns, (int)window, (int)sink, (float)cf[0], (float)cf[1], (float)cf[2],
        (float)(cf[3] * TT + cf[4] * TT * TT), (float)(cf[5] * TT + cf[6] * TT * TT), (float)cf[7], (float)cf[8]);
  }
  if (mode == 0) {
    hyb_run(q, k, v, sel, o, lse, dout, window, sink, scale, BQ, 0, gate, dq, dk, dv);
    v14_run(q, k, v, sel, o, lse, dout, window, sink, scale, BQ14, gate, dq, dk, dv);
  } else if (mode == 1) {
    hyb_run(q, k, v, sel, o, lse, dout, window, sink, scale, BQ, 0, nullptr, dq, dk, dv);
  } else if (mode == 2) {
    v14_run(q, k, v, sel, o, lse, dout, window, sink, scale, BQ14, nullptr, dq, dk, dv);
  }
  return {dq, dk, dv, st};
}
"""


def _build():
    kern, host = _hyb_parts()
    src = HEADER + _gen("wg_ss_n64", 64, False) + _gen("wg_ss_n64_tt", 64, False, 1, 1)
    src += _gen("wg_rs_n128_t", 128, True, 0, 1) + _gen("wg_rs_n64_t", 64, True, 0, 1)
    src += r"""
template <int NO> struct RsSel;
template <> struct RsSel<64> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n128_t(d, a, db, sd); } };
template <> struct RsSel<32> { static __device__ __forceinline__ void run(float* d, const unsigned* a, unsigned long long db, int sd) { wg_rs_n64_t(d, a, db, sd); } };
#define wg_rs_nD(d, a, db, sd) RsSel<NO>::run(d, a, db, sd)
"""
    defs = "#define KV_NS 4\n#define DQ_NS 2\n#define DQ2_NS 5\n#define DQ_PP 0\n#define DQ3_NS 4\n#define DQ_SPLIT 1\n#define DQ3_PIPE 0\n#define KV_PERSIST " + os.environ.get("HYB_KV_PERSIST", "0") + "\n#define RCONS 232\n#define RPROD 40\n"
    return defs + src + kern + STATS + _v14_kern() + host + V14_HOST


_ext = load_inline(
    name="lsa_bwd_hybrid" + os.environ.get("HYB_SUFFIX", ""),
    cpp_sources="std::vector<torch::Tensor> bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor sel, "
    "torch::Tensor o, torch::Tensor lse, torch::Tensor dout, int64_t window, int64_t sink, double scale, "
    "int64_t mode, int64_t nsamp, int64_t BQ, int64_t BQ14, std::vector<double> cf);",
    cuda_sources=_build(),
    functions=["bwd"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"],
    verbose=bool(int(os.environ.get("SA_VERBOSE", "0"))),
)

MIN_T = int(os.environ.get("HYB_MIN_T", "16384"))
NSAMP = int(os.environ.get("HYB_NSAMP", "128"))
BQ = int(os.environ.get("HYB_BQ", "2048"))
# time model (ms): hybrid = a_kv*touches + a_dq*union tiles + fix_hyb(t); v14 = a_v14*active halves + fix_v14(t);
# fixed terms are c1*(t/32K) + c2*(t/32K)^2. Fitted in calib.py.
COEF = [float(x) for x in os.environ.get("HYB_COEF", "8.99444e-07,6.04479e-06,8.30123e-06,0.371737,0.679119,1.87439,0.920012").split(",")]
RATIO = float(os.environ.get("HYB_RATIO", "7.0"))   # hybrid only if kvT < RATIO * v14H
MARGIN = float(os.environ.get("HYB_MARGIN", "0.95"))   # hybrid only if its estimate < MARGIN x v14 estimate
MODE = int(os.environ.get("HYB_MODE", "0"))
last_stats = None


def _prep(q, k, v, selected, o, lse, do):
    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    lse = lse.contiguous().float()
    selected = selected.to(torch.int32).contiguous()
    return q, k, v, selected, o, lse, do


def attention_backward(q, k, v, selected, o, lse, do, window=0, sink=0, scale=None, mode=None, nsamp=None, flags=None):
    global last_stats
    b, t, hq, d = q.shape
    hkv = k.shape[2]
    scale = d ** -0.5 if scale is None else scale
    q, k, v, selected, o, lse, do = _prep(q, k, v, selected, o, lse, do)
    m = MODE if mode is None else mode
    if not (d == 128 and hq == 8 * hkv and t % 64 == 0):
        raise NotImplementedError("final.py: d128, G8, t % 64 == 0 only")
    if m == 0 and t < MIN_T:
        m = 2
    bq14 = 128 if t <= 16384 else 256
    dq, dk, dv, st = _ext.bwd(q, k, v, selected, o, lse, do, int(window), int(sink), float(scale), m,
                              NSAMP if nsamp is None else nsamp, BQ, bq14, COEF + [MARGIN, RATIO])
    last_stats = st
    return dq, dk, dv


def stats(q, k, v, selected, o, lse, do, window=0, sink=0, nsamp=None):
    """(kv touches, dQ union tiles, v14 active halves) from the stats kernel (exact when nsamp >= b*hkv*t/8)."""
    q, k, v, selected, o, lse, do = _prep(q, k, v, selected, o, lse, do)
    ns = nsamp or (q.shape[0] * k.shape[2] * q.shape[1] // 8)
    st = _ext.bwd(q, k, v, selected, o, lse, do, int(window), int(sink), 1.0, 3, ns, BQ, 256, COEF + [MARGIN, RATIO])[3]
    return [int(x) for x in st[:3].tolist()]


def debug_choice():
    """1 = hybrid, 0 = v14, for the last dispatched call (reads the device gate; synchronizes)."""
    return None if last_stats is None else int(last_stats[4].item() & 0xffffffff)
