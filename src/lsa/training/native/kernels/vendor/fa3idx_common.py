# Vendored from common.py (md5 d1e5c77a0509f2b6d2ac92beca54b579),
# verbatim; see __init__.py.
"""Shared CUDA snippets for the FA3-style indexer: mbarriers, TMA, wgmma, named barriers, and the
host-side tensor map. Generated as text so the wgmma operand lists can be written for any N."""

def wgmma_fn(n) -> str:
    """`wgmma_m64n{n}(float (&d)[n/2], const unsigned (&a)[4], uint64_t desc, int accum)`: A from
    registers, B (K-major, 128B swizzle) from shared memory via its descriptor."""
    nr = n // 2
    dl = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return f"""
__device__ __forceinline__ void wgmma_m64n{n}(float (&d)[{nr}], const unsigned (&a)[4], uint64_t desc, int accum) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nr + 5}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{%{nr},%{nr + 1},%{nr + 2},%{nr + 3}}}, %{nr + 4}, p, 1, 1, 0;\\n}}\\n"
               : {outs}
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(accum));
}}
template <int R> __device__ __forceinline__ void fence_regs(float (&d)[R]) {{
#pragma unroll
  for (int i = 0; i < R; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}}
"""

HDR = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
typedef __nv_bfloat16 bf16;
#define FULLM 0xffffffffu

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
  unsigned a = smem_u32(bar), done;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(a), "r"(phase) : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* tm, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem_u32(dst)), "l"((uint64_t)tm), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
// K-major, 128-byte swizzle: 8 rows x 128 B atoms, 1024 B between 8-row groups (SBO), LBO unused.
__device__ __forceinline__ uint64_t desc_sw128(const void* p) {
  uint64_t a = (smem_u32(p) & 0x3FFFF) >> 4;
  return a | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
// K (rows x 64) bf16 as a 2-D map, box = 64 dims x bn keys, 128-byte swizzle.
static CUtensorMap make_kmap(const void* ptr, long rows, int bn) {
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    void* fn = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    g_encode = (PFN_cuTensorMapEncodeTiled_v12000)fn;
  }
  CUtensorMap tm;
  cuuint64_t dims[2] = {64, (cuuint64_t)rows};
  cuuint64_t strides[1] = {128};
  cuuint32_t box[2] = {64, (cuuint32_t)bn};
  cuuint32_t es[2] = {1, 1};
  CUresult r = g_encode(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  return tm;
}
"""


def wgmma_ss_fn(n) -> str:
    """`wgmma_ss_m64n{n}(float (&d)[n/2], uint64_t adesc, uint64_t bdesc, int accum)`: A and B both
    K-major, 128B-swizzled in shared memory."""
    nr = n // 2
    dl = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return f"""
template <int R> __device__ __forceinline__ void fence_regs(float (&d)[R]) {{
#pragma unroll
  for (int i = 0; i < R; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}}
__device__ __forceinline__ void wgmma_ss_m64n{n}(float (&d)[{nr}], uint64_t adesc, uint64_t bdesc, int accum) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nr + 2}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {outs}
               : "l"(adesc), "l"(bdesc), "r"(accum));
}}
"""
