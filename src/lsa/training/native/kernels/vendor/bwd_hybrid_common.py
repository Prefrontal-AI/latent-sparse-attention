# Vendored from frozen_v13/hyb_common.py (md5 117569b86e01731cec3d35146b768c52), by a local Claude Opus 5.5 session, 2026-10-14;
# see __init__.py. Not edited.
# copied from v14.py (wgmma generator + PTX helpers)
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

