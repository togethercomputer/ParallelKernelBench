"""
TP Muon Newton-Schulz with symm_mem multimem all-reduce replacing NCCL.
"""

from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


_COEFFICIENTS: dict[str, Sequence[tuple[float, float, float]]] = {
    "simple": ((3.4445, -4.7750, 2.0315),),
    "quintic": (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ),
    "polar_express": (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4647, 0.5074),
        (2.2779, -1.6447, 0.4162),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ),
    "aol": (
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ),
}


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void barrier_relaxed(const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int t = threadIdx.x;
    if (t >= (unsigned int)world_size) return;
    uint64_t lb = signal_pad_ptrs[rank];
    uint64_t rb = signal_pad_ptrs[t];
    uint32_t* s = reinterpret_cast<uint32_t*>(rb + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* w = reinterpret_cast<uint32_t*>(lb + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_relaxed(s); wait_signal_relaxed(w);
}
__device__ void barrier_acq_rel(const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int t = threadIdx.x;
    if (t >= (unsigned int)world_size) return;
    uint64_t lb = signal_pad_ptrs[rank];
    uint64_t rb = signal_pad_ptrs[t];
    uint32_t* s = reinterpret_cast<uint32_t*>(rb + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* w = reinterpret_cast<uint32_t*>(lb + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_acq_rel(s); wait_signal_acq_rel(w);
}

// f32x4 multimem reduce/store
__device__ __forceinline__ void mm_ldred_f32x4(const uint64_t* a, uint32_t& x, uint32_t& y, uint32_t& z, uint32_t& w) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a) : "memory");
}
__device__ __forceinline__ void mm_st_f32x4(const uint64_t* a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "l"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ void mm_ldred_f32(const uint64_t* a, uint32_t& x) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
        : "=r"(x) : "l"(a) : "memory");
}
__device__ __forceinline__ void mm_st_f32(const uint64_t* a, uint32_t x) {
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;"
        :: "l"(a), "r"(x) : "memory");
}

// bf16x2 v4 multimem
__device__ __forceinline__ void mm_ldred_bf16x4(const uint64_t* a, uint32_t& x, uint32_t& y, uint32_t& z, uint32_t& w) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a) : "memory");
}
__device__ __forceinline__ void mm_st_bf16x4(const uint64_t* a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "l"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

// All-reduce kernel for f32 (in-place on symm buffer via multicast)
__global__ void mm_allreduce_f32_kernel(
    uint64_t mc_base, const uint64_t* sig, int64_t n, int world_size, int rank
) {
    uint64_t bid = blockIdx.x;
    barrier_relaxed(sig, bid, rank, world_size);
    __syncthreads();

    int64_t per_rank4 = ((n / 4) + world_size - 1) / world_size;
    int64_t base4 = (int64_t)rank * per_rank4;
    int64_t total4 = n / 4;
    int tid = threadIdx.x;
    int nthr = blockDim.x * gridDim.x;
    int64_t gtid = (int64_t)blockIdx.x * blockDim.x + tid;

    for (int64_t i = gtid; i < per_rank4; i += nthr) {
        int64_t idx4 = base4 + i;
        if (idx4 >= total4) break;
        uint64_t* p = reinterpret_cast<uint64_t*>(mc_base) + idx4 * 2; // 16B units
        uint32_t x,y,z,w;
        mm_ldred_f32x4(p, x, y, z, w);
        mm_st_f32x4(p, x, y, z, w);
    }
    // tail elements (n % 4)
    int64_t tail_start = (n / 4) * 4;
    if (gtid == 0) {
        for (int64_t i = tail_start; i < n; ++i) {
            uint32_t* fp = reinterpret_cast<uint32_t*>(mc_base) + i;
            uint64_t* p64 = reinterpret_cast<uint64_t*>(fp);
            uint32_t v;
            mm_ldred_f32(p64, v);
            mm_st_f32(p64, v);
        }
    }

    __syncthreads();
    barrier_acq_rel(sig, bid, rank, world_size);
}

// bf16 all-reduce in place
__global__ void mm_allreduce_bf16_kernel(
    uint64_t mc_base, const uint64_t* sig, int64_t numel_128, int world_size, int rank
) {
    uint64_t bid = blockIdx.x;
    barrier_relaxed(sig, bid, rank, world_size);
    __syncthreads();

    int64_t per_rank = (numel_128 + world_size - 1) / world_size;
    int tid = threadIdx.x;
    int nthr = blockDim.x * gridDim.x;
    int64_t gtid = (int64_t)blockIdx.x * blockDim.x + tid;

    for (int64_t i = gtid; i < per_rank; i += nthr) {
        int64_t idx = (int64_t)rank * per_rank + i;
        if (idx >= numel_128) break;
        uint64_t* p = reinterpret_cast<uint64_t*>(mc_base) + idx * 2;
        uint32_t x,y,z,w;
        mm_ldred_bf16x4(p, x, y, z, w);
        mm_st_bf16x4(p, x, y, z, w);
    }
    __syncthreads();
    barrier_acq_rel(sig, bid, rank, world_size);
}

void launch_mm_allreduce_f32(uint64_t mc_ptr, torch::Tensor sig_dev, int64_t n,
                              int world_size, int rank, int blocks, int threads) {
    const uint64_t* s = reinterpret_cast<const uint64_t*>(sig_dev.data_ptr<int64_t>());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    mm_allreduce_f32_kernel<<<blocks, threads, 0, st>>>(mc_ptr, s, n, world_size, rank);
}
void launch_mm_allreduce_bf16(uint64_t mc_ptr, torch::Tensor sig_dev, int64_t numel_128,
                               int world_size, int rank, int blocks, int threads) {
    const uint64_t* s = reinterpret_cast<const uint64_t*>(sig_dev.data_ptr<int64_t>());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    mm_allreduce_bf16_kernel<<<blocks, threads, 0, st>>>(mc_ptr, s, numel_128, world_size, rank);
}

// Peer-pointer fallback (no multicast)
__global__ void p2p_allreduce_f32_kernel(const long long* ptrs, float* out, int ws, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float s = 0.f;
        for (int r = 0; r < ws; ++r) s += ((const float*)ptrs[r])[idx];
        out[idx] = s;
    }
}
void launch_p2p_allreduce_f32(torch::Tensor ptrs, torch::Tensor out, int64_t n) {
    int ws = ptrs.size(0);
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    p2p_allreduce_f32_kernel<<<blocks, threads, 0, st>>>(
        (const long long*)ptrs.data_ptr<int64_t>(), out.data_ptr<float>(), ws, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_mm_allreduce_f32", &launch_mm_allreduce_f32);
    m.def("launch_mm_allreduce_bf16", &launch_mm_allreduce_bf16);
    m.def("launch_p2p_allreduce_f32", &launch_p2p_allreduce_f32);
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("muon_tp_symmmem_ext", CUDA_SRC)
    return _ext


_buf_cache: dict = {}

def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, group):
    key = (numel, dtype, device, id(group))
    e = _buf_cache.get(key)
    if e is not None:
        return e
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    e = (buf, hdl, ptrs_tensor)
    _buf_cache[key] = e
    return e


def _has_multicast(hdl) -> bool:
    try:
        return int(hdl.multicast_ptr) != 0
    except Exception:
        return False


def _allreduce_inplace(t: torch.Tensor, group) -> torch.Tensor:
    """All-reduce SUM via symm_mem multimem; returns tensor with same shape/dtype."""
    assert t.is_cuda and t.is_contiguous()
    n = t.numel()
    dtype = t.dtype
    device = t.device

    buf, hdl, ptrs_tensor = _get_symm_buf(n, dtype, device, group)
    buf.copy_(t.view(-1))

    ws = hdl.world_size
    rank = hdl.rank
    ext = _get_ext()

    if dtype == torch.bfloat16 and (n * 2) % 16 == 0 and _has_multicast(hdl):
        numel_128 = (n * 2) // 16  # 16-byte chunks of bf16
        threads = 256
        blocks = max(1, min(8, (numel_128 + ws - 1) // ws // threads + 1))
        # ensure all ranks have written buf
        dist.barrier(group=group)
        ext.launch_mm_allreduce_bf16(int(hdl.multicast_ptr), hdl.signal_pad_ptrs_dev,
                                     numel_128, ws, rank, blocks, threads)
        out = buf.clone().view_as(t)
        return out

    if dtype == torch.float32 and _has_multicast(hdl):
        threads = 256
        blocks = max(1, min(8, (n + ws - 1) // ws // threads + 1))
        dist.barrier(group=group)
        ext.launch_mm_allreduce_f32(int(hdl.multicast_ptr), hdl.signal_pad_ptrs_dev,
                                    n, ws, rank, blocks, threads)
        out = buf.clone().view_as(t)
        return out

    # peer-pointer fallback (f32)
    if dtype == torch.float32:
        hdl.barrier(channel=0)
        out_flat = torch.empty(n, device=device, dtype=dtype)
        ext.launch_p2p_allreduce_f32(ptrs_tensor, out_flat, n)
        return out_flat.view_as(t)

    # generic fallback
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def _coefficient_at(coefficients, step):
    return coefficients[step % len(coefficients)]


def _distributed_normalize_symm(x: torch.Tensor, group, eps: float = 1e-7) -> torch.Tensor:
    norm_sq = (x * x).sum().reshape(1).contiguous()
    norm_sq = _allreduce_inplace(norm_sq, group)
    return x / torch.sqrt(norm_sq).clamp_min(eps)


def _ns_step_symm(x: torch.Tensor, a: float, b: float, c: float, group) -> torch.Tensor:
    gram = x @ x.mT
    gram = gram.contiguous()
    gram = _allreduce_inplace(gram, group)
    update = torch.addmm(gram, gram, gram, alpha=c, beta=b)
    return torch.addmm(x, update, x, alpha=1.0, beta=a)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    steps: int = 5,
    coefficient_type: str = "quintic",
    partition_dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert x.ndim == 2
    assert x.dtype == torch.float32
    assert coefficient_type in _COEFFICIENTS
    coefficients = _COEFFICIENTS[coefficient_type]
    assert steps % len(coefficients) == 0

    # Pre-compile on rank 0 then sync
    if dist.get_rank(group) == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    if partition_dim == 0:
        x_work = x.mT.contiguous()
    elif partition_dim == 1:
        x_work = x
    else:
        raise AssertionError("invalid partition_dim")

    # Cast to bf16 for NS iteration (per hardware note); norm in fp32 for stability.
    x_work = _distributed_normalize_symm(x_work, group)
    x_bf = x_work.to(torch.bfloat16).contiguous()

    for step in range(steps):
        a, b, c = _coefficient_at(coefficients, step)
        # bf16 matmuls via tensor cores
        gram = x_bf @ x_bf.mT  # bf16 @ bf16 -> bf16 (torch will use TC)
        gram = gram.contiguous()
        gram = _allreduce_inplace(gram, group)
        update = torch.addmm(gram, gram, gram, alpha=c, beta=b)
        x_bf = torch.addmm(x_bf, update, x_bf, alpha=1.0, beta=a)

    x_work = x_bf.to(torch.float32)
    if partition_dim == 0:
        return x_work.mT.contiguous()
    return x_work.contiguous()