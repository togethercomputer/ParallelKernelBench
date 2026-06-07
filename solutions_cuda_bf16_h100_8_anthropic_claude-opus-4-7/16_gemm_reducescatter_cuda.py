import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

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

__device__ void barrier_block(
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size,
    bool acq_rel
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    if (acq_rel) {
        send_signal_acq_rel(send_addr);
        wait_signal_acq_rel(wait_addr);
    } else {
        send_signal_relaxed(send_addr);
        wait_signal_relaxed(wait_addr);
    }
}

// Reduce a slice [M_local, N] from all peers' C_partial buffers.
// Each rank reads rank*M_local..(rank+1)*M_local rows from every peer.
// peer buffer layout: [M, N] bf16, row-major.
__global__ void reduce_scatter_bf16_kernel(
    const uint64_t* __restrict__ buf_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t M_local,
    int64_t N,
    int rank,
    int world_size
) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    barrier_block(signal_pad_ptrs, bid, rank, world_size, false);
    __syncthreads();

    int64_t total = M_local * N;
    int64_t row_off = (int64_t)rank * M_local;  // start row in peer's [M,N]
    int64_t base = row_off * N;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Process 8 bf16 (16 bytes) per iteration when aligned
    int64_t total8 = total / 8;
    for (int64_t i = tid; i < total8; i += stride) {
        int64_t elem = i * 8;
        float acc[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = 0.0f;

        #pragma unroll 1
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src =
                reinterpret_cast<const __nv_bfloat16*>(buf_ptrs[r]) + base + elem;
            uint4 v = *reinterpret_cast<const uint4*>(src);
            __nv_bfloat162 a0 = *reinterpret_cast<__nv_bfloat162*>(&v.x);
            __nv_bfloat162 a1 = *reinterpret_cast<__nv_bfloat162*>(&v.y);
            __nv_bfloat162 a2 = *reinterpret_cast<__nv_bfloat162*>(&v.z);
            __nv_bfloat162 a3 = *reinterpret_cast<__nv_bfloat162*>(&v.w);
            float2 f0 = __bfloat1622float2(a0);
            float2 f1 = __bfloat1622float2(a1);
            float2 f2 = __bfloat1622float2(a2);
            float2 f3 = __bfloat1622float2(a3);
            acc[0] += f0.x; acc[1] += f0.y;
            acc[2] += f1.x; acc[3] += f1.y;
            acc[4] += f2.x; acc[5] += f2.y;
            acc[6] += f3.x; acc[7] += f3.y;
        }

        __nv_bfloat162 o0 = __floats2bfloat162_rn(acc[0], acc[1]);
        __nv_bfloat162 o1 = __floats2bfloat162_rn(acc[2], acc[3]);
        __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4], acc[5]);
        __nv_bfloat162 o3 = __floats2bfloat162_rn(acc[6], acc[7]);
        uint4 outv;
        outv.x = *reinterpret_cast<uint32_t*>(&o0);
        outv.y = *reinterpret_cast<uint32_t*>(&o1);
        outv.z = *reinterpret_cast<uint32_t*>(&o2);
        outv.w = *reinterpret_cast<uint32_t*>(&o3);
        *reinterpret_cast<uint4*>(out + elem) = outv;
    }

    // Tail
    int64_t tail_start = total8 * 8;
    for (int64_t i = tail_start + tid; i < total; i += stride) {
        float s = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src =
                reinterpret_cast<const __nv_bfloat16*>(buf_ptrs[r]) + base + i;
            s += __bfloat162float(*src);
        }
        out[i] = __float2bfloat16(s);
    }

    __syncthreads();
    barrier_block(signal_pad_ptrs, bid, rank, world_size, true);
}

__global__ void reduce_scatter_f32_kernel(
    const uint64_t* __restrict__ buf_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    float* __restrict__ out,
    int64_t M_local,
    int64_t N,
    int rank,
    int world_size
) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    barrier_block(signal_pad_ptrs, bid, rank, world_size, false);
    __syncthreads();

    int64_t total = M_local * N;
    int64_t base = (int64_t)rank * M_local * N;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < total; i += stride) {
        float s = 0.0f;
        #pragma unroll 1
        for (int r = 0; r < world_size; ++r) {
            const float* src = reinterpret_cast<const float*>(buf_ptrs[r]) + base + i;
            s += *src;
        }
        out[i] = s;
    }

    __syncthreads();
    barrier_block(signal_pad_ptrs, bid, rank, world_size, true);
}

void launch_reduce_scatter_bf16(
    torch::Tensor buf_ptrs,       // int64 [world_size]
    torch::Tensor signal_ptrs,    // int64 [world_size]
    torch::Tensor out,
    int64_t M_local,
    int64_t N,
    int rank,
    int world_size
) {
    const uint64_t* d_buf = reinterpret_cast<const uint64_t*>(buf_ptrs.data_ptr<int64_t>());
    const uint64_t* d_sig = reinterpret_cast<const uint64_t*>(signal_ptrs.data_ptr<int64_t>());
    int threads = 256;
    int64_t total = M_local * N;
    int64_t total8 = (total + 7) / 8;
    int blocks = (int)((total8 + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 512) blocks = 512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_scatter_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_buf, d_sig,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        M_local, N, rank, world_size);
}

void launch_reduce_scatter_f32(
    torch::Tensor buf_ptrs,
    torch::Tensor signal_ptrs,
    torch::Tensor out,
    int64_t M_local,
    int64_t N,
    int rank,
    int world_size
) {
    const uint64_t* d_buf = reinterpret_cast<const uint64_t*>(buf_ptrs.data_ptr<int64_t>());
    const uint64_t* d_sig = reinterpret_cast<const uint64_t*>(signal_ptrs.data_ptr<int64_t>());
    int threads = 256;
    int64_t total = M_local * N;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 512) blocks = 512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_scatter_f32_kernel<<<blocks, threads, 0, stream>>>(
        d_buf, d_sig, out.data_ptr<float>(),
        M_local, N, rank, world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_reduce_scatter_bf16", &launch_reduce_scatter_bf16, "rs bf16");
    m.def("launch_reduce_scatter_f32", &launch_reduce_scatter_f32, "rs f32");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_rs_ext", CUDA_SRC)
    return _ext

_cache = {}

def _get_resources(M, N, dtype, device):
    key = (M, N, dtype, device)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty((M, N), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    buf_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    res = (buf, hdl, buf_ptrs, sig_ptrs)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(A_local: torch.Tensor, B_local: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, K_local = A_local.shape
    _, N = B_local.shape
    M_local = M // world_size
    dtype = A_local.dtype
    device = A_local.device

    A_local = A_local.contiguous()
    B_local = B_local.contiguous()

    # Compile ext (rank 0 first to avoid race)
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    buf, hdl, buf_ptrs, sig_ptrs = _get_resources(M, N, dtype, device)

    # Local matmul directly into symmetric buffer
    torch.matmul(A_local, B_local, out=buf)

    out = torch.empty((M_local, N), dtype=dtype, device=device)

    if dtype == torch.bfloat16:
        ext.launch_reduce_scatter_bf16(buf_ptrs, sig_ptrs, out, M_local, N, rank, world_size)
    elif dtype == torch.float32:
        ext.launch_reduce_scatter_f32(buf_ptrs, sig_ptrs, out, M_local, N, rank, world_size)
    else:
        # Fallback: cast to f32 path via clone
        buf_f = buf.float().contiguous()
        # Use NCCL fallback
        C_local_f = torch.empty((M_local, N), dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(C_local_f, buf_f, op=dist.ReduceOp.SUM)
        out = C_local_f.to(dtype)

    return out