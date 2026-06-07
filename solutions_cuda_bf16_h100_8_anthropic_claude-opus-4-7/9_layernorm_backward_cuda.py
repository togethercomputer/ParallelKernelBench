"""
LayerNorm backward param-grad aggregation with fused local reduction +
multimem all-reduce on NVSwitch (H100 + NVLink/NVSwitch).

Strategy:
- Fuse d_beta = sum(dY) and d_gamma = sum(dY * X_hat) into a single CUDA
  kernel that writes both [H] outputs directly into a symmetric memory
  buffer of size [2*H] (bf16).
- Then perform a single multimem.ld_reduce / multimem.st all-reduce on the
  combined [2*H] buffer (one collective instead of two) using NVSwitch
  multimem PTX.
- Falls back to peer-pointer reduction kernel for non-bf16 dtypes or
  non-aligned sizes.
"""

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

// ---------------- signal pad blockwise barrier ----------------
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

__device__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size)
{
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size)
{
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// ---------------- multimem PTX helpers ----------------
__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr,
    uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

// ---------------- multimem all-reduce (bf16, 8 elems per thread) ----------------
__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride)
{
    const uint64_t block_id = blockIdx.x;
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank =
        (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride;
         block_start < numel_per_rank;
         block_start += (int64_t)num_programs * (int64_t)block_stride)
    {
        const int64_t offsets = block_start + (int64_t)tid;
        if (offsets >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// ---------------- fused local LN-backward param-grad reduction (bf16) ----------------
// Computes:
//   out[0..H)   = d_gamma_local = sum_b dY[b,h] * X_hat[b,h]
//   out[H..2H)  = d_beta_local  = sum_b dY[b,h]
// One block per H column-tile (BLOCK_H), threads strided over rows.
template <int BLOCK_H, int BLOCK_R>
__global__ void ln_bwd_partials_bf16_kernel(
    const __nv_bfloat16* __restrict__ dY,
    const __nv_bfloat16* __restrict__ Xh,
    __nv_bfloat16* __restrict__ out,  // size 2*H
    int B, int H)
{
    int h0 = blockIdx.x * BLOCK_H;
    int tx = threadIdx.x;            // 0..BLOCK_H
    int ty = threadIdx.y;            // 0..BLOCK_R
    int h = h0 + tx;

    __shared__ float s_g[BLOCK_R][BLOCK_H];
    __shared__ float s_b[BLOCK_R][BLOCK_H];

    float acc_g = 0.0f, acc_b = 0.0f;
    if (h < H) {
        for (int r = ty; r < B; r += BLOCK_R) {
            float dy = __bfloat162float(dY[r * H + h]);
            float xh = __bfloat162float(Xh[r * H + h]);
            acc_g += dy * xh;
            acc_b += dy;
        }
    }
    s_g[ty][tx] = acc_g;
    s_b[ty][tx] = acc_b;
    __syncthreads();

    // reduce along ty
    if (ty == 0 && h < H) {
        float gg = 0.0f, bb = 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_R; ++i) {
            gg += s_g[i][tx];
            bb += s_b[i][tx];
        }
        out[h]     = __float2bfloat16(gg);
        out[H + h] = __float2bfloat16(bb);
    }
}

// ---------------- fused local for fp32/fp16 fallback (generic via float) ----------------
template <int BLOCK_H, int BLOCK_R>
__global__ void ln_bwd_partials_f32_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ Xh,
    float* __restrict__ out, int B, int H)
{
    int h0 = blockIdx.x * BLOCK_H;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int h = h0 + tx;
    __shared__ float s_g[BLOCK_R][BLOCK_H];
    __shared__ float s_b[BLOCK_R][BLOCK_H];
    float acc_g = 0.0f, acc_b = 0.0f;
    if (h < H) {
        for (int r = ty; r < B; r += BLOCK_R) {
            float dy = dY[r * H + h];
            float xh = Xh[r * H + h];
            acc_g += dy * xh;
            acc_b += dy;
        }
    }
    s_g[ty][tx] = acc_g;
    s_b[ty][tx] = acc_b;
    __syncthreads();
    if (ty == 0 && h < H) {
        float gg = 0.0f, bb = 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_R; ++i) {
            gg += s_g[i][tx];
            bb += s_b[i][tx];
        }
        out[h]     = gg;
        out[H + h] = bb;
    }
}

// ---------------- peer-ptr fallback all-reduce ----------------
__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}
__global__ void allreduce_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size, int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const float* src = (const float*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

// ---------------- launchers ----------------
void launch_ln_partials_bf16(
    torch::Tensor dY, torch::Tensor Xh, torch::Tensor out, int B, int H)
{
    constexpr int BLOCK_H = 128;
    constexpr int BLOCK_R = 4;
    dim3 block(BLOCK_H, BLOCK_R);
    dim3 grid((H + BLOCK_H - 1) / BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ln_bwd_partials_bf16_kernel<BLOCK_H, BLOCK_R><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)dY.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)Xh.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        B, H);
}
void launch_ln_partials_f32(
    torch::Tensor dY, torch::Tensor Xh, torch::Tensor out, int B, int H)
{
    constexpr int BLOCK_H = 128;
    constexpr int BLOCK_R = 4;
    dim3 block(BLOCK_H, BLOCK_R);
    dim3 grid((H + BLOCK_H - 1) / BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ln_bwd_partials_f32_kernel<BLOCK_H, BLOCK_R><<<grid, block, 0, stream>>>(
        dY.data_ptr<float>(), Xh.data_ptr<float>(),
        out.data_ptr<float>(), B, H);
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size, int rank,
    int num_blocks, int block_size, int block_stride)
{
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

void launch_allreduce(
    torch::Tensor ptrs_tensor, torch::Tensor out,
    int64_t n, int dtype_enum)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (dtype_enum == 0) {
        allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
    } else {
        allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<float>(), world_size, n);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_ln_partials_bf16", &launch_ln_partials_bf16);
    m.def("launch_ln_partials_f32", &launch_ln_partials_f32);
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce", &launch_allreduce);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ln_bwd_param_allreduce_ext", CUDA_SRC)
    return _ext


WARP_SIZE = 32
MAX_NUM_BLOCKS = 8
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16 -> 8
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(num_threads, 1):
            block_size *= 2
        block_size = max(block_size, world_size)
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(
            (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
            MAX_NUM_BLOCKS,
        )
    return num_blocks, block_size, block_size


_resource_cache = {}


def _get_resources(H: int, dtype: torch.dtype, device: torch.device):
    key = (H, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    # symm buffer holds [2*H] (concat of d_gamma, d_beta)
    buf = symm_mem.empty(2 * H, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs_tensor)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(X_hat: torch.Tensor, dY: torch.Tensor):
    assert dist.is_initialized()
    assert X_hat.is_cuda and dY.is_cuda
    assert X_hat.is_contiguous() and dY.is_contiguous()
    assert X_hat.shape == dY.shape

    B, H = X_hat.shape
    dtype = X_hat.dtype
    device = X_hat.device

    if not dist.is_initialized() or dist.get_world_size() == 1:
        d_beta = dY.sum(dim=0)
        d_gamma = (dY * X_hat).sum(dim=0)
        return d_gamma, d_beta

    ext = _get_ext()
    buf, hdl, ptrs_tensor = _get_resources(H, dtype, device)

    # Fused local partials directly into symmetric buffer
    if dtype == torch.bfloat16:
        ext.launch_ln_partials_bf16(dY, X_hat, buf, B, H)
    elif dtype == torch.float32:
        ext.launch_ln_partials_f32(dY, X_hat, buf, B, H)
    else:
        # generic fallback via PyTorch into buf
        d_beta = dY.sum(dim=0)
        d_gamma = (dY * X_hat).sum(dim=0)
        buf[:H].copy_(d_gamma)
        buf[H:].copy_(d_beta)

    n = 2 * H

    # Single all-reduce on the combined [2H] symm buffer
    if dtype == torch.bfloat16:
        numel_per_thread = BYTES_PER_THREAD // 2  # 8
        if n % numel_per_thread == 0:
            numel_128 = n // numel_per_thread
            num_blocks, block_size, block_stride = _multimem_launch_config(n, hdl.world_size)
            dist.barrier()
            ext.launch_multimem_allreduce_bf16(
                int(hdl.multicast_ptr),
                hdl.signal_pad_ptrs_dev,
                numel_128,
                hdl.world_size,
                hdl.rank,
                num_blocks,
                block_size,
                block_stride,
            )
            full = buf.clone()
        else:
            hdl.barrier(channel=0)
            full = torch.empty(n, device=device, dtype=dtype)
            ext.launch_allreduce(ptrs_tensor, full, n, 0)
    elif dtype == torch.float32:
        hdl.barrier(channel=0)
        full = torch.empty(n, device=device, dtype=dtype)
        ext.launch_allreduce(ptrs_tensor, full, n, 1)
    else:
        # other dtypes: fallback to NCCL on temporaries
        d_beta_t = buf[H:].clone()
        d_gamma_t = buf[:H].clone()
        dist.all_reduce(d_beta_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(d_gamma_t, op=dist.ReduceOp.SUM)
        return d_gamma_t, d_beta_t

    d_gamma = full[:H].contiguous()
    d_beta = full[H:].contiguous()
    return d_gamma, d_beta