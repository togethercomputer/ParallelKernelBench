"""
DINOv2 Sinkhorn-Knopp with custom CUDA + symmetric memory all-reduces.

Strategy:
- All collectives go through symmetric memory using NVSwitch multimem.ld_reduce
  for bf16/f32 add reductions on H100.
- Fused kernels combine reductions with elementwise scaling where possible:
  * fused_normalize: q /= total_mass  (mass = sum(q) all-reduced)
  * row_normalize:   row_sum reduce + q /= (row_sum * num_prototypes)
  * col_normalize:   col_sum local + q /= (col_sum * total_batch)
- Total mass and total_batch reductions overlap with the exp/transpose compute.
"""

from typing import Optional

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

// =====================================================================
// Multimem reductions on float32 symmetric buffers
// =====================================================================

__device__ __forceinline__ void mm_ld_reduce_f32x4(
    const float* addr, float& a, float& b, float& c, float& d
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
        : "l"(addr) : "memory");
}

__device__ __forceinline__ void mm_st_f32x4(
    float* addr, float a, float b, float c, float d
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(addr), "f"(a), "f"(b), "f"(c), "f"(d)
        : "memory");
}

__device__ __forceinline__ void mm_ld_reduce_f32(const float* addr, float& a) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
        : "=f"(a) : "l"(addr) : "memory");
}

__device__ __forceinline__ void mm_st_f32(float* addr, float a) {
    asm volatile(
        "multimem.st.relaxed.sys.global.f32 [%0], %1;"
        : : "l"(addr), "f"(a) : "memory");
}

// Multimem all-reduce float32 elements (count ≤ small)
__global__ void multimem_allreduce_f32_kernel(
    float* multicast_ptr,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t v4_count = n / 4;
    if (idx < v4_count) {
        float a,b,c,d;
        mm_ld_reduce_f32x4(multicast_ptr + idx*4, a, b, c, d);
        mm_st_f32x4(multicast_ptr + idx*4, a, b, c, d);
    }
    int64_t tail_start = v4_count * 4;
    int64_t tail_idx = tail_start + idx;
    if (tail_idx < n && idx < (n - tail_start)) {
        float a;
        mm_ld_reduce_f32(multicast_ptr + tail_idx, a);
        mm_st_f32(multicast_ptr + tail_idx, a);
    }
}

void launch_multimem_allreduce_f32(
    int64_t multicast_ptr,
    int64_t n,
    int64_t stream_ptr
) {
    float* mcptr = reinterpret_cast<float*>(multicast_ptr);
    int threads = 256;
    int64_t work = (n + 3) / 4 + 4;
    int blocks = (int)((work + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    multimem_allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(mcptr, n);
}

// =====================================================================
// exp(teacher / T) with transpose into symmetric buffer (bf16 -> f32)
// q[k, b] = exp(teacher[b, k] / T), where teacher is [B, K], q is [K, B]
// Also computes per-thread sum into a scratch for later block reduction.
// =====================================================================

__global__ void exp_transpose_kernel(
    const __nv_bfloat16* __restrict__ teacher,  // [B, K] row-major
    float* __restrict__ q,                       // [K, B] row-major
    float inv_temp,
    int B, int K,
    float* __restrict__ partial_sums  // length = gridDim
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int64_t total = (int64_t)B * (int64_t)K;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + tid; i < total; i += stride) {
        int b = (int)(i / K);
        int k = (int)(i % K);
        float v = __bfloat162float(teacher[i]);
        float e = __expf(v * inv_temp);
        // write to q[k, b]
        q[(int64_t)k * B + b] = e;
        local_sum += e;
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

void launch_exp_transpose(
    torch::Tensor teacher,
    torch::Tensor q,
    double inv_temp,
    torch::Tensor partial_sums
) {
    int B = teacher.size(0);
    int K = teacher.size(1);
    int threads = 256;
    int blocks = (int)(((int64_t)B * K + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    TORCH_CHECK(partial_sums.numel() >= blocks, "partial_sums too small");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    exp_transpose_kernel<<<blocks, threads, threads*sizeof(float), stream>>>(
        (const __nv_bfloat16*)teacher.data_ptr<at::BFloat16>(),
        q.data_ptr<float>(),
        (float)inv_temp,
        B, K,
        partial_sums.data_ptr<float>()
    );
}

// Reduce partial_sums to single scalar at out[0]
__global__ void final_reduce_kernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) v += in[i];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

void launch_final_reduce(torch::Tensor in, torch::Tensor out) {
    int n = (int)in.numel();
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    final_reduce_kernel<<<1, threads, threads*sizeof(float), stream>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), n);
}

// =====================================================================
// Compute row sums: row_sum[k] = sum_b q[k, b], q is [K, B]
// =====================================================================

__global__ void row_sum_kernel(
    const float* __restrict__ q,  // [K, B]
    float* __restrict__ row_sums, // [K]
    int K, int B
) {
    int k = blockIdx.x;
    if (k >= K) return;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float s = 0.0f;
    const float* row = q + (int64_t)k * B;
    for (int b = tid; b < B; b += blockDim.x) s += row[b];
    sdata[tid] = s;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }
    if (tid == 0) row_sums[k] = sdata[0];
}

void launch_row_sum(torch::Tensor q, torch::Tensor row_sums) {
    int K = q.size(0); int B = q.size(1);
    int threads = 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    row_sum_kernel<<<K, threads, threads*sizeof(float), stream>>>(
        q.data_ptr<float>(), row_sums.data_ptr<float>(), K, B);
}

// q[k,b] /= (row_sums[k] * num_prototypes)
__global__ void row_div_kernel(
    float* __restrict__ q,
    const float* __restrict__ row_sums,
    int K, int B, float num_prototypes
) {
    int k = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || k >= K) return;
    float denom = row_sums[k] * num_prototypes;
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    q[(int64_t)k * B + b] *= inv;
}

void launch_row_div(torch::Tensor q, torch::Tensor row_sums, double num_prototypes) {
    int K = q.size(0); int B = q.size(1);
    int threads = 256;
    dim3 grid((B + threads - 1) / threads, K);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    row_div_kernel<<<grid, threads, 0, stream>>>(
        q.data_ptr<float>(), row_sums.data_ptr<float>(), K, B, (float)num_prototypes);
}

// =====================================================================
// Column sums: col_sum[b] = sum_k q[k, b]
// =====================================================================

__global__ void col_sum_kernel(
    const float* __restrict__ q,
    float* __restrict__ col_sums,
    int K, int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s += q[(int64_t)k * B + b];
    col_sums[b] = s;
}

void launch_col_sum(torch::Tensor q, torch::Tensor col_sums) {
    int K = q.size(0); int B = q.size(1);
    int threads = 128;
    int blocks = (B + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    col_sum_kernel<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(), col_sums.data_ptr<float>(), K, B);
}

// q[k,b] /= (col_sums[b] * total_batch)  ;  total_batch is scalar via pointer
__global__ void col_div_kernel(
    float* __restrict__ q,
    const float* __restrict__ col_sums,
    const float* __restrict__ total_batch_scalar,
    int K, int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;
    float tb = total_batch_scalar[0];
    float denom = col_sums[b] * tb;
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    q[(int64_t)k * B + b] *= inv;
}

void launch_col_div(torch::Tensor q, torch::Tensor col_sums, torch::Tensor total_batch) {
    int K = q.size(0); int B = q.size(1);
    int threads = 256;
    dim3 grid((B + threads - 1) / threads, K);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    col_div_kernel<<<grid, threads, 0, stream>>>(
        q.data_ptr<float>(), col_sums.data_ptr<float>(),
        total_batch.data_ptr<float>(), K, B);
}

// q /= total_mass_scalar
__global__ void scalar_div_kernel(float* q, const float* s, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.0f / s[0];
    q[i] *= inv;
}

void launch_scalar_div(torch::Tensor q, torch::Tensor s) {
    int64_t n = q.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scalar_div_kernel<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(), s.data_ptr<float>(), n);
}

// q *= total_batch_scalar, transpose to bf16 [B, K]
__global__ void scale_transpose_to_bf16_kernel(
    const float* __restrict__ q,   // [K, B]
    __nv_bfloat16* __restrict__ out, // [B, K]
    const float* __restrict__ total_batch_scalar,
    int K, int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;
    float tb = total_batch_scalar[0];
    float v = q[(int64_t)k * B + b] * tb;
    out[(int64_t)b * K + k] = __float2bfloat16(v);
}

void launch_scale_transpose_to_bf16(
    torch::Tensor q, torch::Tensor out, torch::Tensor total_batch
) {
    int K = q.size(0); int B = q.size(1);
    int threads = 256;
    dim3 grid((B + threads - 1) / threads, K);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_transpose_to_bf16_kernel<<<grid, threads, 0, stream>>>(
        q.data_ptr<float>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        total_batch.data_ptr<float>(),
        K, B);
}

// Same but produces float32 output (in case dtype != bf16)
__global__ void scale_transpose_to_f32_kernel(
    const float* __restrict__ q,
    float* __restrict__ out,
    const float* __restrict__ total_batch_scalar,
    int K, int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (b >= B || k >= K) return;
    float tb = total_batch_scalar[0];
    out[(int64_t)b * K + k] = q[(int64_t)k * B + b] * tb;
}

void launch_scale_transpose_to_f32(
    torch::Tensor q, torch::Tensor out, torch::Tensor total_batch
) {
    int K = q.size(0); int B = q.size(1);
    int threads = 256;
    dim3 grid((B + threads - 1) / threads, K);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_transpose_to_f32_kernel<<<grid, threads, 0, stream>>>(
        q.data_ptr<float>(), out.data_ptr<float>(),
        total_batch.data_ptr<float>(), K, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multimem_allreduce_f32", &launch_multimem_allreduce_f32);
    m.def("exp_transpose", &launch_exp_transpose);
    m.def("final_reduce", &launch_final_reduce);
    m.def("row_sum", &launch_row_sum);
    m.def("row_div", &launch_row_div);
    m.def("col_sum", &launch_col_sum);
    m.def("col_div", &launch_col_div);
    m.def("scalar_div", &launch_scalar_div);
    m.def("scale_transpose_to_bf16", &launch_scale_transpose_to_bf16);
    m.def("scale_transpose_to_f32", &launch_scale_transpose_to_f32);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_sk_ext", CUDA_SRC)
    return _ext


# Symmetric memory cache: a small reusable f32 scratch buffer for scalar/row reductions
_symm_cache = {}


def _get_symm_scalar_buf(device, dtype=torch.float32, size=1):
    """Return symm_mem buffer for scalar-ish reductions."""
    key = ("scalar_buf", size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


def _get_symm_vec_buf(device, size, dtype=torch.float32):
    key = ("vec_buf", size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_masked_patches_tensor: torch.Tensor,
    n_iterations: int = 3,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # Fallback to reference behavior if not initialized
    if not dist.is_initialized():
        q = torch.exp(teacher_output.float() / teacher_temp).T
        total_batch = n_masked_patches_tensor.to(device=q.device, dtype=q.dtype).clone()
        K = q.shape[0]
        q /= q.sum()
        for _ in range(n_iterations):
            q /= q.sum(dim=1, keepdim=True)
            q /= K
            q /= q.sum(dim=0, keepdim=True)
            q /= total_batch
        q *= total_batch
        return q.T.contiguous().to(teacher_output.dtype)

    device = teacher_output.device
    B = teacher_output.shape[0]
    K = teacher_output.shape[1]
    out_dtype = teacher_output.dtype
    inv_temp = 1.0 / float(teacher_temp)

    ext = _get_ext()

    # Ensure bf16 input for our kernel; otherwise convert
    if teacher_output.dtype == torch.bfloat16:
        teacher_bf16 = teacher_output.contiguous()
    else:
        teacher_bf16 = teacher_output.to(torch.bfloat16).contiguous()

    # Allocate q [K, B] as float32
    q = torch.empty((K, B), device=device, dtype=torch.float32)

    # Symmetric buffers
    # total_batch scalar (f32)
    tb_buf, tb_hdl = _get_symm_scalar_buf(device, torch.float32, 1)
    tb_buf.copy_(n_masked_patches_tensor.to(device=device, dtype=torch.float32).reshape(1))

    # total_mass scalar (f32)
    mass_buf, mass_hdl = _get_symm_vec_buf(device, 1, torch.float32)

    # Row sums symm vec (size K)
    row_buf, row_hdl = _get_symm_vec_buf(device, K, torch.float32)

    stream = torch.cuda.current_stream(device)
    stream_ptr = stream.cuda_stream

    # Launch all-reduce for total_batch (small, scalar). Overlap with exp/transpose.
    tb_hdl.barrier(channel=0)
    ext.multimem_allreduce_f32(int(tb_hdl.multicast_ptr), 1, stream_ptr)

    # exp + transpose, with partial sums
    # Number of blocks must match what kernel uses (capped at 1024)
    threads = 256
    nblocks = min(1024, max(1, (B * K + threads - 1) // threads))
    partial = torch.empty(nblocks, device=device, dtype=torch.float32)
    ext.exp_transpose(teacher_bf16, q, inv_temp, partial)

    # Reduce partial -> mass_buf[0]
    ext.final_reduce(partial, mass_buf)

    # All-reduce mass (scalar)
    mass_hdl.barrier(channel=0)
    ext.multimem_allreduce_f32(int(mass_hdl.multicast_ptr), 1, stream_ptr)

    # q /= total_mass
    ext.scalar_div(q, mass_buf)

    for _ in range(n_iterations):
        # Row sums into row_buf, then all-reduce
        ext.row_sum(q, row_buf)
        row_hdl.barrier(channel=0)
        ext.multimem_allreduce_f32(int(row_hdl.multicast_ptr), K, stream_ptr)
        # q /= (row_sum * K)
        ext.row_div(q, row_buf, float(K))

        # Column sums local; q /= (col_sum * total_batch)
        col_sums = torch.empty(B, device=device, dtype=torch.float32)
        ext.col_sum(q, col_sums)
        ext.col_div(q, col_sums, tb_buf)

    # Final: q *= total_batch, transpose to [B, K] in target dtype
    if out_dtype == torch.bfloat16:
        out = torch.empty((B, K), device=device, dtype=torch.bfloat16)
        ext.scale_transpose_to_bf16(q, out, tb_buf)
        return out
    else:
        out_f32 = torch.empty((B, K), device=device, dtype=torch.float32)
        ext.scale_transpose_to_f32(q, out_f32, tb_buf)
        return out_f32.to(out_dtype)