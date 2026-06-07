"""
Multi-GPU RMSNorm with partitioned hidden dimension.

Strategy:
- Place input in symmetric memory; use multimem.ld_reduce on NVSwitch to compute
  global sum-of-squares with a single in-switch reduction.
- Fuse: load bf16 -> upcast -> square-sum (block reduction) -> multimem reduce
  across ranks -> rsqrt -> normalize -> scale by local weight -> store bf16.
- One kernel per row-tile; one symmetric scratch tensor (one float per row) carries
  the partial sum-of-squares between ranks via multimem load-reduce.
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

__device__ void global_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid < (unsigned int)world_size) {
        uint64_t local_base = signal_pad_ptrs[rank];
        uint64_t remote_base = signal_pad_ptrs[tid];
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(
            remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
            local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
        send_signal_relaxed(send_addr);
        wait_signal_relaxed(wait_addr);
    }
    __syncthreads();
}

// Multimem float add load-reduce
__device__ __forceinline__ float multimem_ld_reduce_f32(const float* addr) {
    float v;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
        : "=f"(v) : "l"(addr) : "memory");
    return v;
}
__device__ __forceinline__ void multimem_st_f32(float* addr, float v) {
    asm volatile(
        "multimem.st.relaxed.sys.global.f32 [%0], %1;"
        : : "l"(addr), "f"(v) : "memory");
}

// Phase 1: compute local sum-of-squares per row, write to symmetric scratch
__global__ void rmsnorm_phase1_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ scratch_local,   // symmetric buffer, [num_rows]
    int64_t num_rows,
    int64_t local_hidden
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const __nv_bfloat16* row_ptr = x + (int64_t)row * local_hidden;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float sum = 0.0f;
    // Vectorized load: 8 bf16 = 16 bytes
    int64_t vec_count = local_hidden / 8;
    const uint4* row_v = reinterpret_cast<const uint4*>(row_ptr);
    for (int64_t i = tid; i < vec_count; i += bs) {
        uint4 v = row_v[i];
        __nv_bfloat162 a0 = *reinterpret_cast<__nv_bfloat162*>(&v.x);
        __nv_bfloat162 a1 = *reinterpret_cast<__nv_bfloat162*>(&v.y);
        __nv_bfloat162 a2 = *reinterpret_cast<__nv_bfloat162*>(&v.z);
        __nv_bfloat162 a3 = *reinterpret_cast<__nv_bfloat162*>(&v.w);
        float2 f0 = __bfloat1622float2(a0);
        float2 f1 = __bfloat1622float2(a1);
        float2 f2 = __bfloat1622float2(a2);
        float2 f3 = __bfloat1622float2(a3);
        sum += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y
             + f2.x*f2.x + f2.y*f2.y + f3.x*f3.x + f3.y*f3.y;
    }
    int64_t tail_start = vec_count * 8;
    for (int64_t i = tail_start + tid; i < local_hidden; i += bs) {
        float v = __bfloat162float(row_ptr[i]);
        sum += v * v;
    }

    // block reduce
    __shared__ float sdata[32];
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(mask, sum, off);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (bs + 31) >> 5;
        sum = (lane < nwarps) ? sdata[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(mask, sum, off);
        if (lane == 0) {
            scratch_local[row] = sum;
        }
    }
}

// Phase 2: each rank reduces across ranks via multimem, then normalizes its own slice.
__global__ void rmsnorm_phase2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float* __restrict__ scratch_mc,    // multicast pointer to scratch
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t num_rows,
    int64_t local_hidden,
    int64_t global_hidden,
    float eps,
    int rank,
    int world_size
) {
    // barrier so all ranks have written phase1
    global_barrier(signal_pad_ptrs, blockIdx.x, rank, world_size);

    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int bs = blockDim.x;

    // shared scale
    __shared__ float s_scale;
    if (tid == 0) {
        float total = multimem_ld_reduce_f32(scratch_mc + row);
        float var = total / (float)global_hidden;
        s_scale = rsqrtf(var + eps);
    }
    __syncthreads();
    float scale = s_scale;

    const __nv_bfloat16* x_row = x + (int64_t)row * local_hidden;
    __nv_bfloat16* y_row = y + (int64_t)row * local_hidden;

    int64_t vec_count = local_hidden / 8;
    const uint4* x_v = reinterpret_cast<const uint4*>(x_row);
    uint4* y_v = reinterpret_cast<uint4*>(y_row);
    const uint4* w_v = reinterpret_cast<const uint4*>(weight);

    for (int64_t i = tid; i < vec_count; i += bs) {
        uint4 xv = x_v[i];
        uint4 wv = w_v[i];
        __nv_bfloat162 x0 = *reinterpret_cast<__nv_bfloat162*>(&xv.x);
        __nv_bfloat162 x1 = *reinterpret_cast<__nv_bfloat162*>(&xv.y);
        __nv_bfloat162 x2 = *reinterpret_cast<__nv_bfloat162*>(&xv.z);
        __nv_bfloat162 x3 = *reinterpret_cast<__nv_bfloat162*>(&xv.w);
        __nv_bfloat162 w0 = *reinterpret_cast<__nv_bfloat162*>(&wv.x);
        __nv_bfloat162 w1 = *reinterpret_cast<__nv_bfloat162*>(&wv.y);
        __nv_bfloat162 w2 = *reinterpret_cast<__nv_bfloat162*>(&wv.z);
        __nv_bfloat162 w3 = *reinterpret_cast<__nv_bfloat162*>(&wv.w);

        float2 fx0 = __bfloat1622float2(x0);
        float2 fx1 = __bfloat1622float2(x1);
        float2 fx2 = __bfloat1622float2(x2);
        float2 fx3 = __bfloat1622float2(x3);
        float2 fw0 = __bfloat1622float2(w0);
        float2 fw1 = __bfloat1622float2(w1);
        float2 fw2 = __bfloat1622float2(w2);
        float2 fw3 = __bfloat1622float2(w3);

        float2 r0 = make_float2(fx0.x*scale*fw0.x, fx0.y*scale*fw0.y);
        float2 r1 = make_float2(fx1.x*scale*fw1.x, fx1.y*scale*fw1.y);
        float2 r2 = make_float2(fx2.x*scale*fw2.x, fx2.y*scale*fw2.y);
        float2 r3 = make_float2(fx3.x*scale*fw3.x, fx3.y*scale*fw3.y);

        __nv_bfloat162 o0 = __float22bfloat162_rn(r0);
        __nv_bfloat162 o1 = __float22bfloat162_rn(r1);
        __nv_bfloat162 o2 = __float22bfloat162_rn(r2);
        __nv_bfloat162 o3 = __float22bfloat162_rn(r3);
        uint4 ov;
        ov.x = *reinterpret_cast<uint32_t*>(&o0);
        ov.y = *reinterpret_cast<uint32_t*>(&o1);
        ov.z = *reinterpret_cast<uint32_t*>(&o2);
        ov.w = *reinterpret_cast<uint32_t*>(&o3);
        y_v[i] = ov;
    }
    int64_t tail = vec_count * 8;
    for (int64_t i = tail + tid; i < local_hidden; i += bs) {
        float xv = __bfloat162float(x_row[i]);
        float wv = __bfloat162float(weight[i]);
        float r = xv * scale * wv;
        y_row[i] = __float2bfloat16(r);
    }
}

void launch_rmsnorm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor y,
    torch::Tensor scratch_local,
    int64_t scratch_mc_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t num_rows,
    int64_t local_hidden,
    int64_t global_hidden,
    double eps,
    int64_t rank,
    int64_t world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int block = 256;
    if (local_hidden >= 4096) block = 512;
    if (local_hidden >= 8192) block = 1024;

    rmsnorm_phase1_kernel<<<num_rows, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        scratch_local.data_ptr<float>(),
        num_rows, local_hidden);

    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    rmsnorm_phase2_kernel<<<num_rows, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)y.data_ptr<at::BFloat16>(),
        reinterpret_cast<float*>(static_cast<uintptr_t>(scratch_mc_ptr)),
        d_signal,
        num_rows, local_hidden, global_hidden,
        (float)eps, (int)rank, (int)world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_rmsnorm", &launch_rmsnorm, "Distributed RMSNorm with multimem reduce");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dist_rmsnorm_mm_ext", CUDA_SRC)
    return _ext


_scratch_cache = {}  # num_rows -> (buf, hdl)

def _get_scratch(num_rows: int, device):
    key = (num_rows, device)
    if key in _scratch_cache:
        return _scratch_cache[key]
    buf = symm_mem.empty(num_rows, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _scratch_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(local_hidden_states: torch.Tensor, local_weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    assert local_hidden_states.is_cuda
    assert dist.is_initialized()

    x = local_hidden_states.contiguous()
    orig_shape = x.shape
    local_hidden = orig_shape[-1]
    num_rows = x.numel() // local_hidden
    x2d = x.view(num_rows, local_hidden)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    global_hidden = local_hidden * world_size

    # Fallback: if dtype not bf16, use reference
    if x.dtype != torch.bfloat16:
        input_dtype = x.dtype
        xf = x.to(torch.float32)
        ls = xf.pow(2).sum(dim=-1, keepdim=True)
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        var = ls / global_hidden
        xf = xf * torch.rsqrt(var + variance_epsilon)
        return local_weight * xf.to(input_dtype)

    # Ensure extension compiled before any rank uses it
    _get_ext()
    dist.barrier()

    weight = local_weight.contiguous()
    y = torch.empty_like(x2d)

    scratch_buf, scratch_hdl = _get_scratch(num_rows, x.device)
    signal_dev = scratch_hdl.signal_pad_ptrs_dev
    multicast_ptr = int(scratch_hdl.multicast_ptr)

    _get_ext().launch_rmsnorm(
        x2d, weight, y,
        scratch_buf, multicast_ptr, signal_dev,
        num_rows, local_hidden, global_hidden,
        float(variance_epsilon), rank, world_size,
    )

    return y.view(orig_shape)