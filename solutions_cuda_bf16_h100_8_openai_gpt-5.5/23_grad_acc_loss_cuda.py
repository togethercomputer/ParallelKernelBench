# Strategy:
# - Replace the scalar NCCL all-reduce with one fused CUDA kernel over symmetric-memory UVA peer pointers.
# - The kernel writes this rank's loss contribution, performs a device-side signal-pad barrier,
#   directly loads all peer contributions, reduces them, and computes forward/backward outputs.
# - Forward all-reduce, normalization, and backward gradient math are fused into a single launch.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Optional

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

enum DTypeCode : int {
    DT_BF16 = 0,
    DT_F32  = 1,
    DT_F64  = 2,
    DT_F16  = 3,
    DT_I64  = 4,
    DT_I32  = 5,
    DT_I16  = 6,
    DT_I8   = 7,
    DT_U8   = 8,
    DT_BOOL = 9
};

static int dtype_code(const torch::Tensor& t) {
    const auto st = t.scalar_type();
    if (st == at::kBFloat16) return DT_BF16;
    if (st == at::kFloat)    return DT_F32;
    if (st == at::kDouble)   return DT_F64;
    if (st == at::kHalf)     return DT_F16;
    if (st == at::kLong)     return DT_I64;
    if (st == at::kInt)      return DT_I32;
    if (st == at::kShort)    return DT_I16;
    if (st == at::kChar)     return DT_I8;
    if (st == at::kByte)     return DT_U8;
    if (st == at::kBool)     return DT_BOOL;
    TORCH_CHECK(false, "unsupported dtype");
}

__device__ __forceinline__ double read_scalar_as_double(const void* p, int dt) {
    switch (dt) {
        case DT_BF16:
            return (double)__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
        case DT_F32:
            return (double)(*reinterpret_cast<const float*>(p));
        case DT_F64:
            return *reinterpret_cast<const double*>(p);
        case DT_F16:
            return (double)__half2float(*reinterpret_cast<const __half*>(p));
        case DT_I64:
            return (double)(*reinterpret_cast<const int64_t*>(p));
        case DT_I32:
            return (double)(*reinterpret_cast<const int32_t*>(p));
        case DT_I16:
            return (double)(*reinterpret_cast<const int16_t*>(p));
        case DT_I8:
            return (double)(*reinterpret_cast<const int8_t*>(p));
        case DT_U8:
            return (double)(*reinterpret_cast<const uint8_t*>(p));
        case DT_BOOL:
            return *reinterpret_cast<const bool*>(p) ? 1.0 : 0.0;
        default:
            return 0.0;
    }
}

__device__ __forceinline__ double round_to_dtype(double v, int dt) {
    switch (dt) {
        case DT_BF16:
            return (double)__bfloat162float(__float2bfloat16((float)v));
        case DT_F16:
            return (double)__half2float(__float2half((float)v));
        case DT_F32:
            return (double)((float)v);
        case DT_F64:
            return v;
        default:
            return v;
    }
}

__device__ __forceinline__ void write_scalar_from_double(void* p, int dt, double v) {
    switch (dt) {
        case DT_BF16:
            *reinterpret_cast<__nv_bfloat16*>(p) = __float2bfloat16((float)v);
            break;
        case DT_F32:
            *reinterpret_cast<float*>(p) = (float)v;
            break;
        case DT_F64:
            *reinterpret_cast<double*>(p) = v;
            break;
        case DT_F16:
            *reinterpret_cast<__half*>(p) = __float2half((float)v);
            break;
        default:
            break;
    }
}

__device__ __forceinline__ void send_signal_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void scalar_block_barrier(
    const long long* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size
) {
    const int tid = threadIdx.x;
    if (tid >= world_size) {
        return;
    }

    uint32_t* local_base  = reinterpret_cast<uint32_t*>((uintptr_t)signal_pad_ptrs[rank]);
    uint32_t* remote_base = reinterpret_cast<uint32_t*>((uintptr_t)signal_pad_ptrs[tid]);

    // One CUDA block uses the first world_size x world_size signal slots.
    uint32_t* send_addr = remote_base + rank;
    uint32_t* wait_addr = local_base + tid;

    send_signal_release(send_addr);
    wait_signal_acquire(wait_addr);
}

__global__ void fused_loss_grad_scalar_kernel(
    const void* __restrict__ loss,
    const void* __restrict__ local_valid_tokens,
    const void* __restrict__ global_valid_tokens,
    const void* __restrict__ grad_normalized_loss,
    const void* __restrict__ grad_loss_sum,
    void* __restrict__ symm_contrib,
    const long long* __restrict__ buffer_ptrs,
    const long long* __restrict__ signal_pad_ptrs,
    void* __restrict__ normalized_loss_out,
    void* __restrict__ loss_sum_out,
    void* __restrict__ grad_loss_out,
    int loss_dt,
    int local_dt,
    int global_dt,
    int grad_norm_dt,
    int grad_sum_dt,
    int has_grad_loss_sum,
    int rank,
    int world_size
) {
    const int tid = threadIdx.x;

    if (tid == 0) {
        const double local_tokens = read_scalar_as_double(local_valid_tokens, local_dt);
        double local_loss = read_scalar_as_double(loss, loss_dt);

        double contrib;
        if (local_tokens == 0.0) {
            // Matches nan_to_num(loss) followed by multiplication by zero:
            // NaN/Inf do not poison the reduction when this rank has no valid tokens.
            contrib = 0.0;
        } else {
            contrib = round_to_dtype(local_loss * local_tokens, loss_dt);
        }

        write_scalar_from_double(symm_contrib, loss_dt, contrib);
        __threadfence_system();
    }

    __syncthreads();

    // Device-side inter-rank synchronization: no NCCL/all_reduce.
    scalar_block_barrier(signal_pad_ptrs, rank, world_size);

    __syncthreads();

    if (tid == 0) {
        double reduced = 0.0;

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < world_size) {
                const void* peer_ptr = reinterpret_cast<const void*>((uintptr_t)buffer_ptrs[r]);
                const double v = read_scalar_as_double(peer_ptr, loss_dt);
                reduced += v;
            }
        }

        reduced = round_to_dtype(reduced, loss_dt);

        const double global_tokens = read_scalar_as_double(global_valid_tokens, global_dt);
        const double normalized = round_to_dtype(reduced / global_tokens, loss_dt);

        write_scalar_from_double(loss_sum_out, loss_dt, reduced);
        write_scalar_from_double(normalized_loss_out, loss_dt, normalized);

        const double local_tokens = read_scalar_as_double(local_valid_tokens, local_dt);
        const double grad_norm = read_scalar_as_double(grad_normalized_loss, grad_norm_dt);

        double grad_from_normalized = round_to_dtype(grad_norm * local_tokens, grad_norm_dt);
        grad_from_normalized = round_to_dtype(grad_from_normalized / global_tokens, grad_norm_dt);

        double grad_from_sum = 0.0;
        if (has_grad_loss_sum) {
            const double gs = read_scalar_as_double(grad_loss_sum, grad_sum_dt);
            grad_from_sum = round_to_dtype(gs * local_tokens, grad_norm_dt);
        }

        const double grad_loss = round_to_dtype(grad_from_normalized + grad_from_sum, grad_norm_dt);
        write_scalar_from_double(grad_loss_out, grad_norm_dt, grad_loss);
    }
}

void launch_fused_loss_grad_scalar(
    torch::Tensor loss,
    torch::Tensor local_valid_tokens,
    torch::Tensor global_valid_tokens,
    torch::Tensor grad_normalized_loss,
    torch::Tensor grad_loss_sum,
    torch::Tensor symm_contrib,
    torch::Tensor buffer_ptrs,
    torch::Tensor signal_pad_ptrs,
    torch::Tensor normalized_loss_out,
    torch::Tensor loss_sum_out,
    torch::Tensor grad_loss_out,
    bool has_grad_loss_sum,
    int rank,
    int world_size
) {
    TORCH_CHECK(loss.is_cuda(), "loss must be CUDA");
    TORCH_CHECK(local_valid_tokens.is_cuda(), "local_valid_tokens must be CUDA");
    TORCH_CHECK(global_valid_tokens.is_cuda(), "global_valid_tokens must be CUDA");
    TORCH_CHECK(grad_normalized_loss.is_cuda(), "grad_normalized_loss must be CUDA");
    TORCH_CHECK(symm_contrib.is_cuda(), "symm_contrib must be CUDA");
    TORCH_CHECK(buffer_ptrs.is_cuda(), "buffer_ptrs must be CUDA");
    TORCH_CHECK(signal_pad_ptrs.is_cuda(), "signal_pad_ptrs must be CUDA");
    TORCH_CHECK(normalized_loss_out.is_cuda(), "normalized_loss_out must be CUDA");
    TORCH_CHECK(loss_sum_out.is_cuda(), "loss_sum_out must be CUDA");
    TORCH_CHECK(grad_loss_out.is_cuda(), "grad_loss_out must be CUDA");

    TORCH_CHECK(loss.numel() == 1, "loss must be scalar/one element");
    TORCH_CHECK(local_valid_tokens.numel() == 1, "local_valid_tokens must be scalar/one element");
    TORCH_CHECK(global_valid_tokens.numel() == 1, "global_valid_tokens must be scalar/one element");
    TORCH_CHECK(grad_normalized_loss.numel() == 1, "grad_normalized_loss must be scalar/one element");
    TORCH_CHECK(!has_grad_loss_sum || grad_loss_sum.numel() == 1, "grad_loss_sum must be scalar/one element");
    TORCH_CHECK(symm_contrib.numel() == 1, "symm_contrib must be one element");
    TORCH_CHECK(buffer_ptrs.numel() >= world_size, "buffer_ptrs too small");
    TORCH_CHECK(signal_pad_ptrs.numel() >= world_size, "signal_pad_ptrs too small");

    const int loss_dt = dtype_code(loss);
    const int local_dt = dtype_code(local_valid_tokens);
    const int global_dt = dtype_code(global_valid_tokens);
    const int grad_norm_dt = dtype_code(grad_normalized_loss);
    const int grad_sum_dt = has_grad_loss_sum ? dtype_code(grad_loss_sum) : grad_norm_dt;

    TORCH_CHECK(
        loss_dt == DT_BF16 || loss_dt == DT_F32 || loss_dt == DT_F64 || loss_dt == DT_F16,
        "loss dtype must be floating"
    );
    TORCH_CHECK(
        grad_norm_dt == DT_BF16 || grad_norm_dt == DT_F32 || grad_norm_dt == DT_F64 || grad_norm_dt == DT_F16,
        "grad_normalized_loss dtype must be floating"
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    fused_loss_grad_scalar_kernel<<<1, 32, 0, stream>>>(
        loss.data_ptr(),
        local_valid_tokens.data_ptr(),
        global_valid_tokens.data_ptr(),
        grad_normalized_loss.data_ptr(),
        has_grad_loss_sum ? grad_loss_sum.data_ptr() : grad_normalized_loss.data_ptr(),
        symm_contrib.data_ptr(),
        reinterpret_cast<const long long*>(buffer_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(signal_pad_ptrs.data_ptr<int64_t>()),
        normalized_loss_out.data_ptr(),
        loss_sum_out.data_ptr(),
        grad_loss_out.data_ptr(),
        loss_dt,
        local_dt,
        global_dt,
        grad_norm_dt,
        grad_sum_dt,
        has_grad_loss_sum ? 1 : 0,
        rank,
        world_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "launch_fused_loss_grad_scalar",
        &launch_fused_loss_grad_scalar,
        "Fused scalar loss normalization + symmetric-memory all-reduce + backward"
    );
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_loss_grad_symm_scalar_bf16_h100_ext", CUDA_SRC)
    return _ext


def _device_key(device: torch.device):
    return (device.type, device.index)


def _get_resources(loss: torch.Tensor, grad_normalized_loss: torch.Tensor):
    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()

    key = (
        _device_key(loss.device),
        loss.dtype,
        tuple(loss.shape),
        grad_normalized_loss.dtype,
        tuple(grad_normalized_loss.shape),
        world_size,
    )

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    # Symmetric one-scalar contribution buffer. Each rank publishes its local
    # loss * local_valid_tokens contribution here; peers read through UVA ptrs.
    symm_contrib = symm_mem.empty((1,), device=loss.device, dtype=loss.dtype)
    hdl = symm_mem.rendezvous(symm_contrib, dist.group.WORLD)

    buffer_ptrs = torch.tensor(hdl.buffer_ptrs, device=loss.device, dtype=torch.int64)

    normalized_loss_out = torch.empty_like(loss)
    loss_sum_out = torch.empty_like(loss)
    grad_loss_out = torch.empty_like(grad_normalized_loss)

    cached = (
        symm_contrib,
        hdl,
        buffer_ptrs,
        hdl.signal_pad_ptrs_dev,
        normalized_loss_out,
        loss_sum_out,
        grad_loss_out,
    )
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused forward/backward for scalar loss normalization.

    Returns:
        (normalized_loss, loss_sum, grad_loss)
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert loss.is_cuda, "loss must be CUDA"
    assert local_valid_tokens.is_cuda, "local_valid_tokens must be CUDA"
    assert global_valid_tokens.is_cuda, "global_valid_tokens must be CUDA"
    assert grad_normalized_loss.is_cuda, "grad_normalized_loss must be CUDA"
    assert loss.numel() == 1, "loss must be scalar/one element"
    assert local_valid_tokens.numel() == 1, "local_valid_tokens must be scalar/one element"
    assert global_valid_tokens.numel() == 1, "global_valid_tokens must be scalar/one element"
    assert grad_normalized_loss.numel() == 1, "grad_normalized_loss must be scalar/one element"

    if grad_loss_sum is not None:
        assert grad_loss_sum.is_cuda, "grad_loss_sum must be CUDA"
        assert grad_loss_sum.numel() == 1, "grad_loss_sum must be scalar/one element"

    ext = _get_ext()

    (
        symm_contrib,
        _hdl,
        buffer_ptrs,
        signal_pad_ptrs,
        normalized_loss_out,
        loss_sum_out,
        grad_loss_out,
    ) = _get_resources(loss, grad_normalized_loss)

    dummy_grad_sum = grad_loss_sum if grad_loss_sum is not None else grad_normalized_loss

    ext.launch_fused_loss_grad_scalar(
        loss,
        local_valid_tokens,
        global_valid_tokens,
        grad_normalized_loss,
        dummy_grad_sum,
        symm_contrib,
        buffer_ptrs,
        signal_pad_ptrs,
        normalized_loss_out,
        loss_sum_out,
        grad_loss_out,
        grad_loss_sum is not None,
        dist.get_rank(),
        dist.get_world_size(),
    )

    return normalized_loss_out, loss_sum_out, grad_loss_out