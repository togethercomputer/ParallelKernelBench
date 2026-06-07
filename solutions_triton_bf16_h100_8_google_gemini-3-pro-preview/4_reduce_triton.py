"""
Strategy:
- **Device-Side Communication via UVA:** Replaced NCCL `reduce` with a direct memory pull over NVLink. Ranks expose their local tensors via `torch.distributed._symmetric_memory`, allowing the destination rank to concurrently read all peers' inputs directly into the local output tensor.
- **Compute-Communication Overlap & Vectorization:** The custom CUDA kernel on the destination rank overlaps cross-GPU memory reads with local accumulation. To maximize memory bandwidth over NVLink, we cast `bfloat16`/`float16` pointers to 128-bit `uint4` types when perfectly aligned, fetching and reducing 8 elements per instruction.
- **Barrier Safety:** Stream-aware symmetric memory barriers (`hdl.barrier`) ensure data is securely committed to the symmetric buffer before reads begin and protected from subsequent overwrites until the destination's reduction finishes.
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
#include <cuda_fp16.h>

template<typename T>
struct Ptrs {
    const T* p[32];
};

template<typename T, typename AccT>
__global__ void reduce_generic_kernel(
    Ptrs<T> ptrs,
    T* __restrict__ out,
    int64_t n,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        AccT sum = 0;
        for (int i = 0; i < world_size; ++i) {
            sum += static_cast<AccT>(ptrs.p[i][idx]);
        }
        out[idx] = static_cast<T>(sum);
    }
}

__global__ void reduce_bf16_kernel_vec8(
    Ptrs<__nv_bfloat16> ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t n_vec,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float sums[8] = {0.0f};
        for (int i = 0; i < world_size; ++i) {
            const uint4* p = reinterpret_cast<const uint4*>(ptrs.p[i]);
            uint4 val = p[idx];
            __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&val);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sums[j] += __bfloat162float(vals[j]);
            }
        }
        uint4 out_val;
        __nv_bfloat16* out_vals = reinterpret_cast<__nv_bfloat16*>(&out_val);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_vals[j] = __float2bfloat16(sums[j]);
        }
        reinterpret_cast<uint4*>(out)[idx] = out_val;
    }
}

__global__ void reduce_bf16_kernel_scalar(
    Ptrs<__nv_bfloat16> ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t offset,
    int64_t n,
    int world_size
) {
    int64_t idx = offset + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < world_size; ++i) {
            sum += __bfloat162float(ptrs.p[i][idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

__global__ void reduce_fp16_kernel_vec8(
    Ptrs<__half> ptrs,
    __half* __restrict__ out,
    int64_t n_vec,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float sums[8] = {0.0f};
        for (int i = 0; i < world_size; ++i) {
            const uint4* p = reinterpret_cast<const uint4*>(ptrs.p[i]);
            uint4 val = p[idx];
            __half* vals = reinterpret_cast<__half*>(&val);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sums[j] += __half2float(vals[j]);
            }
        }
        uint4 out_val;
        __half* out_vals = reinterpret_cast<__half*>(&out_val);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_vals[j] = __float2half(sums[j]);
        }
        reinterpret_cast<uint4*>(out)[idx] = out_val;
    }
}

__global__ void reduce_fp16_kernel_scalar(
    Ptrs<__half> ptrs,
    __half* __restrict__ out,
    int64_t offset,
    int64_t n,
    int world_size
) {
    int64_t idx = offset + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < world_size; ++i) {
            sum += __half2float(ptrs.p[i][idx]);
        }
        out[idx] = __float2half(sum);
    }
}

void reduce_cuda(
    std::vector<int64_t> ptrs_int,
    torch::Tensor out,
    int64_t n
) {
    int world_size = ptrs_int.size();
    TORCH_CHECK(world_size <= 32, "World size too large");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;
    
    if (out.scalar_type() == at::ScalarType::BFloat16) {
        Ptrs<__nv_bfloat16> ptrs;
        bool aligned = (reinterpret_cast<uintptr_t>(out.data_ptr<at::BFloat16>()) % 16 == 0);
        for (int i = 0; i < world_size; ++i) {
            ptrs.p[i] = reinterpret_cast<const __nv_bfloat16*>(ptrs_int[i]);
            if (ptrs_int[i] % 16 != 0) aligned = false;
        }
        if (aligned) {
            int64_t n_vec = n / 8;
            int64_t n_tail = n % 8;
            if (n_vec > 0) {
                int blocks = (n_vec + threads - 1) / threads;
                reduce_bf16_kernel_vec8<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n_vec, world_size);
            }
            if (n_tail > 0) {
                int blocks = (n_tail + threads - 1) / threads;
                reduce_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n_vec * 8, n, world_size);
            }
        } else {
            int blocks = (n + threads - 1) / threads;
            reduce_bf16_kernel_scalar<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), 0, n, world_size);
        }
    } else if (out.scalar_type() == at::ScalarType::Half) {
        Ptrs<__half> ptrs;
        bool aligned = (reinterpret_cast<uintptr_t>(out.data_ptr<at::Half>()) % 16 == 0);
        for (int i = 0; i < world_size; ++i) {
            ptrs.p[i] = reinterpret_cast<const __half*>(ptrs_int[i]);
            if (ptrs_int[i] % 16 != 0) aligned = false;
        }
        if (aligned) {
            int64_t n_vec = n / 8;
            int64_t n_tail = n % 8;
            if (n_vec > 0) {
                int blocks = (n_vec + threads - 1) / threads;
                reduce_fp16_kernel_vec8<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_vec, world_size);
            }
            if (n_tail > 0) {
                int blocks = (n_tail + threads - 1) / threads;
                reduce_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_vec * 8, n, world_size);
            }
        } else {
            int blocks = (n + threads - 1) / threads;
            reduce_fp16_kernel_scalar<<<blocks, threads, 0, stream>>>(ptrs, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), 0, n, world_size);
        }
    } else if (out.scalar_type() == at::ScalarType::Float) {
        Ptrs<float> ptrs;
        for (int i = 0; i < world_size; ++i) ptrs.p[i] = reinterpret_cast<const float*>(ptrs_int[i]);
        int blocks = (n + threads - 1) / threads;
        reduce_generic_kernel<float, float><<<blocks, threads, 0, stream>>>(ptrs, out.data_ptr<float>(), n, world_size);
    } else if (out.scalar_type() == at::ScalarType::Int) {
        Ptrs<int> ptrs;
        for (int i = 0; i < world_size; ++i) ptrs.p[i] = reinterpret_cast<const int*>(ptrs_int[i]);
        int blocks = (n + threads - 1) / threads;
        reduce_generic_kernel<int, int><<<blocks, threads, 0, stream>>>(ptrs, out.data_ptr<int>(), n, world_size);
    } else if (out.scalar_type() == at::ScalarType::Double) {
        Ptrs<double> ptrs;
        for (int i = 0; i < world_size; ++i) ptrs.p[i] = reinterpret_cast<const double*>(ptrs_int[i]);
        int blocks = (n + threads - 1) / threads;
        reduce_generic_kernel<double, double><<<blocks, threads, 0, stream>>>(ptrs, out.data_ptr<double>(), n, world_size);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_cuda", &reduce_cuda, "UVA reduce sum kernel over peers");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reduce_symm_mem_ext", CUDA_SRC)
    return _ext

_symm_cache = None

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] >= n and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"]
            
    # Over-allocate slightly to prevent frequent symmetric memory reallocation
    # on dynamically changing (but similar) sizes.
    alloc_n = max(n, 1024 * 1024)
    buf = symm_mem.empty(alloc_n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"n": alloc_n, "dtype": dtype, "device": device, "buf": buf, "hdl": hdl}
    return buf, hdl


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    rank = dist.get_rank()
    n = tensor.numel()
    
    if n == 0:
        return torch.empty_like(tensor) if rank == dst else tensor

    if rank == 0:
        _get_ext()
    dist.barrier()
    _get_ext()
    
    buf, hdl = _get_symm_state(n, tensor.dtype, tensor.device)
    
    # Fast D2D local copy into the symmetric memory buffer so peers can read it
    buf[:n].copy_(tensor.view(-1))
    
    # Barrier: guarantee everyone finishes their local copy before dst starts reading
    hdl.barrier(channel=0)
    
    if rank == dst:
        out = torch.empty_like(tensor)
        ptrs = [int(p) for p in hdl.buffer_ptrs]
        _get_ext().reduce_cuda(ptrs, out.view(-1), n)
    else:
        out = tensor
        
    # Barrier: ensure dst has completely finished reading before ranks continue.
    # This protects `buf` from being overwritten by subsequent iterations.
    hdl.barrier(channel=0)
    
    return out