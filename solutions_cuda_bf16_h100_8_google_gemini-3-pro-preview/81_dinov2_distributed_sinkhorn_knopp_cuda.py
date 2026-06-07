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

template <typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// Compute initial row sums and pack the scalar total_batch into the end of the buffer
template <typename T>
__global__ void compute_U_init_kernel(
    const T* __restrict__ T_mat,
    float* __restrict__ local_U,
    int B, int K, float tau_inv,
    const float* __restrict__ n_masked_patches
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        float sum = 0.0f;
        for (int j = 0; j < B; ++j) {
            float val = to_float(T_mat[j * K + i]);
            sum += expf(val * tau_inv);
        }
        local_U[i] = sum * (float)K;
    }
    // Thread 0 seamlessly packs the local masked patches for fused all-reduce
    if (i == 0 && n_masked_patches != nullptr) {
        local_U[K] = *n_masked_patches;
    }
}

// Compute column scales V using blocked warp reductions
template <typename T>
__global__ void compute_V_kernel(
    const T* __restrict__ T_mat,
    const float* __restrict__ U,
    float* __restrict__ V,
    int B, int K, float tau_inv, 
    const float* __restrict__ total_batch_ptr, int batch_offset
) {
    int j = blockIdx.x; 
    float total_batch = total_batch_ptr[batch_offset];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = to_float(T_mat[j * K + i]);
        sum += expf(val * tau_inv) / U[i];
    }
    
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) shared[warpId] = sum;
    __syncthreads();
    
    if (warpId == 0) {
        sum = (lane < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            V[j] = sum * total_batch;
        }
    }
}

// Update row scales U using coalesced global reads
template <typename T>
__global__ void compute_U_kernel(
    const T* __restrict__ T_mat,
    const float* __restrict__ V,
    float* __restrict__ local_U,
    int B, int K, float tau_inv
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        float sum = 0.0f;
        for (int j = 0; j < B; ++j) {
            float val = to_float(T_mat[j * K + i]);
            sum += expf(val * tau_inv) / V[j];
        }
        local_U[i] = sum * (float)K;
    }
}

// Resolve final matrix out-of-place 
template <typename T>
__global__ void compute_Final_kernel(
    const T* __restrict__ T_mat,
    const float* __restrict__ U,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int K, float tau_inv,
    const float* __restrict__ total_batch_ptr, int batch_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * K) {
        int j = idx / K;
        int i = idx % K;
        float val = to_float(T_mat[idx]);
        float total_batch = total_batch_ptr[batch_offset];
        Out[idx] = expf(val * tau_inv) * total_batch / (U[i] * V[j]);
    }
}

// ---------------------------------------------------------------------------
// Acquire-Release Device Barrier Logic
// ---------------------------------------------------------------------------
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
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) {
        return;
    }
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__global__ void allreduce_f32_kernel(
    const uint64_t* __restrict__ ptrs,
    float* __restrict__ out,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t n,
    int world_size,
    int rank,
    int64_t iteration_offset
) {
    const uint64_t base_block_id = static_cast<uint64_t>(blockIdx.x) + iteration_offset;
    
    __syncthreads(); // Ensure peers sync locally
    blockwise_barrier(signal_pad_ptrs, base_block_id, rank, world_size);
    __syncthreads(); // Wait for barrier subset to unlock blocks

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const float* src = reinterpret_cast<const float*>(ptrs[r]);
            sum += src[idx];
        }
        out[idx] = sum;
    }

    // Fence to avoid peer buffer overwrite before collective read completes
    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, base_block_id + gridDim.x, rank, world_size);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_compute_U_init(
    torch::Tensor T_mat, torch::Tensor local_U,
    int B, int K, float tau_inv, torch::Tensor n_masked_patches
) {
    CHECK_INPUT(T_mat); CHECK_INPUT(local_U); CHECK_INPUT(n_masked_patches);
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float* n_ptr = n_masked_patches.data_ptr<float>();

    if (T_mat.dtype() == torch::kBFloat16) {
        compute_U_init_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(T_mat.data_ptr<at::BFloat16>()),
            local_U.data_ptr<float>(), B, K, tau_inv, n_ptr);
    } else {
        compute_U_init_kernel<<<blocks, threads, 0, stream>>>(
            T_mat.data_ptr<float>(), local_U.data_ptr<float>(), B, K, tau_inv, n_ptr);
    }
}

void launch_compute_V(
    torch::Tensor T_mat, torch::Tensor U, torch::Tensor V,
    int B, int K, float tau_inv, torch::Tensor total_batch_tensor, int batch_offset
) {
    int threads = 256;
    int blocks = B; // Perfect mapping row to block
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (T_mat.dtype() == torch::kBFloat16) {
        compute_V_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(T_mat.data_ptr<at::BFloat16>()),
            U.data_ptr<float>(), V.data_ptr<float>(), B, K, tau_inv,
            total_batch_tensor.data_ptr<float>(), batch_offset);
    } else {
        compute_V_kernel<<<blocks, threads, 0, stream>>>(
            T_mat.data_ptr<float>(), U.data_ptr<float>(), V.data_ptr<float>(),
            B, K, tau_inv, total_batch_tensor.data_ptr<float>(), batch_offset);
    }
}

void launch_compute_U(
    torch::Tensor T_mat, torch::Tensor V, torch::Tensor local_U,
    int B, int K, float tau_inv
) {
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (T_mat.dtype() == torch::kBFloat16) {
        compute_U_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(T_mat.data_ptr<at::BFloat16>()),
            V.data_ptr<float>(), local_U.data_ptr<float>(), B, K, tau_inv);
    } else {
        compute_U_kernel<<<blocks, threads, 0, stream>>>(
            T_mat.data_ptr<float>(), V.data_ptr<float>(), local_U.data_ptr<float>(),
            B, K, tau_inv);
    }
}

void launch_compute_Final(
    torch::Tensor T_mat, torch::Tensor U, torch::Tensor V, torch::Tensor Out,
    int B, int K, float tau_inv, torch::Tensor total_batch_tensor, int batch_offset
) {
    int threads = 256;
    int blocks = (B * K + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (T_mat.dtype() == torch::kBFloat16) {
        compute_Final_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(T_mat.data_ptr<at::BFloat16>()),
            U.data_ptr<float>(), V.data_ptr<float>(), Out.data_ptr<float>(),
            B, K, tau_inv, total_batch_tensor.data_ptr<float>(), batch_offset);
    } else {
        compute_Final_kernel<<<blocks, threads, 0, stream>>>(
            T_mat.data_ptr<float>(), U.data_ptr<float>(), V.data_ptr<float>(), Out.data_ptr<float>(),
            B, K, tau_inv, total_batch_tensor.data_ptr<float>(), batch_offset);
    }
}

void launch_allreduce_f32(
    torch::Tensor ptrs_tensor, torch::Tensor out, torch::Tensor signal_pad_ptrs_tensor,
    int64_t n, int world_size, int rank, int64_t iteration_offset
) {
    int threads = 256;
    int blocks = std::min(1024, (int)((n + threads - 1) / threads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(ptrs_tensor.data_ptr<int64_t>());
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    
    allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, out.data_ptr<float>(), d_signal, n, world_size, rank, iteration_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_compute_U_init", &launch_compute_U_init);
    m.def("launch_compute_V", &launch_compute_V);
    m.def("launch_compute_U", &launch_compute_U);
    m.def("launch_compute_Final", &launch_compute_Final);
    m.def("launch_allreduce_f32", &launch_allreduce_f32);
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_sinkhorn_knopp_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (n, dtype, device, group)
    if key in _resource_cache:
        return _resource_cache[key]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)

    out = torch.empty(n, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_masked_patches_tensor: torch.Tensor,
    n_iterations: int = 3,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    # Input invariants
    B, K = teacher_output.shape
    tau_inv = 1.0 / teacher_temp
    device = teacher_output.device
    ext = _get_ext()
    teacher_output = teacher_output.contiguous()

    if n_masked_patches_tensor.dtype != torch.float32:
        n_masked_patches_tensor = n_masked_patches_tensor.float()

    # K + 1 reserves a slot for parallel scalar reduction of batch patches in the identical buffer
    symm_buf, hdl, global_U, ptrs_tensor = _get_resources(K + 1, torch.float32, device, group)

    V = torch.empty(B, dtype=torch.float32, device=device)
    out = torch.empty((B, K), dtype=torch.float32, device=device)

    # Sync block execution before altering local mapped peers
    dist.barrier(group=group)

    ext.launch_compute_U_init(
        teacher_output, symm_buf, B, K, tau_inv, n_masked_patches_tensor
    )

    barrier_id = 0
    threads = 256
    blocks_allreduce_K1 = min(1024, (K + 1 + threads - 1) // threads)

    # Coalesced collective communication mapping the global mass
    ext.launch_allreduce_f32(
        ptrs_tensor, global_U, hdl.signal_pad_ptrs_dev, K + 1,
        world_size, rank, barrier_id
    )
    barrier_id += 2 * blocks_allreduce_K1
    blocks_allreduce_K = min(1024, (K + threads - 1) // threads)

    # Unrolled analytic matrix iterations (updating projection vectors inplace)
    for _ in range(n_iterations - 1):
        ext.launch_compute_V(
            teacher_output, global_U, V, B, K, tau_inv, global_U, K
        )
        ext.launch_compute_U(
            teacher_output, V, symm_buf, B, K, tau_inv
        )
        ext.launch_allreduce_f32(
            ptrs_tensor, global_U, hdl.signal_pad_ptrs_dev, K,
            world_size, rank, barrier_id
        )
        barrier_id += 2 * blocks_allreduce_K

    ext.launch_compute_V(
        teacher_output, global_U, V, B, K, tau_inv, global_U, K
    )

    # Final matrix assembly mapped back to transposed layout
    ext.launch_compute_Final(
        teacher_output, global_U, V, out, B, K, tau_inv, global_U, K
    )

    return out