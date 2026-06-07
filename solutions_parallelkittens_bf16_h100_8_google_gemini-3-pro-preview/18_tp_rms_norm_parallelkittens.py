import os
import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for fused TP RMSNorm with ThunderKittens all-reduce
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

using namespace kittens;

namespace tk_rms {

// ---------------------------------------------------------
// 1. Local Sum of Squares Kernel
// ---------------------------------------------------------
__global__ void local_sum_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ local_sums,
    int D, int N) 
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __bfloat162float(x[row * D + i]);
        sum += val * val;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (threadIdx.x == 0) {
        local_sums[row] = sum;
    }
}

// ---------------------------------------------------------
// 2. Apply RMSNorm Kernel
// ---------------------------------------------------------
__global__ void apply_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    const float* __restrict__ global_sums,
    float epsilon,
    int D, int N, int global_D) 
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    
    float global_sum = global_sums[row];
    float variance = global_sum / global_D;
    float rsqrt_var = rsqrtf(variance + epsilon);
    
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __bfloat162float(x[row * D + i]);
        float w = __bfloat162float(weight[i]);
        float out = val * rsqrt_var * w;
        y[row * D + i] = __float2bfloat16(out);
    }
}

// ---------------------------------------------------------
// 3. ThunderKittens Multimem All-Reduce and Barrier
// ---------------------------------------------------------
struct config_all_reduce {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 256;
};

struct globals_all_reduce {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 1;
    static constexpr int NUM_ELEMS_PER_BLOCK = config_all_reduce::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<float, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout tensor;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(tensor.numel() / NUM_ELEMS_PER_BLOCK / NUM_DEVICES);
    }
};

__device__ inline void kernel_all_reduce(const globals_all_reduce &G) {
    const size_t N_total = G.tensor.numel();
    const size_t N_per_dev = N_total / globals_all_reduce::NUM_DEVICES;
    const size_t idx = N_per_dev * G.dev_idx +
                       globals_all_reduce::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                       threadIdx.x;

    float tmp;
    multimem<float>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<float*>(&G.tensor.mc_ptr[idx]));
    multimem<float>::st(reinterpret_cast<float*>(&G.tensor.mc_ptr[idx]), tmp);
}

struct config_barrier {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals_barrier {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel_barrier(const globals_barrier &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace tk_rms

// ---------------------------------------------------------
// 4. Host Entrypoint
// ---------------------------------------------------------
void tk_fused_rms_norm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor y,
    kittens::py::TKParallelTensor &sums,
    kittens::py::TKParallelTensor &barrier,
    float epsilon,
    int global_D)
{
    int N = x.numel() / x.size(-1);
    int D = x.size(-1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 block(32, 8); // 8 warps, each processing a row
    dim3 grid((N + 7) / 8);

    // Step 1: Compute local sums over the hidden dim
    tk_rms::local_sum_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<float*>(sums.data_.data_ptr<float>()),
        D, N
    );

    // Setup ThunderKittens globals
    tk_rms::globals_barrier barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<tk_rms::globals_barrier::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    tk_rms::globals_all_reduce all_reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename tk_rms::globals_all_reduce::parallel_layout>(sums),
        .dev_idx = sums.local_rank_
    };

    // Step 2 & 3 & 4: In-place hardware multimem reduction wrapped with device-side barriers
    kittens::py::launch_kernel<tk_rms::config_barrier, tk_rms::globals_barrier, tk_rms::kernel_barrier>(barrier_G);
    kittens::py::launch_kernel<tk_rms::config_all_reduce, tk_rms::globals_all_reduce, tk_rms::kernel_all_reduce>(all_reduce_G);
    kittens::py::launch_kernel<tk_rms::config_barrier, tk_rms::globals_barrier, tk_rms::kernel_barrier>(barrier_G);

    // Step 5: Normalize and apply weight
    tk_rms::apply_rmsnorm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        reinterpret_cast<const float*>(sums.data_.data_ptr<float>()),
        epsilon, D, N, global_D
    );
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_rms_norm", &tk_fused_rms_norm);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False

# Layout constants ensuring correct padding for TK kernel
NUM_DEVICES = 8
NUM_ELEMS_PER_BLOCK = 256
ALIGNMENT = NUM_DEVICES * NUM_ELEMS_PER_BLOCK  # 2048


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_rmsnorm_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext


def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank()
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(
    local_hidden_states: torch.Tensor,
    local_weight: torch.Tensor,
    variance_epsilon: float
) -> torch.Tensor:
    
    assert local_hidden_states.is_cuda and local_hidden_states.is_contiguous()
    assert local_weight.is_cuda and local_weight.is_contiguous()
    assert local_hidden_states.dtype == torch.bfloat16, "Kernel optimized for BF16 hidden states"
    assert local_weight.dtype == torch.bfloat16, "Kernel optimized for BF16 weight"

    world = dist.get_world_size()
    assert world == NUM_DEVICES, f"This ThunderKittens kernel targets {NUM_DEVICES} devices, got {world}"

    ext = _ensure_ext_jit()

    original_shape = local_hidden_states.shape
    D = original_shape[-1]
    N = local_hidden_states.numel() // D

    # Pad symmetric buffer size so TK can safely divide workload amongst all SMs
    padded_N = ((N + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    x_flat = local_hidden_states.view(N, D)
    out_flat = torch.empty_like(x_flat)

    # Cached ThunderKittens arrays (VMM + NVSwitch multicast context bindings)
    sums_tk = get_or_create_parallel_tensor(ext, (padded_N,), torch.float32, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Initialize symmetric padding to 0
    sums_tk.data_.zero_()

    # Launch entire pipeline as one unified stream of C++ kernels
    ext.tk_fused_rms_norm(
        x_flat,
        local_weight,
        out_flat,
        sums_tk,
        barrier_tk,
        float(variance_epsilon),
        D * world
    )

    return out_flat.view(original_shape)