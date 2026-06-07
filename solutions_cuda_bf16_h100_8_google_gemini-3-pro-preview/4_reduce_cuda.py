import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/core/Dispatch.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Hopper NVSwitch Multimem Reduce (dst rank only)
// ---------------------------------------------------------------------------
__global__ void multimem_reduce_bf16_kernel(
    uint64_t multicast_base,
    __nv_bfloat16* __restrict__ out,
    int64_t numel_128
) {
    const int num_programs = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < numel_128; idx += num_programs) {
        uint64_t* ptr = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        
        // Fetch and reduce 128-bits (8 x bfloat16) across all ranks in hardware
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
            : "l"(ptr)
            : "memory");
        
        // 128-bit vectorized store to the destination buffer
        uint4* out_ptr = reinterpret_cast<uint4*>(out) + idx;
        *out_ptr = make_uint4(x, y, z, w);
    }
}

// ---------------------------------------------------------------------------
// UVA Peer-Pointer Fallback Reduce (dst rank only)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void reduce_generic_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const T* src = reinterpret_cast<const T*>(ptrs[r]);
            sum += static_cast<float>(src[idx]);
        }
        out[idx] = static_cast<T>(sum);
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
void launch_multimem_reduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor out,
    int64_t numel_128
) {
    if (numel_128 == 0) return;
    int threads = 512;
    int blocks = (numel_128 + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_reduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        multicast_ptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        numel_128
    );
}

void launch_reduce(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n
) {
    if (n == 0) return;
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());

    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, out.scalar_type(), "reduce_kernel", ([&] {
        reduce_generic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            d_ptrs,
            out.data_ptr<scalar_t>(),
            world_size,
            n
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_reduce_bf16", &launch_multimem_reduce_bf16, "Multimem hardware reduce to dest");
    m.def("launch_reduce", &launch_reduce, "Custom P2P UVA reduce fallback");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reduce_cuda_opt_ext", CUDA_SRC)
    return _ext

_resource_cache = {}
def _get_resources(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    n = math.prod(shape)
    
    # Pad allocations to multiples of 8 elements for pure 128-bit vectorization in BF16
    pad_n = (n + 7) & ~7 if dtype == torch.bfloat16 else n

    buf = symm_mem.empty(pad_n, device=device, dtype=dtype)
    buf.zero_()  # Zero padding elements to safely accumulate +0.0 during multimem tail fetches
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out_pad = torch.empty(pad_n, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, out_pad, ptrs_tensor)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    """
    Optimized device-side collective to replace dist.reduce().
    Favors NVSwitch multimem on Hopper or optimized peer UVA reads.
    """
    if not dist.is_initialized():
        return tensor.clone()

    input_tensor = tensor.contiguous()
    n = input_tensor.numel()
    dtype = input_tensor.dtype
    device = input_tensor.device
    rank = dist.get_rank()

    # Pre-registered resources avoiding allocation on the hot path
    buf, hdl, out_pad, ptrs_tensor = _get_resources(input_tensor.shape, dtype, device)
    
    # 1. Fill registered symmetric buffer (tail elements safely untouched and remain zeroed)
    buf[:n].copy_(input_tensor.flatten())
    
    # 2. Synchronize all ranks before destination reads 
    hdl.barrier(channel=0)

    # 3. Pull and reduce via Switch / NVLink (Executed solely on dst rank)
    multicast_ptr = getattr(hdl, 'multicast_ptr', 0)
    
    if multicast_ptr != 0 and dtype == torch.bfloat16:
        if rank == dst:
            numel_128 = out_pad.numel() // 8
            _get_ext().launch_multimem_reduce_bf16(multicast_ptr, out_pad, numel_128)
    else:
        if rank == dst:
            _get_ext().launch_reduce(ptrs_tensor, out_pad, n)
            
    # 4. Enforce buffer lifespan: ensure dst completes reads before next collective overwrites buf
    hdl.barrier(channel=0)

    # 5. Result isolation
    if rank == dst:
        return out_pad[:n].reshape(input_tensor.shape).clone()
    else:
        return input_tensor.clone()