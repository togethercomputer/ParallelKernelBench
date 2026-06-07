import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

__global__ void broadcast_multimem_kernel_padded(
    const void* __restrict__ src_ptr,
    uint64_t multicast_base,
    int64_t n_padded_bytes
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int64_t n_vec = n_padded_bytes / 16;
    const uint4* src_vec = reinterpret_cast<const uint4*>(src_ptr);

    for (int64_t i = idx; i < n_vec; i += stride) {
        uint4 val = src_vec[i];
        uint64_t* mc_addr = reinterpret_cast<uint64_t*>(multicast_base) + i * 2;
        asm volatile(
            "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(mc_addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
            : "memory");
    }
}

__global__ void pull_broadcast_kernel_padded(
    const void* __restrict__ src_ptr,
    void* __restrict__ dst_ptr,
    int64_t n_padded_bytes
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int64_t n_vec = n_padded_bytes / 16;
    const uint4* src_vec = reinterpret_cast<const uint4*>(src_ptr);
    uint4* dst_vec = reinterpret_cast<uint4*>(dst_ptr);

    for (int64_t i = idx; i < n_vec; i += stride) {
        dst_vec[i] = src_vec[i];
    }
}

void launch_broadcast_multimem(
    int64_t src_ptr,
    int64_t multicast_ptr,
    int64_t n_padded_bytes
) {
    int threads = 512;
    int blocks = std::min((int)((n_padded_bytes / 16 + threads - 1) / threads), 1024);
    if (blocks == 0) blocks = 1;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    broadcast_multimem_kernel_padded<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const void*>(static_cast<uintptr_t>(src_ptr)),
        static_cast<uint64_t>(multicast_ptr),
        n_padded_bytes
    );
}

void launch_pull_broadcast(
    int64_t src_ptr,
    int64_t dst_ptr,
    int64_t n_padded_bytes
) {
    int threads = 512;
    int blocks = std::min((int)((n_padded_bytes / 16 + threads - 1) / threads), 1024);
    if (blocks == 0) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pull_broadcast_kernel_padded<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const void*>(static_cast<uintptr_t>(src_ptr)),
        reinterpret_cast<void*>(static_cast<uintptr_t>(dst_ptr)),
        n_padded_bytes
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_broadcast_multimem", &launch_broadcast_multimem, "Multimem broadcast kernel");
    m.def("launch_pull_broadcast", &launch_pull_broadcast, "Pull broadcast kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("broadcast_multimem_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_state(n_bytes: int, dtype: torch.dtype, device: torch.device):
    key = (n_bytes, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty((n_bytes,), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Symmetric memory Multimem broadcast. Replaces NCCL broadcast with custom 
    multimem.st PTX broadcast (Hopper) or fast UVA pull kernel.
    """
    if not dist.is_initialized():
        return tensor.clone()

    rank = dist.get_rank()
    n_bytes = tensor.numel() * tensor.element_size()
    
    if n_bytes == 0:
        return tensor.clone() if rank == src else torch.empty_like(tensor)

    # Pad buffer size to next 16-byte multiple to allow 100% vectorized 128-bit memory ops
    padded_bytes = (n_bytes + 15) // 16 * 16

    # JIT compile safely
    if rank == 0:
        _get_ext()
    dist.barrier()
    _get_ext()

    buf, hdl = _get_symm_state(padded_bytes, torch.uint8, tensor.device)

    # 1. Sync: ensure previous operations on the cached `buf` are globally done.
    hdl.barrier(channel=0)

    # 2. Source rank stages the local payload into its symmetric, 16-byte aligned buffer.
    if rank == src:
        buf_view = buf[:n_bytes].view(tensor.dtype).view(tensor.shape)
        buf_view.copy_(tensor)

    # 3. Broadcast data directly over symmetric mappings.
    if hdl.multicast_ptr:
        if rank == src:
            _get_ext().launch_broadcast_multimem(
                buf.data_ptr(),
                hdl.multicast_ptr,
                padded_bytes
            )
        # Device sync: Ensure NVSwitch multimem stores land globally across ranks.
        hdl.barrier(channel=0)
    else:
        # Fallback device sync: Ensure src's initial staging memory-copy is globally visible.
        hdl.barrier(channel=0)
        
        if rank != src:
            src_buf_ptr = int(hdl.buffer_ptrs[src])
            _get_ext().launch_pull_broadcast(
                src_buf_ptr,
                buf.data_ptr(),
                padded_bytes
            )
        
        # Device sync: Ensure pull kernels complete on receivers.
        hdl.barrier(channel=0)

    # 4. Expose the populated data out.
    if rank == src:
        out = tensor.clone()
    else:
        out = torch.empty_like(tensor)
        out_view = buf[:n_bytes].view(tensor.dtype).view(tensor.shape)
        out.copy_(out_view)

    return out