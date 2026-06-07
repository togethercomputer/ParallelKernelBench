import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

struct PeerPtrs {
    uintptr_t ptrs[16];
};

template <typename T>
__global__ void uva_push_kernel(
    const T* __restrict__ local_data,
    PeerPtrs peer_ptrs,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < n; i += stride) {
        T val = __ldg(local_data + i);
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            if (p < world_size) {
                T* peer_ptr = reinterpret_cast<T*>(peer_ptrs.ptrs[p]);
                peer_ptr[i] = val;
            }
        }
    }
}

// Specialization for exactly 8 GPUs to maximize loop unrolling on Hopper
template <typename T>
__global__ void uva_push_kernel_8(
    const T* __restrict__ local_data,
    PeerPtrs peer_ptrs,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Load remote pointers into registers 
    T* p0 = reinterpret_cast<T*>(peer_ptrs.ptrs[0]);
    T* p1 = reinterpret_cast<T*>(peer_ptrs.ptrs[1]);
    T* p2 = reinterpret_cast<T*>(peer_ptrs.ptrs[2]);
    T* p3 = reinterpret_cast<T*>(peer_ptrs.ptrs[3]);
    T* p4 = reinterpret_cast<T*>(peer_ptrs.ptrs[4]);
    T* p5 = reinterpret_cast<T*>(peer_ptrs.ptrs[5]);
    T* p6 = reinterpret_cast<T*>(peer_ptrs.ptrs[6]);
    T* p7 = reinterpret_cast<T*>(peer_ptrs.ptrs[7]);

    for (int64_t i = idx; i < n; i += stride) {
        T val = __ldg(local_data + i); // Cache streaming read
        p0[i] = val;
        p1[i] = val;
        p2[i] = val;
        p3[i] = val;
        p4[i] = val;
        p5[i] = val;
        p6[i] = val;
        p7[i] = val;
    }
}

void uva_push(
    torch::Tensor local_tensor,
    std::vector<int64_t> peer_ptrs_vec,
    int64_t n_bytes
) {
    int world_size = peer_ptrs_vec.size();
    TORCH_CHECK(world_size <= 16, "Supports up to 16 GPUs on same NVLink domain");
    
    PeerPtrs peer_ptrs;
    for (int i = 0; i < world_size; ++i) {
        peer_ptrs.ptrs[i] = static_cast<uintptr_t>(peer_ptrs_vec[i]);
    }
    
    const int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Check global alignment across all peers for widest vectorized transaction
    uintptr_t align_mask = reinterpret_cast<uintptr_t>(local_tensor.data_ptr());
    for (int i = 0; i < world_size; ++i) {
        align_mask |= peer_ptrs.ptrs[i];
    }
    
    if (n_bytes % 16 == 0 && (align_mask % 16) == 0) {
        int64_t n = n_bytes / 16;
        const int blocks = std::min((int)((n + threads - 1) / threads), 65535);
        if (world_size == 8) {
            uva_push_kernel_8<uint4><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(local_tensor.data_ptr()), peer_ptrs, n
            );
        } else {
            uva_push_kernel<uint4><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(local_tensor.data_ptr()), peer_ptrs, world_size, n
            );
        }
    } else if (n_bytes % 8 == 0 && (align_mask % 8) == 0) {
        int64_t n = n_bytes / 8;
        const int blocks = std::min((int)((n + threads - 1) / threads), 65535);
        if (world_size == 8) {
            uva_push_kernel_8<uint2><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint2*>(local_tensor.data_ptr()), peer_ptrs, n
            );
        } else {
            uva_push_kernel<uint2><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint2*>(local_tensor.data_ptr()), peer_ptrs, world_size, n
            );
        }
    } else if (n_bytes % 4 == 0 && (align_mask % 4) == 0) {
        int64_t n = n_bytes / 4;
        const int blocks = std::min((int)((n + threads - 1) / threads), 65535);
        if (world_size == 8) {
            uva_push_kernel_8<uint32_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(local_tensor.data_ptr()), peer_ptrs, n
            );
        } else {
            uva_push_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(local_tensor.data_ptr()), peer_ptrs, world_size, n
            );
        }
    } else if (n_bytes % 2 == 0 && (align_mask % 2) == 0) {
        int64_t n = n_bytes / 2;
        const int blocks = std::min((int)((n + threads - 1) / threads), 65535);
        if (world_size == 8) {
            uva_push_kernel_8<uint16_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint16_t*>(local_tensor.data_ptr()), peer_ptrs, n
            );
        } else {
            uva_push_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint16_t*>(local_tensor.data_ptr()), peer_ptrs, world_size, n
            );
        }
    } else {
        int64_t n = n_bytes;
        const int blocks = std::min((int)((n + threads - 1) / threads), 65535);
        if (world_size == 8) {
            uva_push_kernel_8<uint8_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(local_tensor.data_ptr()), peer_ptrs, n
            );
        } else {
            uva_push_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(local_tensor.data_ptr()), peer_ptrs, world_size, n
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push", &uva_push, "UVA symmetric push broadcast for all_gather");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_push_gather_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(out_shape, dtype, device):
    """Retrieves and caches a symmetric rendezvous space for a given gathered shape."""
    key = (out_shape, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    out = symm_mem.empty(out_shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(out, dist.group.WORLD)
    _symm_cache[key] = (out, hdl)
    return out, hdl

@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor.unsqueeze(0).clone()
        
    rank = dist.get_rank()
    
    # Pre-load C++ extension strictly on rank 0 first to safely avoid compilation races
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    out_shape = (world_size,) + tensor.shape
    out, hdl = _get_symm_state(out_shape, tensor.dtype, tensor.device)
    
    # Barrier 0: Ensure all ranks have consumed previous cache values and are ready to be overwritten
    hdl.barrier(channel=0)
    
    numel_bytes = tensor.numel() * tensor.element_size()
    
    if tensor.numel() > 0:
        rank_offset = rank * numel_bytes
        # Map our payload offset to each rank's pre-rendered destination slice
        ptrs = [int(hdl.buffer_ptrs[i]) + rank_offset for i in range(world_size)]
        
        # Deploy parallel H100 direct NVLink pushes 
        _get_ext().uva_push(tensor, ptrs, numel_bytes)
    
    # Barrier 1: Guarantee everyone's payload stream has arrived in local RAM before proceeding
    hdl.barrier(channel=1)
    
    return out