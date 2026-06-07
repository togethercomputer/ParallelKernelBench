import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define MAX_WS 32

struct KernelArgs {
    int64_t L_array[MAX_WS];
    int64_t dst_offset[MAX_WS];
    int64_t total_prefix[MAX_WS];
};

template<typename T>
__global__ void ulysses_allgather_kernel(
    const int64_t* __restrict__ data_ptrs, 
    KernelArgs args,
    T* __restrict__ out,
    int64_t sum_BC,                        
    int64_t total_elements,                
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < total_elements; i += (int64_t)gridDim.x * blockDim.x) {
        int j = 0;
        // Small loop to resolve the target rank. total_prefix tracks the flattened boundaries.
        while (j < world_size - 1 && i >= args.total_prefix[j + 1]) {
            j++;
        }

        int64_t local_i = i - args.total_prefix[j];
        int64_t L_j = args.L_array[j];
        
        int64_t a = local_i / L_j;
        int64_t k = local_i % L_j;

        int64_t src_idx = a * L_j + k;
        int64_t dst_idx = a * sum_BC + args.dst_offset[j] + k;

        const T* src = reinterpret_cast<const T*>(data_ptrs[j]);
        out[dst_idx] = src[src_idx];
    }
}

__global__ void gather_shapes_kernel(
    const int64_t* __restrict__ shape_ptrs,
    int64_t* __restrict__ gathered_shapes,
    int world_size
) {
    int rank = blockIdx.x;
    int idx = threadIdx.x;
    if (rank < world_size && idx < 32) {
        const int64_t* src = reinterpret_cast<const int64_t*>(shape_ptrs[rank]);
        gathered_shapes[rank * 32 + idx] = src[idx];
    }
}

void launch_gather_shapes(
    torch::Tensor shape_ptrs,
    torch::Tensor gathered_shapes,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_shapes_kernel<<<world_size, 32, 0, stream>>>(
        shape_ptrs.data_ptr<int64_t>(),
        gathered_shapes.data_ptr<int64_t>(),
        world_size
    );
}

void launch_ulysses_allgather(
    torch::Tensor data_ptrs,
    std::vector<int64_t> L_array,
    std::vector<int64_t> dst_offset,
    std::vector<int64_t> total_prefix,
    torch::Tensor out,
    int64_t sum_BC,
    int64_t total_elements,
    int world_size,
    int vector_bytes
) {
    KernelArgs args;
    for (int i = 0; i < world_size; ++i) {
        args.L_array[i] = L_array[i];
        args.dst_offset[i] = dst_offset[i];
        args.total_prefix[i] = total_prefix[i];
    }
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t* d_ptrs = data_ptrs.data_ptr<int64_t>();

    // Dynamically dispatch strictly on maximum achievable alignment bandwidth
    if (vector_bytes == 16) {
        ulysses_allgather_kernel<uint4><<<blocks, threads, 0, stream>>>(
            d_ptrs, args, reinterpret_cast<uint4*>(out.data_ptr()), sum_BC, total_elements, world_size);
    } else if (vector_bytes == 8) {
        ulysses_allgather_kernel<uint2><<<blocks, threads, 0, stream>>>(
            d_ptrs, args, reinterpret_cast<uint2*>(out.data_ptr()), sum_BC, total_elements, world_size);
    } else if (vector_bytes == 4) {
        ulysses_allgather_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, args, reinterpret_cast<uint32_t*>(out.data_ptr()), sum_BC, total_elements, world_size);
    } else if (vector_bytes == 2) {
        ulysses_allgather_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, args, reinterpret_cast<uint16_t*>(out.data_ptr()), sum_BC, total_elements, world_size);
    } else {
        ulysses_allgather_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, args, reinterpret_cast<uint8_t*>(out.data_ptr()), sum_BC, total_elements, world_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_shapes", &launch_gather_shapes, "Gather shape info via UVA");
    m.def("launch_ulysses_allgather", &launch_ulysses_allgather, "Ulysses variable allgather custom kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_allgather_var_ext", CUDA_SRC)
    return _ext


class SymmCache:
    def __init__(self, world_size: int, device: torch.device, dtype: torch.dtype, group: dist.ProcessGroup):
        self.world_size = world_size
        self.device = device
        self.dtype = dtype
        self.group = group
        
        # 32-element buffer allows exchanging up to ~30D tensor shapes.
        self.shape_buf = symm_mem.empty(32, dtype=torch.int64, device=device)
        self.shape_hdl = symm_mem.rendezvous(self.shape_buf, group)
        self.shape_ptrs_dev = torch.tensor(self.shape_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
        self.gathered_shapes_dev = torch.empty((world_size, 32), dtype=torch.int64, device=device)
        self.gathered_shapes_host = torch.empty((world_size, 32), dtype=torch.int64, pin_memory=True)
        self.local_shape_host = torch.empty(32, dtype=torch.int64, pin_memory=True)
        
        # 1024 elements default fallback size; lazily expands when size spikes.
        self.data_capacities = [1024] * world_size
        self.data_buf = symm_mem.empty(1024, dtype=dtype, device=device)
        self.data_hdl = symm_mem.rendezvous(self.data_buf, group)
        self.data_ptrs_dev = torch.tensor(self.data_hdl.buffer_ptrs, dtype=torch.int64, device=device)

_cache_dict = {}

def _get_cache(group: dist.ProcessGroup, device: torch.device, dtype: torch.dtype) -> SymmCache:
    key = (group, dtype)
    if key not in _cache_dict:
        _cache_dict[key] = SymmCache(dist.get_world_size(group), device, dtype, group)
    return _cache_dict[key]


@torch.no_grad()
def solution(
    x: torch.Tensor,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()
        
    device = x.device
    dtype = x.dtype
    x = x.contiguous()
    x_dim = x.dim()
    
    # Ensure correct boundary for reverse-indexing formats
    gather_dim = gather_dim % x_dim
    assert x_dim <= 30, "Tensor dimensions exceed shape buffer capacity limits"
    
    cache = _get_cache(group, device, dtype)
    rank = dist.get_rank(group)
    
    # 1. Start shape exchange without synchronizing host
    cache.local_shape_host[0] = x_dim
    for i, s in enumerate(x.shape):
        cache.local_shape_host[i+1] = s
    cache.local_shape_host[31] = x.numel()
    
    cache.shape_buf.copy_(cache.local_shape_host, non_blocking=True)
    
    # 2. Overlap payload copy natively via async queue while shape barrier resolves
    optimistic_copy_done = False
    if x.numel() <= cache.data_buf.numel():
        cache.data_buf[:x.numel()].copy_(x.view(-1), non_blocking=True)
        optimistic_copy_done = True
        
    cache.shape_hdl.barrier(channel=0)
    
    # 3. Harvest device configurations over peer UVA
    ext = _get_ext()
    ext.launch_gather_shapes(cache.shape_ptrs_dev, cache.gathered_shapes_dev, cache.world_size)
    cache.gathered_shapes_host.copy_(cache.gathered_shapes_dev, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    
    # 4. Resolve exact concatenated target configuration and routing
    sum_B = 0
    B_array = []
    max_capacity_needed = [0] * cache.world_size
    
    for i in range(cache.world_size):
        B_i = cache.gathered_shapes_host[i, 1 + gather_dim].item()
        B_array.append(B_i)
        sum_B += B_i
        max_capacity_needed[i] = cache.gathered_shapes_host[i, 31].item()
        
    needs_realloc = False
    for i in range(cache.world_size):
        if max_capacity_needed[i] > cache.data_capacities[i]:
            needs_realloc = True
            cache.data_capacities[i] = int(max_capacity_needed[i] * 1.2)  # Maintain stable symmetric arrays 
            
    # Re-rendezvous path natively isolated only for rare size-spike spikes
    if needs_realloc:
        reallocated = False
        if cache.data_capacities[rank] > cache.data_buf.numel():
            cache.data_buf = symm_mem.empty(cache.data_capacities[rank], dtype=dtype, device=device)
            reallocated = True
            
        cache.data_hdl = symm_mem.rendezvous(cache.data_buf, group)
        cache.data_ptrs_dev = torch.tensor(cache.data_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
        if not optimistic_copy_done or reallocated:
            cache.data_buf[:x.numel()].copy_(x.view(-1), non_blocking=True)
            
    cache.data_hdl.barrier(channel=0)
    
    # 5. Output structure formulation
    out_shape = list(x.shape)
    out_shape[gather_dim] = sum_B
    out = torch.empty(out_shape, dtype=dtype, device=device)
    
    A = 1
    for s in out_shape[:gather_dim]:
        A *= s
    C = 1
    for s in out_shape[gather_dim+1:]:
        C *= s
        
    B_prefix = [0] * cache.world_size
    total_prefix = [0] * cache.world_size
    L_array = [0] * cache.world_size
    dst_offset = [0] * cache.world_size
    
    prefix_b = 0
    prefix_total = 0
    for i in range(cache.world_size):
        B_prefix[i] = prefix_b
        total_prefix[i] = prefix_total
        
        L_i = B_array[i] * C
        L_array[i] = L_i
        dst_offset[i] = prefix_b * C
        
        prefix_b += B_array[i]
        prefix_total += A * L_i
        
    total_elements = prefix_total
    if total_elements == 0:
        return out
        
    # 6. Automatic alignment reduction scaling factor validation
    element_size = x.element_size()
    max_vf = 16 // element_size
    vfs = [max_vf]
    while vfs[-1] > 1:
        vfs.append(vfs[-1] // 2)
        
    VF = 1
    for vf in vfs:
        if all((b * C) % vf == 0 for b in B_array):
            VF = vf
            break
            
    L_array_vf = [l // VF for l in L_array]
    dst_offset_vf = [d // VF for d in dst_offset]
    total_prefix_vf = [t // VF for t in total_prefix]
    sum_BC_vf = (sum_B * C) // VF
    total_elements_vf = total_elements // VF
    
    ext.launch_ulysses_allgather(
        cache.data_ptrs_dev,
        L_array_vf,
        dst_offset_vf,
        total_prefix_vf,
        out,
        sum_BC_vf,
        total_elements_vf,
        cache.world_size,
        VF * element_size
    )
    
    return out