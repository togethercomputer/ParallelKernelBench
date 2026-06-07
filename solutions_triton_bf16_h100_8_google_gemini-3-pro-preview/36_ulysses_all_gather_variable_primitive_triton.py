import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional
import triton
import triton.language as tl
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void read_peer_shapes_kernel(
    const int64_t* ptrs,
    int64_t* out_shapes,
    int world_size,
    int max_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * max_dim;
    if (idx < total) {
        int rank = idx / max_dim;
        int dim = idx % max_dim;
        const int64_t* peer_ptr = reinterpret_cast<const int64_t*>(ptrs[rank]);
        out_shapes[idx] = peer_ptr[dim];
    }
}

void read_peer_shapes(
    torch::Tensor ptrs_tensor,
    torch::Tensor out_shapes,
    int max_dim
) {
    int world_size = ptrs_tensor.numel();
    int total = world_size * max_dim;
    int threads = 64;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    read_peer_shapes_kernel<<<blocks, threads, 0, stream>>>(
        ptrs_tensor.data_ptr<int64_t>(),
        out_shapes.data_ptr<int64_t>(),
        world_size,
        max_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor make_tensor_from_ptr(int64_t ptr, int64_t size, int element_size, int device_idx) {
    auto options = torch::TensorOptions().device(torch::Device(torch::kCUDA, device_idx));
    if (element_size == 2) options = options.dtype(torch::kInt16);
    else if (element_size == 4) options = options.dtype(torch::kInt32);
    else options = options.dtype(torch::kInt8);
    
    return torch::from_blob(reinterpret_cast<void*>(ptr), {size}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_peer_shapes", &read_peer_shapes, "Read peer shapes from symm mem");
    m.def("make_tensor_from_ptr", &make_tensor_from_ptr, "Create tensor from raw UVA pointer");
}
'''

_ext = None

def get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_gather_ext", CUDA_SRC)
    return _ext


@triton.jit
def ulysses_gather_kernel_generic(
    src_ptr, dst_ptr,
    gather_size, gather_offset, total_gather_size,
    outer_size, inner_size,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    inner_idx = offsets % inner_size
    tmp = offsets // inner_size
    local_gather_idx = tmp % gather_size
    outer_idx = tmp // gather_size
    
    dst_idx = outer_idx * (total_gather_size * inner_size) + \
              (local_gather_idx + gather_offset) * inner_size + \
              inner_idx
              
    src_data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_idx, src_data, mask=mask)


_symm_cache = {}

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
    ndim = x.dim()
    gather_dim = gather_dim % ndim
    x = x.contiguous()
    element_size = x.element_size()
    numel_bytes = x.numel() * element_size

    # Ensure extension is compiled once by rank 0 safely
    rank = dist.get_rank(group)
    if rank == 0:
        get_ext()
    dist.barrier(group=group)
    ext = get_ext()

    global _symm_cache
    if group not in _symm_cache:
        # Initial allocation - conservative large default (128MB per rank)
        max_req = torch.tensor([numel_bytes + 128], dtype=torch.int64, device=device)
        dist.all_reduce(max_req, op=dist.ReduceOp.MAX, group=group)
        alloc_bytes = max(max_req.item() * 2, 128 * 1024 * 1024)
        
        buf = symm_mem.empty(alloc_bytes, device=device, dtype=torch.uint8, group=group)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        _symm_cache[group] = {"buf": buf, "hdl": hdl, "ptrs": ptrs}

    cache = _symm_cache[group]

    # 1. Exchange Shapes
    # Write local shape requirements into the first 128 bytes (metadata section) of our symmetric buffer
    buf_i64 = cache["buf"][:128].view(torch.int64)
    x_shape_tensor = torch.tensor(x.size(), dtype=torch.int64, device=device)
    buf_i64[0] = ndim
    buf_i64[1:1+ndim].copy_(x_shape_tensor)
    
    # Wait for all peers to write their metadata
    cache["hdl"].barrier(channel=0)
    
    # Perform UVA device-to-device read to obtain peers' shapes
    out_shapes = torch.empty((world_size, 1 + ndim), dtype=torch.int64, device=device)
    ext.read_peer_shapes(cache["ptrs"], out_shapes, 1 + ndim)
    out_shapes_cpu = out_shapes.cpu().tolist() # CPU synchronization to calculate bounds
    
    # 2. Dynamic Reallocation Check
    max_req_bytes = 0
    gather_sizes = []
    for row in out_shapes_cpu:
        peer_ndim = row[0]
        peer_shape = row[1:1+peer_ndim]
        gather_sizes.append(peer_shape[gather_dim])
        
        peer_numel = 1
        for s in peer_shape:
            peer_numel *= s
        max_req_bytes = max(max_req_bytes, peer_numel * element_size)
        
    required_capacity = max_req_bytes + 128
    
    if required_capacity > cache["buf"].numel():
        # Because all ranks derived sizes identically from peers, this branch perfectly syncs organically
        alloc_bytes = max(required_capacity * 2, 256 * 1024 * 1024)
        buf = symm_mem.empty(alloc_bytes, device=device, dtype=torch.uint8, group=group)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        cache["buf"] = buf
        cache["hdl"] = hdl
        cache["ptrs"] = ptrs

    # 3. Data Transfer and Overlapped Compute Preparation
    # CPU continues setting up while GPU natively copies
    buf = cache["buf"]
    buf_data = buf[128:128+numel_bytes].view(dtype).view(x.shape)
    buf_data.copy_(x)
    
    cache["hdl"].barrier(channel=1) # Wait for all copies to land across the group
    
    total_gather_size = sum(gather_sizes)
    out_shape = list(x.size())
    out_shape[gather_dim] = total_gather_size
    out = torch.empty(out_shape, dtype=dtype, device=device)
    
    outer_size = 1
    for i in range(gather_dim):
        outer_size *= out_shape[i]
        
    inner_size = 1
    for i in range(gather_dim + 1, ndim):
        inner_size *= out_shape[i]
        
    src_ptrs_int = [int(ptr) + 128 for ptr in cache["hdl"].buffer_ptrs]
    dst_tensor_cast = out.view(torch.int16) if element_size == 2 else out.view(torch.int32)
    device_idx = device.index if device.index is not None else torch.cuda.current_device()

    # 4. Fused Triton P2P Pull 
    gather_offset = 0
    for r in range(world_size):
        g_size = gather_sizes[r]
        if g_size == 0:
            continue
            
        N_elements = outer_size * g_size * inner_size
        BLOCK_SIZE = 512
        grid = (triton.cdiv(N_elements, BLOCK_SIZE),)
        
        # Build standard pyTorch Tensor securely referencing the raw peer pointer 
        src_tensor = ext.make_tensor_from_ptr(src_ptrs_int[r], N_elements, element_size, device_idx)
        
        ulysses_gather_kernel_generic[grid](
            src_tensor, dst_tensor_cast,
            g_size, gather_offset, total_gather_size,
            outer_size, inner_size,
            N_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        gather_offset += g_size
        
    return out