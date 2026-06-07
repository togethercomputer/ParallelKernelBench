"""
Optimized MoE EP all_to_all_single using symmetric memory and UVA pull kernels.
"""

from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

struct PtrArray {
    const void* ptrs[32];
};

struct IntArray {
    int64_t data[32];
};

__global__ void set_ctrl_buf_kernel(
    int64_t* ctrl_buf,
    IntArray input_offsets,
    int world_size
) {
    if (threadIdx.x < world_size) {
        ctrl_buf[threadIdx.x] = input_offsets.data[threadIdx.x];
    }
}

__global__ void pull_kernel(
    const PtrArray ctrl_ptrs,
    const PtrArray data_ptrs,
    const IntArray out_offsets,
    const IntArray out_sizes,
    nv_bfloat16* out_data,
    int hidden_dim,
    int world_size,
    int rank
) {
    int peer = blockIdx.y; 
    int64_t size_tokens = out_sizes.data[peer];
    if (size_tokens == 0) return;

    int64_t out_offset_tokens = out_offsets.data[peer];
    
    // Read the remote offset from the peer's control buffer
    const int64_t* peer_ctrl = reinterpret_cast<const int64_t*>(ctrl_ptrs.ptrs[peer]);
    int64_t remote_token_offset = peer_ctrl[rank];
    
    const nv_bfloat16* peer_data = reinterpret_cast<const nv_bfloat16*>(data_ptrs.ptrs[peer]);
    
    int64_t num_elements = size_tokens * hidden_dim;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    int64_t out_base = out_offset_tokens * hidden_dim;
    int64_t in_base = remote_token_offset * hidden_dim;
    
    // 128-bit vectorized pull over NVLink
    int64_t num_f4 = num_elements / 8;
    if ((out_base % 8 == 0) && (in_base % 8 == 0) && 
        ((reinterpret_cast<uintptr_t>(out_data) % 16) == 0) &&
        ((reinterpret_cast<uintptr_t>(peer_data) % 16) == 0)) {
        
        const float4* peer_data_f4 = reinterpret_cast<const float4*>(peer_data + in_base);
        float4* out_data_f4 = reinterpret_cast<float4*>(out_data + out_base);
        
        for (int64_t i = tid; i < num_f4; i += stride) {
            out_data_f4[i] = peer_data_f4[i];
        }
        
        int64_t rem_start = num_f4 * 8;
        for (int64_t i = rem_start + tid; i < num_elements; i += stride) {
            out_data[out_base + i] = peer_data[in_base + i];
        }
    } else {
        // Fallback for non-aligned buffers (rarely hit for hidden_dim % 8 == 0)
        for (int64_t i = tid; i < num_elements; i += stride) {
            out_data[out_base + i] = peer_data[in_base + i];
        }
    }
}

void set_ctrl_buf(
    torch::Tensor ctrl_buf,
    std::vector<int64_t> input_offsets,
    int world_size
) {
    IntArray arr;
    for(int i = 0; i < world_size; ++i) {
        arr.data[i] = input_offsets[i];
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    set_ctrl_buf_kernel<<<1, 32, 0, stream>>>(
        ctrl_buf.data_ptr<int64_t>(), arr, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pull_data(
    std::vector<int64_t> ctrl_ptrs_in,
    std::vector<int64_t> data_ptrs_in,
    std::vector<int64_t> out_offsets_in,
    std::vector<int64_t> out_sizes_in,
    torch::Tensor out_data,
    int hidden_dim,
    int world_size,
    int rank
) {
    PtrArray ctrl_ptrs;
    PtrArray data_ptrs;
    IntArray out_offsets;
    IntArray out_sizes;
    
    for(int i = 0; i < world_size; ++i) {
        ctrl_ptrs.ptrs[i] = reinterpret_cast<const void*>(ctrl_ptrs_in[i]);
        data_ptrs.ptrs[i] = reinterpret_cast<const void*>(data_ptrs_in[i]);
        out_offsets.data[i] = out_offsets_in[i];
        out_sizes.data[i] = out_sizes_in[i];
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Aggressive parallelization per peer channel
    int blocks_per_peer = 256;
    dim3 grid(blocks_per_peer, world_size);
    dim3 block(256);
    
    pull_kernel<<<grid, block, 0, stream>>>(
        ctrl_ptrs, data_ptrs, out_offsets, out_sizes,
        reinterpret_cast<nv_bfloat16*>(out_data.data_ptr<at::BFloat16>()),
        hidden_dim, world_size, rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_ctrl_buf", &set_ctrl_buf, "Set symmetric control buffer async");
    m.def("pull_data", &pull_data, "Pull data directly from peers via UVA");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_moe_all2all_pull", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(max_elements: int, world_size: int, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    if group in _symm_cache:
        return _symm_cache[group]

    # Pre-allocate large buffers to avoid rendezvous on hot path
    data_buf = symm_mem.empty(max_elements, device=device, dtype=torch.bfloat16)
    data_hdl = symm_mem.rendezvous(data_buf, group)
    
    ctrl_buf = symm_mem.empty(world_size, device=device, dtype=torch.int64)
    ctrl_hdl = symm_mem.rendezvous(ctrl_buf, group)
    
    _symm_cache[group] = (data_buf, data_hdl, ctrl_buf, ctrl_hdl)
    return _symm_cache[group]


def solution(
    local_tensor: torch.Tensor,
    input_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    output_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return local_tensor.contiguous()

    local_tensor = local_tensor.contiguous()
    assert local_tensor.dtype == torch.bfloat16, "Kernel optimized strictly for BF16 precision"
    assert world_size <= 32, "Kernel expects a maximum world size of 32"

    rank = dist.get_rank(group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group)
    ext = _get_ext()

    # Parse and synchronize input split sizes
    if input_split_sizes is None:
        chunk_size = local_tensor.size(0) // world_size
        in_sizes = [chunk_size] * world_size
    elif isinstance(input_split_sizes, torch.Tensor):
        in_sizes = input_split_sizes.tolist()
    else:
        in_sizes = input_split_sizes

    # Parse and synchronize output split sizes
    if output_split_sizes is None:
        chunk_size = local_tensor.size(0) // world_size
        out_sizes = [chunk_size] * world_size
    elif isinstance(output_split_sizes, torch.Tensor):
        out_sizes = output_split_sizes.tolist()
    else:
        out_sizes = output_split_sizes

    # Prefix sums (offsets)
    in_offsets = [0] * world_size
    curr = 0
    for i, s in enumerate(in_sizes):
        in_offsets[i] = curr
        curr += s

    out_offsets = [0] * world_size
    curr = 0
    for i, s in enumerate(out_sizes):
        out_offsets[i] = curr
        curr += s

    out_size_total = sum(out_sizes)
    hidden_dim = local_tensor.size(1)

    output = torch.empty(
        (out_size_total, hidden_dim),
        dtype=local_tensor.dtype,
        device=local_tensor.device,
    )

    # Allow up to 128 million bfloat16 elements buffered (~256 MB) to prevent reallocation
    MAX_DATA_ELEMENTS = 128 * 1024 * 1024
    numel = local_tensor.numel()
    if numel > MAX_DATA_ELEMENTS:
        raise RuntimeError(f"Local tensor {numel} exceeds buffer capacity of {MAX_DATA_ELEMENTS}")

    data_buf, data_hdl, ctrl_buf, ctrl_hdl = _get_symm_state(MAX_DATA_ELEMENTS, world_size, local_tensor.device, group)

    # 1. Publish our payload into device symmetric memory
    if numel > 0:
        data_buf[:numel].copy_(local_tensor.view(-1))
    
    # 2. Write outgoing size offsets into control symmetric memory
    ext.set_ctrl_buf(ctrl_buf, in_offsets, world_size)
    
    # 3. Synchronize. Ensure all peers have cleanly published data and control offsets
    data_hdl.barrier(channel=0)
    
    data_ptrs = [int(data_hdl.buffer_ptrs[i]) for i in range(world_size)]
    ctrl_ptrs = [int(ctrl_hdl.buffer_ptrs[i]) for i in range(world_size)]
    
    # 4. Pull our assigned target data from peer buffers over NVLink
    ext.pull_data(
        ctrl_ptrs, data_ptrs, out_offsets, out_sizes,
        output, hidden_dim, world_size, rank
    )
    
    # 5. Fast synchronization for loop pipelining constraints
    data_hdl.barrier(channel=1)

    return output