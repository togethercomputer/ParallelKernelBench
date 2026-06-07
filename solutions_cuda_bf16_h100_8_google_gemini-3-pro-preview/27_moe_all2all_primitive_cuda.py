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

template <typename T>
__global__ void pull_all_to_all_kernel(
    const uint64_t* __restrict__ remote_data_ptrs,
    const uint64_t* __restrict__ remote_offset_ptrs,
    const int64_t* __restrict__ my_out_offsets,
    const int64_t* __restrict__ my_output_splits,
    T* __restrict__ out_data,
    int64_t hidden_dim,
    int world_size,
    int rank
) {
    int p = blockIdx.y; // peer index
    if (p >= world_size) return;

    int64_t pull_rows = my_output_splits[p];
    if (pull_rows == 0) return;

    __shared__ int64_t shared_remote_offset;
    if (threadIdx.x == 0) {
        // Read remote offset from the peer's offset buffer natively via UVA
        const int64_t* peer_offset_buf = reinterpret_cast<const int64_t*>(remote_offset_ptrs[p]);
        shared_remote_offset = peer_offset_buf[rank];
    }
    __syncthreads();
    
    int64_t remote_offset = shared_remote_offset;
    int64_t my_out_offset = my_out_offsets[p];

    const T* peer_data = reinterpret_cast<const T*>(remote_data_ptrs[p]);
    const T* src = peer_data + remote_offset * hidden_dim;
    T* dst = out_data + my_out_offset * hidden_dim;

    int64_t total_elements = pull_rows * hidden_dim;
    
    // Fast path: 128-bit aligned vectorized loads across NVLink
    if (reinterpret_cast<uintptr_t>(src) % 16 == 0 && 
        reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
        total_elements % 8 == 0) 
    {
        int64_t total_vecs = total_elements / 8;
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        const ulonglong2* src_vec = reinterpret_cast<const ulonglong2*>(src);
        ulonglong2* dst_vec = reinterpret_cast<ulonglong2*>(dst);
        
        for (; idx < total_vecs; idx += (int64_t)gridDim.x * blockDim.x) {
            dst_vec[idx] = src_vec[idx];
        }
    } else {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        for (; idx < total_elements; idx += (int64_t)gridDim.x * blockDim.x) {
            dst[idx] = src[idx];
        }
    }
}

void launch_pull_all_to_all(
    torch::Tensor remote_data_ptrs_tensor,
    torch::Tensor remote_offset_ptrs_tensor,
    torch::Tensor my_out_offsets_tensor,
    torch::Tensor my_output_splits_tensor,
    torch::Tensor out_data,
    int64_t hidden_dim,
    int world_size,
    int rank
) {
    const uint64_t* remote_data_ptrs = reinterpret_cast<const uint64_t*>(remote_data_ptrs_tensor.data_ptr<int64_t>());
    const uint64_t* remote_offset_ptrs = reinterpret_cast<const uint64_t*>(remote_offset_ptrs_tensor.data_ptr<int64_t>());
    const int64_t* my_out_offsets = my_out_offsets_tensor.data_ptr<int64_t>();
    const int64_t* my_output_splits = my_output_splits_tensor.data_ptr<int64_t>();

    int threads = 512;
    int blocks_x = 32; // Over-subscribe SMs to hide latency
    dim3 grid(blocks_x, world_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out_data.dtype() == torch::kBFloat16) {
        __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out_data.data_ptr<at::BFloat16>());
        pull_all_to_all_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            remote_data_ptrs, remote_offset_ptrs, my_out_offsets, my_output_splits,
            out_ptr, hidden_dim, world_size, rank
        );
    } else if (out_data.dtype() == torch::kFloat16) {
        half* out_ptr = reinterpret_cast<half*>(out_data.data_ptr<at::Half>());
        pull_all_to_all_kernel<half><<<grid, threads, 0, stream>>>(
            remote_data_ptrs, remote_offset_ptrs, my_out_offsets, my_output_splits,
            out_ptr, hidden_dim, world_size, rank
        );
    } else if (out_data.dtype() == torch::kFloat32) {
        float* out_ptr = out_data.data_ptr<float>();
        pull_all_to_all_kernel<float><<<grid, threads, 0, stream>>>(
            remote_data_ptrs, remote_offset_ptrs, my_out_offsets, my_output_splits,
            out_ptr, hidden_dim, world_size, rank
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype, must be bf16, fp16, or fp32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pull_all_to_all", &launch_pull_all_to_all, "UVA Pull All-to-All kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_all2all_pull_ext", CUDA_SRC)
    return _ext

class SymmMemCache:
    def __init__(self):
        self.max_tokens = 0
        self.hidden_dim = 0
        self.data_buf = None
        self.data_hdl = None
        self.offset_buf = None
        self.offset_hdl = None
        self.data_ptrs = None
        self.offset_ptrs = None

_cache = SymmMemCache()

def _get_symm_buffers(num_tokens: int, hidden_dim: int, device: torch.device, dtype: torch.dtype, world_size: int, group: dist.ProcessGroup):
    global _cache
    if _cache.data_buf is None:
        # Initial allocation - make it huge to avoid costly runtime reallocations/all_reduce barriers.
        local_info = torch.tensor([num_tokens, hidden_dim], dtype=torch.int64, device=device)
        dist.all_reduce(local_info, op=dist.ReduceOp.MAX, group=group)
        global_tokens = local_info[0].item()
        global_hidden = local_info[1].item()
        
        # 8x upper bound margin to absorb MoE imbalance/load spikes.
        # Fallback 262144 ensures tiny initial batches don't undershoot later huge inputs.
        new_max = max(global_tokens * 8, 262144)
        _cache.max_tokens = new_max
        _cache.hidden_dim = global_hidden
        
        _cache.data_buf = symm_mem.empty((new_max, global_hidden), device=device, dtype=dtype)
        _cache.data_hdl = symm_mem.rendezvous(_cache.data_buf, group)
        _cache.data_ptrs = torch.tensor(_cache.data_hdl.buffer_ptrs, device=device, dtype=torch.int64)
        
        _cache.offset_buf = symm_mem.empty((world_size,), device=device, dtype=torch.int64)
        _cache.offset_hdl = symm_mem.rendezvous(_cache.offset_buf, group)
        _cache.offset_ptrs = torch.tensor(_cache.offset_hdl.buffer_ptrs, device=device, dtype=torch.int64)
        
    elif num_tokens > _cache.max_tokens or hidden_dim != _cache.hidden_dim:
        raise RuntimeError(
            f"Dynamic reallocation needed but omitted to avoid deadlock. "
            f"num_tokens={num_tokens} exceeds max {_cache.max_tokens} or hidden_dim={hidden_dim} changed."
        )
        
    return _cache

@torch.no_grad()
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

    if _ext is None:
        if dist.get_rank(group) == 0:
            _get_ext()
        dist.barrier(group=group)

    rank = dist.get_rank(group)
    device = local_tensor.device
    dtype = local_tensor.dtype
    local_tensor = local_tensor.contiguous()
    num_tokens = local_tensor.size(0)
    hidden_dim = local_tensor.size(1)

    # 1. Compute split sizes and block offsets purely on the device
    if input_split_sizes is None:
        split = num_tokens // world_size
        input_splits_dev = torch.full((world_size,), split, dtype=torch.int64, device=device)
    elif isinstance(input_split_sizes, list):
        input_splits_dev = torch.tensor(input_split_sizes, dtype=torch.int64, device=device)
    else:
        input_splits_dev = input_split_sizes.to(device)

    input_offsets_dev = torch.zeros(world_size, dtype=torch.int64, device=device)
    input_offsets_dev[1:] = torch.cumsum(input_splits_dev[:-1], dim=0)

    if output_split_sizes is None:
        split = num_tokens // world_size
        output_splits_dev = torch.full((world_size,), split, dtype=torch.int64, device=device)
        out_size = num_tokens
    elif isinstance(output_split_sizes, list):
        output_splits_dev = torch.tensor(output_split_sizes, dtype=torch.int64, device=device)
        out_size = sum(output_split_sizes)
    else:
        output_splits_dev = output_split_sizes.to(device)
        out_size = int(output_splits_dev.sum().item())

    output_offsets_dev = torch.zeros(world_size, dtype=torch.int64, device=device)
    output_offsets_dev[1:] = torch.cumsum(output_splits_dev[:-1], dim=0)

    output = torch.empty((out_size, hidden_dim), dtype=dtype, device=device)

    if out_size == 0 and num_tokens == 0:
        return output

    # 2. Acquire persistent symmetric layout structures
    cache = _get_symm_buffers(num_tokens, hidden_dim, device, dtype, world_size, group)

    # 3. Synchronize stream before populating new data to ensure no ongoing trailing reads from peers
    cache.data_hdl.barrier(channel=0)

    # 4. Fill local symmetric views
    if num_tokens > 0:
        cache.data_buf[:num_tokens].copy_(local_tensor)
    cache.offset_buf[:world_size].copy_(input_offsets_dev)

    # 5. Synchronize after populating the arrays to assure readiness globally
    cache.data_hdl.barrier(channel=0)

    # 6. Execute highly parallel CUDA pull directly bypassing NCCL 
    if out_size > 0:
        _get_ext().launch_pull_all_to_all(
            cache.data_ptrs,
            cache.offset_ptrs,
            output_offsets_dev,
            output_splits_dev,
            output,
            hidden_dim,
            world_size,
            rank
        )

    return output