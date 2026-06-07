import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Union
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Direct PUSH: Reads from unsorted local expert_outputs and writes directly to remote recv_buf.
__global__ void push_kernel_vec(
    const __nv_bfloat16* __restrict__ expert_outputs,
    const int32_t* __restrict__ meta_info,
    const uint64_t* __restrict__ recv_buf_ptrs,
    int E,
    int hidden_dim
) {
    int k = blockIdx.y;
    if (k >= E) return;
    
    int src_offset = meta_info[k * 4 + 0];
    int dest_offset = meta_info[k * 4 + 1];
    int size = meta_info[k * 4 + 2];
    int dest_rank = meta_info[k * 4 + 3];
    
    if (size == 0) return;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    if (hidden_dim % 8 == 0) {
        int total_vecs = (size * hidden_dim) / 8;
        const float4* src = reinterpret_cast<const float4*>(expert_outputs + src_offset * hidden_dim);
        float4* dest = reinterpret_cast<float4*>(
            reinterpret_cast<__nv_bfloat16*>(recv_buf_ptrs[dest_rank]) + dest_offset * hidden_dim
        );
        for (int i = tid; i < total_vecs; i += stride) {
            dest[i] = src[i];
        }
    } else {
        int total_elements = size * hidden_dim;
        const __nv_bfloat16* src = expert_outputs + src_offset * hidden_dim;
        __nv_bfloat16* dest = reinterpret_cast<__nv_bfloat16*>(recv_buf_ptrs[dest_rank]) + dest_offset * hidden_dim;
        for (int i = tid; i < total_elements; i += stride) {
            dest[i] = src[i];
        }
    }
}

// Fuses the elementwise multiply with the routing weight and the atomic scatter_add
__global__ void unpermute_fused_kernel(
    const __nv_bfloat16* __restrict__ recv_buf,
    const __nv_bfloat16* __restrict__ tokens_weight,
    const int64_t* __restrict__ permutation_mapping,
    __nv_bfloat16* __restrict__ unpermuted_tokens,
    int total_received,
    int hidden_dim
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_received) return;

    int orig_idx = permutation_mapping[token_idx];
    float weight = __bfloat162float(tokens_weight[token_idx]);

    const __nv_bfloat16* src = recv_buf + token_idx * hidden_dim;
    __nv_bfloat16* dst = unpermuted_tokens + orig_idx * hidden_dim;

    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        float val = __bfloat162float(src[d]) * weight;
        #if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
        atomicAdd(dst + d, __float2bfloat16(val));
        #endif
    }
}

void launch_push(
    torch::Tensor expert_outputs,
    torch::Tensor meta_info,
    torch::Tensor recv_buf_ptrs,
    int E,
    int hidden_dim
) {
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(expert_outputs.data_ptr<at::BFloat16>());
    const int32_t* meta = meta_info.data_ptr<int32_t>();
    const uint64_t* ptrs = reinterpret_cast<const uint64_t*>(recv_buf_ptrs.data_ptr<int64_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(32, E);
    dim3 block(256);
    push_kernel_vec<<<grid, block, 0, stream>>>(src, meta, ptrs, E, hidden_dim);
}

void launch_unpermute(
    torch::Tensor recv_buf,
    torch::Tensor tokens_weight,
    torch::Tensor permutation_mapping,
    torch::Tensor unpermuted_tokens,
    int total_received,
    int hidden_dim
) {
    if (total_received == 0) return;
    
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr<at::BFloat16>());
    const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(tokens_weight.data_ptr<at::BFloat16>());
    const int64_t* mapping = permutation_mapping.data_ptr<int64_t>();
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(unpermuted_tokens.data_ptr<at::BFloat16>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(total_received);
    dim3 block(256);
    
    unpermute_fused_kernel<<<grid, block, 0, stream>>>(
        src, weights, mapping, dst, total_received, hidden_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push", &launch_push, "Push chunks to remote symmetric recv_buf");
    m.def("launch_unpermute", &launch_unpermute, "Fused weight unpermute and scatter_add");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_post_all2all_ext", CUDA_SRC)
    return _ext

_comm_stream = None
def _get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
    return _comm_stream

@torch.no_grad()
def solution(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or dist.group.WORLD
    W = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = expert_outputs.device
    
    expert_outputs = expert_outputs.contiguous()
    if expert_outputs.dtype != torch.bfloat16:
        expert_outputs = expert_outputs.to(torch.bfloat16)
        
    hidden_dim = expert_outputs.size(1)
    
    input_splits_list = input_splits.tolist() if isinstance(input_splits, torch.Tensor) else input_splits
    out_size = sum(input_splits_list)
    
    # Fast path for single rank (W=1)
    if W == 1:
        # Standard unpermute block, ignoring P2P logic
        unpermuted_tokens = torch.zeros(org_hidden_states_shape, dtype=torch.bfloat16, device=device)
        weights_idx = torch.zeros((routing_weights.size(0), num_experts), dtype=routing_weights.dtype, device=device)
        weights_idx.scatter_add_(1, selected_experts, routing_weights)
        tokens_weight = weights_idx.T.contiguous().masked_select(routing_map.bool()).to(torch.bfloat16)
        
        # Sort using Python lists
        L = num_experts
        split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist() if isinstance(num_global_tokens_per_local_expert, torch.Tensor) else num_global_tokens_per_local_expert
        unpermute_order = torch.arange(num_experts).reshape(L, -1).T.ravel().tolist()
        
        chunks = torch.split(expert_outputs, split_sizes, dim=0)
        recv_buf = torch.cat([chunks[i] for i in unpermute_order], dim=0)
        
        _get_ext().launch_unpermute(
            recv_buf, tokens_weight, local_input_permutation_mapping.to(torch.int64),
            unpermuted_tokens.view(-1, hidden_dim), out_size, hidden_dim
        )
        return unpermuted_tokens

    # --- P2P MULTI-GPU PIPELINE ---
    
    # 1. Swiftly exchange split counts to precisely know remote destination offsets
    output_splits_t = output_splits.to(torch.int32) if isinstance(output_splits, torch.Tensor) else torch.tensor(output_splits, dtype=torch.int32, device=device)
    gathered_splits = torch.empty(W * W, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(gathered_splits, output_splits_t, group=group)
    gathered_splits = gathered_splits.view(W, W)
    
    # 2. Build explicit map of where every chunk needs to land remotely (compute safely on CPU)
    E = num_experts
    L = E // W
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist() if isinstance(num_global_tokens_per_local_expert, torch.Tensor) else num_global_tokens_per_local_expert

    chunk_src_offsets = [0] * E
    curr = 0
    for i in range(E):
        chunk_src_offsets[i] = curr
        curr += split_sizes[i]

    dest_offsets = gathered_splits[:rank, :].sum(dim=0).tolist()
    curr_dest_offsets = dest_offsets.copy()
    unpermute_order = torch.arange(E).reshape(L, -1).T.ravel().tolist()

    meta_info_cpu = torch.zeros((E, 4), dtype=torch.int32)
    for k in range(E):
        dest_rank = k // L
        orig_idx = unpermute_order[k]
        size = split_sizes[orig_idx]
        meta_info_cpu[k, 0] = chunk_src_offsets[orig_idx]
        meta_info_cpu[k, 1] = curr_dest_offsets[dest_rank]
        meta_info_cpu[k, 2] = size
        meta_info_cpu[k, 3] = dest_rank
        curr_dest_offsets[dest_rank] += size

    meta_info = meta_info_cpu.to(device, non_blocking=True)

    # 3. Setup Symmetric Memory Buffer
    recv_buf = symm_mem.empty((out_size, hidden_dim), dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(recv_buf, group)
    hdl.barrier(channel=0)
    
    # 4. Overlapped Async Network PUSH 
    # Reads unordered `expert_outputs` and correctly sorts *during* the NVLink PUSH copy.
    comm_stream = _get_comm_stream()
    comm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(comm_stream):
        recv_buf_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        _get_ext().launch_push(expert_outputs, meta_info, recv_buf_ptrs, E, hidden_dim)

    # 5. Hide routing math latency behind the PUSH using Default Stream
    num_tokens = routing_weights.size(0)
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    tokens_weight = weights_idx.T.contiguous().masked_select(routing_map.bool()).to(torch.bfloat16)
    
    # 6. Global P2P Finalization
    torch.cuda.current_stream().wait_stream(comm_stream)
    hdl.barrier(channel=0)

    # 7. Execute Native Fused Unpermute
    unpermuted_tokens = torch.zeros(org_hidden_states_shape, dtype=torch.bfloat16, device=device)
    _get_ext().launch_unpermute(
        recv_buf,
        tokens_weight,
        local_input_permutation_mapping.to(torch.int64),
        unpermuted_tokens.view(-1, hidden_dim),
        out_size,
        hidden_dim
    )
    
    return unpermuted_tokens