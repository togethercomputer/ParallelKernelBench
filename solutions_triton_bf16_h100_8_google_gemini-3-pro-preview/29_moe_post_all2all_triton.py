# Strategy:
# 1. We exploit UVA symmetric memory to fuse the pre-all-to-all chunk sorting and cross-rank routing.
#    Instead of slicing tensors and relying on a host-driven all_to_all_single, ranks exchange WxW offset metadata via
#    a lightweight symm_mem buffer. Then, a single custom CUDA kernel (`push_kernel_flat`) writes the chunks of 
#    `expert_outputs` directly into the destination peer's symmetric memory over NVLink.
# 2. We fuse the unpermute steps. Instead of materializing `weights_idx` via scatter_add and running `masked_select`, 
#    our `unpermute_kernel_token` performs a binary search over expert bounds per token, retrieves the weight, scales, 
#    and scatters directly into the final `unpermuted_tokens` tensor using hardware `atomicAdd`.
# 3. Compute and communication overlap is intrinsically optimized since memory writes happen cross-device from a 
#    grid-stride loop with vectorized 128-bit memory instructions, achieving near peak NVLink bandwidth.

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

__global__ void compute_offsets_kernel(
    const int32_t* __restrict__ S, // [E, W]
    const int32_t* __restrict__ all_send_sizes, // [W, W]
    int32_t* __restrict__ src_offsets,
    int32_t* __restrict__ dst_offsets,
    int32_t* __restrict__ chunk_r,
    int* __restrict__ num_valid_chunks,
    int rank, int E, int W
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current_src = 0;
        int valid_idx = 0;
        for (int e = 0; e < E; ++e) {
            for (int r = 0; r < W; ++r) {
                int sz = S[e * W + r];
                if (sz > 0) {
                    src_offsets[valid_idx] = current_src;
                    
                    int dst_base = 0;
                    for (int j = 0; j < rank; ++j) {
                        dst_base += all_send_sizes[j * W + r];
                    }
                    int dst_expert_offset = 0;
                    for (int prev_e = 0; prev_e < e; ++prev_e) {
                        dst_expert_offset += S[prev_e * W + r];
                    }
                    dst_offsets[valid_idx] = dst_base + dst_expert_offset;
                    chunk_r[valid_idx] = r;
                    valid_idx++;
                }
                current_src += sz;
            }
        }
        *num_valid_chunks = valid_idx;
    }
}

__global__ void push_kernel_flat(
    const at::BFloat16* __restrict__ expert_outputs,
    const int64_t* __restrict__ peer_recv_ptrs_int,
    const int32_t* __restrict__ src_offsets,
    const int32_t* __restrict__ dst_offsets,
    const int32_t* __restrict__ chunk_r,
    const int* __restrict__ num_valid_chunks_ptr,
    int total_tokens, int H_vecs
) {
    int num_chunks = *num_valid_chunks_ptr;
    if (num_chunks == 0) return;
    
    int total_vecs = total_tokens * H_vecs;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = tid; i < total_vecs; i += blockDim.x * gridDim.x) {
        int token_idx = i / H_vecs;
        int vec_in_token = i % H_vecs;
        
        int low = 0, high = num_chunks - 1;
        int chunk_idx = 0;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (token_idx >= src_offsets[mid]) {
                chunk_idx = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        int r = chunk_r[chunk_idx];
        int token_offset_in_chunk = token_idx - src_offsets[chunk_idx];
        int dst_token_idx = dst_offsets[chunk_idx] + token_offset_in_chunk;
        
        const float4* src = reinterpret_cast<const float4*>(expert_outputs);
        at::BFloat16* dst_ptr = reinterpret_cast<at::BFloat16*>(peer_recv_ptrs_int[r]);
        float4* dst = reinterpret_cast<float4*>(dst_ptr);
        
        dst[dst_token_idx * H_vecs + vec_in_token] = src[i];
    }
}

__global__ void unpermute_kernel_token(
    const at::BFloat16* __restrict__ recv_buf,
    const int64_t* __restrict__ permutation_mapping,
    const int32_t* __restrict__ expert_bounds,
    const float* __restrict__ routing_weights,
    const int64_t* __restrict__ selected_experts,
    at::BFloat16* __restrict__ unpermuted_tokens,
    int N_routed, int H, int topk, int num_experts
) {
    int idx = blockIdx.x; 
    if (idx >= N_routed) return;
    
    __shared__ int expert_id;
    __shared__ float weight;
    __shared__ int64_t t;
    
    if (threadIdx.x == 0) {
        int low = 0, high = num_experts - 1;
        int found_expert = 0;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (idx < expert_bounds[mid]) {
                found_expert = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        expert_id = found_expert;
        t = permutation_mapping[idx];
        
        float w = 0.0f;
        for (int k = 0; k < topk; ++k) {
            if (selected_experts[t * topk + k] == expert_id) {
                w += routing_weights[t * topk + k]; // Accumulate in case of duplicate expert assignment
            }
        }
        weight = w;
    }
    
    __syncthreads();
    
    float w = weight;
    int64_t token_t = t;
    
    const at::BFloat16* src = recv_buf + idx * H;
    at::BFloat16* dst = unpermuted_tokens + token_t * H;
    
    int H_vecs = H / 8;
    for (int i = threadIdx.x; i < H_vecs; i += blockDim.x) {
        float4 val_vec = reinterpret_cast<const float4*>(src)[i];
        at::BFloat16* val_ptr = reinterpret_cast<at::BFloat16*>(&val_vec);
        
        for (int v = 0; v < 8; ++v) {
            float val_f = static_cast<float>(val_ptr[v]) * w;
            atomicAdd(reinterpret_cast<__nv_bfloat16*>(dst + i * 8 + v),
                      __float2bfloat16(val_f));
        }
    }
}

void run_push(
    torch::Tensor expert_outputs, 
    torch::Tensor chunk_sizes, 
    torch::Tensor all_send_sizes, 
    torch::Tensor peer_ptrs_tensor, 
    int rank, int E, int W
) {
    int total_tokens = expert_outputs.size(0);
    if (total_tokens == 0) return;
    int H = expert_outputs.size(1);
    
    auto options = torch::TensorOptions().device(expert_outputs.device()).dtype(torch::kInt32);
    torch::Tensor src_offsets = torch::empty({E * W}, options);
    torch::Tensor dst_offsets = torch::empty({E * W}, options);
    torch::Tensor chunk_r = torch::empty({E * W}, options);
    torch::Tensor num_valid_chunks = torch::empty({1}, options);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    compute_offsets_kernel<<<1, 1, 0, stream>>>(
        chunk_sizes.data_ptr<int32_t>(),
        all_send_sizes.data_ptr<int32_t>(),
        src_offsets.data_ptr<int32_t>(),
        dst_offsets.data_ptr<int32_t>(),
        chunk_r.data_ptr<int32_t>(),
        num_valid_chunks.data_ptr<int>(),
        rank, E, W
    );
    
    int H_vecs = H / 8;
    int total_vecs = total_tokens * H_vecs;
    int threads = 256;
    int blocks = std::min(65535, (total_vecs + threads - 1) / threads);
    
    push_kernel_flat<<<blocks, threads, 0, stream>>>(
        expert_outputs.data_ptr<at::BFloat16>(),
        peer_ptrs_tensor.data_ptr<int64_t>(),
        src_offsets.data_ptr<int32_t>(),
        dst_offsets.data_ptr<int32_t>(),
        chunk_r.data_ptr<int32_t>(),
        num_valid_chunks.data_ptr<int>(),
        total_tokens, H_vecs
    );
}

void run_unpermute(
    torch::Tensor recv_buf, 
    torch::Tensor permutation_mapping, 
    torch::Tensor expert_bounds, 
    torch::Tensor routing_weights, 
    torch::Tensor selected_experts, 
    torch::Tensor unpermuted_tokens, 
    int num_experts
) {
    int N_routed = recv_buf.size(0);
    if (N_routed == 0) return;
    int H = recv_buf.size(1);
    int topk = routing_weights.size(1);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    int threads = 256;
    int blocks = N_routed; 
    
    unpermute_kernel_token<<<blocks, threads, 0, stream>>>(
        recv_buf.data_ptr<at::BFloat16>(),
        permutation_mapping.data_ptr<int64_t>(),
        expert_bounds.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        selected_experts.data_ptr<int64_t>(),
        unpermuted_tokens.data_ptr<at::BFloat16>(),
        N_routed, H, topk, num_experts
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_push", &run_push, "Push tokens to peers via UVA");
    m.def("run_unpermute", &run_unpermute, "Unpermute and scatter-add tokens");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_post_all2all_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(name: str, size: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if name in _symm_cache:
        c = _symm_cache[name]
        if c['size'] >= size:
            return c['buf'], c['hdl']
    
    buf = symm_mem.empty(size, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[name] = {'size': size, 'buf': buf, 'hdl': hdl}
    return buf, hdl

# Reference PyTorch implementation for non-optimized edge cases (H % 8 != 0)
def _ref_solution(expert_outputs, routing_weights, selected_experts, num_experts, input_splits, output_splits, 
                  num_global_tokens_per_local_expert, routing_map, local_input_permutation_mapping, org_hidden_states_shape, group):
    from tommi.distributed.moe.moe_layer import _sort_chunks_by_idxs, _all_to_all_forward, _generate_weights_idx, _unpermute
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, num_global_tokens_per_local_expert.T.ravel(), unpermute_order)
    unpermute_outputs = _all_to_all_forward(group, expert_outputs, input_splits, output_splits)
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    unpermute_outputs = _unpermute(unpermute_outputs, weights_idx, org_hidden_states_shape, local_input_permutation_mapping, routing_map)
    return unpermute_outputs

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
    H = org_hidden_states_shape[-1]
    
    # Kernel requires vectorization bounds (H multiple of 8)
    if H % 8 != 0:
        return _ref_solution(
            expert_outputs, routing_weights, selected_experts, num_experts, input_splits, 
            output_splits, num_global_tokens_per_local_expert, routing_map, 
            local_input_permutation_mapping, org_hidden_states_shape, group
        )
    
    ext = _get_ext()
    device = expert_outputs.device
    
    output_splits_list = output_splits.tolist() if isinstance(output_splits, torch.Tensor) else output_splits
    input_splits_list = input_splits.tolist() if isinstance(input_splits, torch.Tensor) else input_splits

    # 1. Exchange WxW offset metadata via symmetric memory
    meta_buf_full, meta_hdl = _get_symm_state("meta", W * W, torch.int32, device)
    meta_buf = meta_buf_full[:W * W]
    meta_buf[rank * W : (rank + 1) * W] = torch.tensor(output_splits_list, dtype=torch.int32, device=device)
    meta_hdl.barrier(channel=0)
    all_send_sizes = meta_buf.view(W, W)
    
    # 2. Prepare dynamically sized symmetric receive buffer
    N_routed = sum(input_splits_list)
    recv_buf_full, recv_hdl = _get_symm_state("recv", N_routed * H, torch.bfloat16, device)
    recv_buf = recv_buf_full[:N_routed * H].view(N_routed, H)
    
    peer_ptrs = [int(recv_hdl.buffer_ptrs[r]) for r in range(W)]
    peer_ptrs_tensor = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)
    
    expert_outputs = expert_outputs.contiguous()
    chunk_sizes = num_global_tokens_per_local_expert.to(torch.int32).contiguous()
    E = num_experts // W
    
    # 3. Direct UVA push over NVLink (implicit pre-all2all scatter + network cross)
    ext.run_push(expert_outputs, chunk_sizes, all_send_sizes, peer_ptrs_tensor, rank, E, W)
    
    # Wait for peers to finish writing to our symmetric buffer
    recv_hdl.barrier(channel=0)
    
    # 4. Process the received payload: scale and accumulate tokens directly to dest tensor
    expert_bounds = routing_map.to(torch.int32).sum(dim=1).cumsum(0, dtype=torch.int32)
    local_input_permutation_mapping = local_input_permutation_mapping.to(torch.int64)
    selected_experts = selected_experts.to(torch.int64)
    routing_weights = routing_weights.to(torch.float32).contiguous()
    
    unpermuted_tokens = torch.zeros(org_hidden_states_shape, dtype=torch.bfloat16, device=device)
    
    ext.run_unpermute(
        recv_buf,
        local_input_permutation_mapping,
        expert_bounds,
        routing_weights,
        selected_experts,
        unpermuted_tokens,
        num_experts
    )
    
    return unpermuted_tokens