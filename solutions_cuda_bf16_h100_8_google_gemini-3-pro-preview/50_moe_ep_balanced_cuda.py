"""
Strategy:
1. **Direct P2P Dispatch/Combine**: Fuses the `permute`, `all_to_all`, and `sort` into a single Push-based direct-memory-access CUDA kernel. Fuses the reverse `all_to_all` and `unpermute` into a single Pull-based CUDA kernel.
2. **Symmetric Memory via UVA**: Tokens and gradients are exchanged by writing/reading directly to/from symmetric buffers (`symm_mem.rendezvous`) over NVLink. This entirely bypasses NCCL host launch overheads and intermediate tensor copies.
3. **Zero Atomics Backward**: By generating token-to-expert mapping offsets dynamically on the forward pass, the backward passes for both dispatch and combine are entirely conflict-free and require zero atomics.
4. **Hardware Barriers**: Compute and communication are seamlessly ordered using fast device-side barriers (`hdl.barrier(channel=0)`), avoiding CPU synchronization stalls.
"""

from typing import Optional
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Helper to convert to float for accumulation
template <typename T> __device__ __forceinline__ float to_float(T val);
template <> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template <> __device__ __forceinline__ float to_float<double>(double val) { return static_cast<float>(val); }
template <> __device__ __forceinline__ float to_float<at::Half>(at::Half val) { return __half2float(val); }
template <> __device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 val) { return __bfloat162float(val); }

// Helper to convert from float
template <typename T> __device__ __forceinline__ T from_float(float val);
template <> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template <> __device__ __forceinline__ double from_float<double>(float val) { return static_cast<double>(val); }
template <> __device__ __forceinline__ at::Half from_float<at::Half>(float val) { return __float2half(val); }
template <> __device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float val) { return __float2bfloat16(val); }

template <typename scalar_t>
__global__ void dispatch_forward_kernel(
    const scalar_t* __restrict__ hidden_states,
    const int64_t* __restrict__ selected_experts,
    int* __restrict__ expert_counters,
    int* __restrict__ token_local_offsets,
    const int* __restrict__ recv_offsets,
    const uint64_t* __restrict__ remote_ptrs,
    int N, int K, int H
) {
    int nk = blockIdx.x;
    int i = nk / K;
    int e = selected_experts[nk];
    
    __shared__ int shared_local_offset;
    if (threadIdx.x == 0) {
        shared_local_offset = atomicAdd(&expert_counters[e], 1);
        token_local_offsets[nk] = shared_local_offset;
    }
    __syncthreads();
    
    int local_offset = shared_local_offset;
    int remote_offset = recv_offsets[e] + local_offset;
    scalar_t* remote_buf = reinterpret_cast<scalar_t*>(remote_ptrs[e]);
    
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        remote_buf[remote_offset * H + h] = hidden_states[i * H + h];
    }
}

template <typename scalar_t>
__global__ void dispatch_backward_kernel(
    scalar_t* __restrict__ grad_hidden_states,
    const int64_t* __restrict__ selected_experts,
    const int* __restrict__ token_local_offsets,
    const int* __restrict__ recv_offsets,
    const uint64_t* __restrict__ remote_ptrs,
    int N, int K, int H
) {
    int i = blockIdx.x;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e = selected_experts[i * K + k];
            int local_offset = token_local_offsets[i * K + k];
            int remote_offset = recv_offsets[e] + local_offset;
            const scalar_t* remote_buf = reinterpret_cast<const scalar_t*>(remote_ptrs[e]);
            sum += to_float(remote_buf[remote_offset * H + h]);
        }
        grad_hidden_states[i * H + h] = from_float<scalar_t>(sum);
    }
}

template <typename scalar_t>
__global__ void combine_forward_kernel(
    scalar_t* __restrict__ combined_output,
    const int64_t* __restrict__ selected_experts,
    const scalar_t* __restrict__ routing_weights,
    const int* __restrict__ token_local_offsets,
    const int* __restrict__ recv_offsets,
    const uint64_t* __restrict__ remote_ptrs,
    int N, int K, int H
) {
    int i = blockIdx.x;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e = selected_experts[i * K + k];
            float w = to_float(routing_weights[i * K + k]);
            int local_offset = token_local_offsets[i * K + k];
            int remote_offset = recv_offsets[e] + local_offset;
            const scalar_t* remote_buf = reinterpret_cast<const scalar_t*>(remote_ptrs[e]);
            sum += w * to_float(remote_buf[remote_offset * H + h]);
        }
        combined_output[i * H + h] = from_float<scalar_t>(sum);
    }
}

template <typename scalar_t>
__global__ void combine_backward_kernel(
    const scalar_t* __restrict__ grad_combined_output,
    const int64_t* __restrict__ selected_experts,
    const scalar_t* __restrict__ routing_weights,
    const int* __restrict__ token_local_offsets,
    const int* __restrict__ recv_offsets,
    const uint64_t* __restrict__ remote_grad_ptrs,
    const uint64_t* __restrict__ remote_expert_out_ptrs,
    scalar_t* __restrict__ grad_weights,
    int N, int K, int H
) {
    int nk = blockIdx.x;
    int i = nk / K;
    
    int e = selected_experts[nk];
    float w = to_float(routing_weights[nk]);
    int local_offset = token_local_offsets[nk];
    int remote_offset = recv_offsets[e] + local_offset;
    
    scalar_t* remote_grad_buf = reinterpret_cast<scalar_t*>(remote_grad_ptrs[e]);
    const scalar_t* remote_expert_out_buf = reinterpret_cast<const scalar_t*>(remote_expert_out_ptrs[e]);
    
    float dot_product = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float grad_out = to_float(grad_combined_output[i * H + h]);
        float expert_out = to_float(remote_expert_out_buf[remote_offset * H + h]);
        
        remote_grad_buf[remote_offset * H + h] = from_float<scalar_t>(grad_out * w);
        dot_product += grad_out * expert_out;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot_product += __shfl_down_sync(0xffffffff, dot_product, offset);
    }
    if (threadIdx.x == 0) {
        grad_weights[nk] = from_float<scalar_t>(dot_product);
    }
}

void launch_dispatch_forward(
    torch::Tensor hidden_states,
    torch::Tensor selected_experts,
    torch::Tensor expert_counters,
    torch::Tensor token_local_offsets,
    torch::Tensor recv_offsets,
    torch::Tensor remote_ptrs,
    int N, int K, int H
) {
    int threads = std::min(H, 1024);
    int blocks = N * K;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, hidden_states.scalar_type(), "dispatch_forward", ([&] {
        dispatch_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            hidden_states.data_ptr<scalar_t>(),
            selected_experts.data_ptr<int64_t>(),
            expert_counters.data_ptr<int>(),
            token_local_offsets.data_ptr<int>(),
            recv_offsets.data_ptr<int>(),
            reinterpret_cast<const uint64_t*>(remote_ptrs.data_ptr<int64_t>()),
            N, K, H
        );
    }));
}

void launch_dispatch_backward(
    torch::Tensor grad_hidden_states,
    torch::Tensor selected_experts,
    torch::Tensor token_local_offsets,
    torch::Tensor recv_offsets,
    torch::Tensor remote_ptrs,
    int N, int K, int H
) {
    int threads = std::min(H, 1024);
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_hidden_states.scalar_type(), "dispatch_backward", ([&] {
        dispatch_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_hidden_states.data_ptr<scalar_t>(),
            selected_experts.data_ptr<int64_t>(),
            token_local_offsets.data_ptr<int>(),
            recv_offsets.data_ptr<int>(),
            reinterpret_cast<const uint64_t*>(remote_ptrs.data_ptr<int64_t>()),
            N, K, H
        );
    }));
}

void launch_combine_forward(
    torch::Tensor combined_output,
    torch::Tensor selected_experts,
    torch::Tensor routing_weights,
    torch::Tensor token_local_offsets,
    torch::Tensor recv_offsets,
    torch::Tensor remote_ptrs,
    int N, int K, int H
) {
    int threads = std::min(H, 1024);
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, combined_output.scalar_type(), "combine_forward", ([&] {
        combine_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            combined_output.data_ptr<scalar_t>(),
            selected_experts.data_ptr<int64_t>(),
            routing_weights.data_ptr<scalar_t>(),
            token_local_offsets.data_ptr<int>(),
            recv_offsets.data_ptr<int>(),
            reinterpret_cast<const uint64_t*>(remote_ptrs.data_ptr<int64_t>()),
            N, K, H
        );
    }));
}

void launch_combine_backward(
    torch::Tensor grad_combined_output,
    torch::Tensor selected_experts,
    torch::Tensor routing_weights,
    torch::Tensor token_local_offsets,
    torch::Tensor recv_offsets,
    torch::Tensor remote_grad_ptrs,
    torch::Tensor remote_expert_out_ptrs,
    torch::Tensor grad_weights,
    int N, int K, int H
) {
    int threads = 32;
    int blocks = N * K;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_combined_output.scalar_type(), "combine_backward", ([&] {
        combine_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_combined_output.data_ptr<scalar_t>(),
            selected_experts.data_ptr<int64_t>(),
            routing_weights.data_ptr<scalar_t>(),
            token_local_offsets.data_ptr<int>(),
            recv_offsets.data_ptr<int>(),
            reinterpret_cast<const uint64_t*>(remote_grad_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(remote_expert_out_ptrs.data_ptr<int64_t>()),
            grad_weights.data_ptr<scalar_t>(),
            N, K, H
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_forward", &launch_dispatch_forward);
    m.def("dispatch_backward", &launch_dispatch_backward);
    m.def("combine_forward", &launch_combine_forward);
    m.def("combine_backward", &launch_combine_backward);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_moe_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def get_symm_buffer(name: str, shape: tuple, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (name, shape, dtype, device, group)
    if key not in _symm_cache:
        buf = symm_mem.empty(shape, dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        _symm_cache[key] = (buf, hdl, ptrs)
    return _symm_cache[key]

class FusedMoEDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, selected_experts, recv_offsets, M, dispatch_buf, dispatch_ptrs, group):
        ctx.group = group
        N, H = hidden_states.shape
        K = selected_experts.shape[1]
        W = recv_offsets.shape[0]
        
        expert_counters = torch.zeros(W, dtype=torch.int32, device=hidden_states.device)
        token_local_offsets = torch.empty_like(selected_experts, dtype=torch.int32)
        
        _, dispatch_hdl, _ = get_symm_buffer("dispatch", dispatch_buf.shape, hidden_states.dtype, hidden_states.device, group)
        dispatch_hdl.barrier(channel=0)
        
        _get_ext().dispatch_forward(
            hidden_states, selected_experts, expert_counters, token_local_offsets,
            recv_offsets, dispatch_ptrs, N, K, H
        )
        
        dispatch_hdl.barrier(channel=0)
        expert_inputs = dispatch_buf[:M].clone()
        
        ctx.save_for_backward(selected_experts, token_local_offsets, recv_offsets, dispatch_ptrs)
        ctx.N, ctx.K, ctx.H, ctx.W = N, K, H, W
        ctx.mark_non_differentiable(token_local_offsets)
        return expert_inputs, token_local_offsets

    @staticmethod
    def backward(ctx, grad_expert_inputs, grad_token_local_offsets):
        selected_experts, token_local_offsets, recv_offsets, dispatch_ptrs = ctx.saved_tensors
        
        grad_dispatch_buf, grad_dispatch_hdl, grad_dispatch_ptrs = get_symm_buffer(
            "grad_dispatch", (ctx.W * ctx.N * ctx.K, ctx.H), 
            grad_expert_inputs.dtype, grad_expert_inputs.device, ctx.group
        )
        grad_dispatch_buf[:grad_expert_inputs.shape[0]].copy_(grad_expert_inputs)
        
        grad_dispatch_hdl.barrier(channel=0)
        
        grad_hidden_states = torch.empty((ctx.N, ctx.H), dtype=grad_expert_inputs.dtype, device=grad_expert_inputs.device)
        _get_ext().dispatch_backward(
            grad_hidden_states, selected_experts, token_local_offsets,
            recv_offsets, grad_dispatch_ptrs, ctx.N, ctx.K, ctx.H
        )
        
        grad_dispatch_hdl.barrier(channel=0)
        return grad_hidden_states, None, None, None, None, None, None

class FusedMoECombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, selected_experts, routing_weights, token_local_offsets, recv_offsets, combine_buf, combine_ptrs, group):
        ctx.group = group
        N, K = selected_experts.shape
        M, H = expert_outputs.shape
        
        combine_buf[:M].copy_(expert_outputs)
        
        _, combine_hdl, _ = get_symm_buffer("combine", combine_buf.shape, expert_outputs.dtype, expert_outputs.device, group)
        combine_hdl.barrier(channel=0)
        
        combined_output = torch.empty((N, H), dtype=expert_outputs.dtype, device=expert_outputs.device)
        _get_ext().combine_forward(
            combined_output, selected_experts, routing_weights, token_local_offsets,
            recv_offsets, combine_ptrs, N, K, H
        )
        
        combine_hdl.barrier(channel=0)
        
        ctx.save_for_backward(expert_outputs, selected_experts, routing_weights, token_local_offsets, recv_offsets)
        ctx.N, ctx.K, ctx.H, ctx.M, ctx.W = N, K, H, M, recv_offsets.shape[0]
        return combined_output

    @staticmethod
    def backward(ctx, grad_combined_output):
        expert_outputs, selected_experts, routing_weights, token_local_offsets, recv_offsets = ctx.saved_tensors
        MAX_TOKENS = ctx.W * ctx.N * ctx.K
        
        combine_bwd_buf, combine_bwd_hdl, combine_bwd_ptrs = get_symm_buffer(
            "combine_bwd_expert_out", (MAX_TOKENS, ctx.H), 
            expert_outputs.dtype, expert_outputs.device, ctx.group
        )
        combine_bwd_buf[:ctx.M].copy_(expert_outputs)
        
        grad_combine_buf, grad_combine_hdl, grad_combine_ptrs = get_symm_buffer(
            "grad_combine", (MAX_TOKENS, ctx.H), 
            grad_combined_output.dtype, grad_combined_output.device, ctx.group
        )
        grad_combine_buf[:ctx.M].zero_()
        
        combine_bwd_hdl.barrier(channel=0)
        grad_combine_hdl.barrier(channel=0)
        
        grad_weights = torch.empty_like(routing_weights)
        _get_ext().combine_backward(
            grad_combined_output, selected_experts, routing_weights,
            token_local_offsets, recv_offsets, grad_combine_ptrs,
            combine_bwd_ptrs, grad_weights, ctx.N, ctx.K, ctx.H
        )
        
        grad_combine_hdl.barrier(channel=0)
        grad_expert_outputs = grad_combine_buf[:ctx.M].clone()
        return grad_expert_outputs, None, grad_weights, None, None, None, None, None

def expert_forward(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
) -> torch.Tensor:
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)

def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    hidden_dim = hidden_states.size(-1)
    original_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    N, H = hidden_states.shape
    K = top_k
    W = world_size
    MAX_TOKENS = W * N * K
    
    router_logits = torch.nn.functional.linear(hidden_states, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    
    send_counts = torch.bincount(selected_experts.view(-1), minlength=W).to(torch.int32)
    counts_matrix = torch.empty((W, W), dtype=torch.int32, device=hidden_states.device)
    dist.all_gather_into_tensor(counts_matrix.view(-1), send_counts, group=group)
    
    recv_offsets_matrix = counts_matrix.cumsum(dim=0) - counts_matrix
    recv_offsets = recv_offsets_matrix[rank].contiguous()
    M = counts_matrix[:, rank].sum().item()
    
    dispatch_buf, _, dispatch_ptrs = get_symm_buffer(
        "dispatch", (MAX_TOKENS, H), hidden_states.dtype, hidden_states.device, group
    )
    combine_buf, _, combine_ptrs = get_symm_buffer(
        "combine", (MAX_TOKENS, H), hidden_states.dtype, hidden_states.device, group
    )
    
    expert_inputs, token_local_offsets = FusedMoEDispatch.apply(
        hidden_states, selected_experts, recv_offsets, M, dispatch_buf, dispatch_ptrs, group
    )
    
    expert_outputs = expert_forward(expert_inputs, gate_proj, up_proj, down_proj)
    
    out = FusedMoECombine.apply(
        expert_outputs, selected_experts, routing_weights, token_local_offsets,
        recv_offsets, combine_buf, combine_ptrs, group
    )
    
    return out.reshape(original_shape)