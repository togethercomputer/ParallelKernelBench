import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple, Union
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename T, typename TVec>
__global__ void uva_push_kernel_vec(
    const TVec* __restrict__ input,
    TVec* const* __restrict__ peer_ptrs,
    const int* __restrict__ push_counts,
    const int* __restrict__ read_offsets,
    const int* __restrict__ write_offsets,
    int H_vec
) {
    int dst_rank = blockIdx.y;
    int count = push_counts[dst_rank];
    if (count == 0) return;

    const TVec* src = input + read_offsets[dst_rank] * H_vec;
    TVec* dst = peer_ptrs[dst_rank] + write_offsets[dst_rank] * H_vec;

    int total_elements = count * H_vec;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

void uva_push(
    torch::Tensor input,
    std::vector<int64_t> peer_ptrs,
    torch::Tensor push_counts,
    torch::Tensor read_offsets,
    torch::Tensor write_offsets,
    int H
) {
    int ep_size = peer_ptrs.size();
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
    torch::Tensor peer_ptrs_tensor = torch::empty({ep_size}, options);
    peer_ptrs_tensor.copy_(torch::tensor(peer_ptrs, torch::kInt64));

    int blocks_x = 4;
    dim3 grid(blocks_x, ep_size);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (input.dtype() == torch::kBFloat16) {
        int H_vec = H / 8; // 8 bf16 = 16 bytes = uint4
        uva_push_kernel_vec<at::BFloat16, uint4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint4*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<uint4* const*>(peer_ptrs_tensor.data_ptr<int64_t>()),
            push_counts.data_ptr<int>(),
            read_offsets.data_ptr<int>(),
            write_offsets.data_ptr<int>(),
            H_vec
        );
    } else if (input.dtype() == torch::kFloat32) {
        int H_vec = H / 4; // 4 fp32 = 16 bytes = float4
        uva_push_kernel_vec<float, float4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4* const*>(peer_ptrs_tensor.data_ptr<int64_t>()),
            push_counts.data_ptr<int>(),
            read_offsets.data_ptr<int>(),
            write_offsets.data_ptr<int>(),
            H_vec
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void gather_counts_kernel(
    int* const* __restrict__ peer_ptrs,
    int* __restrict__ count_matrix,
    int ep_size
) {
    int dst = blockIdx.x;
    int src = threadIdx.x;
    if (dst < ep_size && src < ep_size) {
        count_matrix[dst * ep_size + src] = peer_ptrs[dst][src];
    }
}

void gather_counts(
    std::vector<int64_t> peer_ptrs,
    torch::Tensor count_matrix
) {
    int ep_size = peer_ptrs.size();
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(count_matrix.device());
    torch::Tensor peer_ptrs_tensor = torch::empty({ep_size}, options);
    peer_ptrs_tensor.copy_(torch::tensor(peer_ptrs, torch::kInt64));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_counts_kernel<<<ep_size, ep_size, 0, stream>>>(
        reinterpret_cast<int* const*>(peer_ptrs_tensor.data_ptr<int64_t>()),
        count_matrix.data_ptr<int>(),
        ep_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push", &uva_push, "UVA push fused permutation");
    m.def("gather_counts", &gather_counts, "Gather split sizes from peers");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_moe_ext", CUDA_SRC)
    return _ext


_EP_SUBGROUP_CACHE: dict[tuple[int, int], None | list] = {}

def _resolve_ep_group_for_narrow_moe(num_experts: int) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    ws = dist.get_world_size()
    rank = dist.get_rank()
    key = (ws, num_experts)
    if key not in _EP_SUBGROUP_CACHE:
        if num_experts >= ws:
            _EP_SUBGROUP_CACHE[key] = None
        elif ws % num_experts != 0:
            raise ValueError(f"narrow EP requires world_size ({ws}) % num_experts ({num_experts}) == 0")
        else:
            groups = []
            for r in range(ws // num_experts):
                ranks = list(range(r * num_experts, (r + 1) * num_experts))
                groups.append(dist.new_group(ranks))
            _EP_SUBGROUP_CACHE[key] = groups
    entry = _EP_SUBGROUP_CACHE[key]
    if entry is None:
        return dist.group.WORLD
    return entry[rank // num_experts]


_symm_cache = {}

def _get_symm_state(max_tokens, hidden_dim, ep_size, dtype, device, group):
    key = (max_tokens, hidden_dim, ep_size, dtype, group)
    if key in _symm_cache:
        return _symm_cache[key]

    counts_buf = symm_mem.empty((ep_size,), dtype=torch.int32, device=device)
    hdl_counts = symm_mem.rendezvous(counts_buf, group)

    fwd_buf = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    hdl_fwd = symm_mem.rendezvous(fwd_buf, group)

    bwd_buf = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    hdl_bwd = symm_mem.rendezvous(bwd_buf, group)

    state = {
        "counts_buf": counts_buf,
        "hdl_counts": hdl_counts,
        "fwd_buf": fwd_buf,
        "hdl_fwd": hdl_fwd,
        "bwd_buf": bwd_buf,
        "hdl_bwd": hdl_bwd,
        "peer_counts_ptrs": [int(ptr) for ptr in hdl_counts.buffer_ptrs],
        "peer_fwd_ptrs": [int(ptr) for ptr in hdl_fwd.buffer_ptrs],
        "peer_bwd_ptrs": [int(ptr) for ptr in hdl_bwd.buffer_ptrs],
    }
    _symm_cache[key] = state
    return state


def get_push_params(cm, rank, ep_size, is_pattern_a, device):
    counts = []
    read_offsets = []
    write_offsets = []
    if is_pattern_a:
        for D in range(ep_size):
            counts.append(cm[rank][D])
            read_offsets.append(sum(cm[rank][:D]))
            write_offsets.append(sum(cm[s][D] for s in range(rank)))
        expected_recv = sum(cm[s][rank] for s in range(ep_size))
    else:
        for S in range(ep_size):
            counts.append(cm[S][rank])
            read_offsets.append(sum(cm[s][rank] for s in range(S)))
            write_offsets.append(sum(cm[S][d] for d in range(rank)))
        expected_recv = sum(cm[rank][d] for d in range(ep_size))
    
    return (
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(read_offsets, dtype=torch.int32, device=device),
        torch.tensor(write_offsets, dtype=torch.int32, device=device),
        expected_recv
    )


class UvaAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, cm, ep_rank, ep_size,
                peer_fwd_ptrs, peer_bwd_ptrs, my_fwd_buf, my_bwd_buf,
                hdl_fwd_sync, hdl_bwd_sync, is_pattern_a):
        
        ctx.cm = cm
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size
        ctx.peer_fwd_ptrs = peer_fwd_ptrs
        ctx.peer_bwd_ptrs = peer_bwd_ptrs
        ctx.my_fwd_buf = my_fwd_buf
        ctx.my_bwd_buf = my_bwd_buf
        ctx.hdl_fwd_sync = hdl_fwd_sync
        ctx.hdl_bwd_sync = hdl_bwd_sync
        ctx.is_pattern_a = is_pattern_a

        counts, read_offsets, write_offsets, expected_recv = get_push_params(
            cm, ep_rank, ep_size, is_pattern_a, input_tensor.device
        )

        push_ptrs = peer_fwd_ptrs if is_pattern_a else peer_bwd_ptrs
        recv_buf = my_fwd_buf if is_pattern_a else my_bwd_buf

        _get_ext().uva_push(
            input_tensor.contiguous(), push_ptrs, counts, read_offsets, write_offsets, input_tensor.size(-1)
        )
        torch.cuda.current_stream().synchronize()
        hdl_fwd_sync.barrier(channel=0)

        return recv_buf[:expected_recv].clone()

    @staticmethod
    def backward(ctx, grad_output):
        is_pattern_a = not ctx.is_pattern_a
        grad_output = grad_output.contiguous()

        counts, read_offsets, write_offsets, expected_recv = get_push_params(
            ctx.cm, ctx.ep_rank, ctx.ep_size, is_pattern_a, grad_output.device
        )

        push_ptrs = ctx.peer_fwd_ptrs if is_pattern_a else ctx.peer_bwd_ptrs
        recv_buf = ctx.my_fwd_buf if is_pattern_a else ctx.my_bwd_buf

        _get_ext().uva_push(
            grad_output, push_ptrs, counts, read_offsets, write_offsets, grad_output.size(-1)
        )
        torch.cuda.current_stream().synchronize()
        ctx.hdl_bwd_sync.barrier(channel=0)

        grad_input = recv_buf[:expected_recv].clone()
        return grad_input, None, None, None, None, None, None, None, None, None, None


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
    _get_ext()
    if group is None:
        group = _resolve_ep_group_for_narrow_moe(num_experts)

    ep_rank = dist.get_rank(group)
    ep_size = dist.get_world_size(group)
    device = hidden_states.device
    dtype = hidden_states.dtype
    hidden_dim = hidden_states.size(-1)
    
    hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states_flat.size(0)

    # 1. Routing logic
    router_logits = torch.nn.functional.linear(hidden_states_flat, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    my_send_counts = expert_mask.sum(dim=(1, 2)).to(torch.int32)
    max_tokens = num_tokens * top_k * ep_size
    symm_state = _get_symm_state(max_tokens, hidden_dim, ep_size, dtype, device, group)

    # 2. Gather routing count matrix securely over Symmetric Memory (bypassing NCCL)
    symm_state["counts_buf"].copy_(my_send_counts)
    torch.cuda.current_stream().synchronize()
    symm_state["hdl_counts"].barrier(channel=0)

    count_matrix = torch.empty((ep_size, ep_size), dtype=torch.int32, device=device)
    _get_ext().gather_counts(symm_state["peer_counts_ptrs"], count_matrix)
    torch.cuda.current_stream().synchronize()
    cm = count_matrix.cpu().tolist()

    # 3. Permute local items
    routing_map = expert_mask.sum(dim=1).bool()
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(ep_size, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    local_permuted_hidden_states = hidden_states_flat.index_select(0, sorted_indices)

    # 4. Phase A: Local -> Remote Expert UVA P2P All2All
    my_expert_input = UvaAllToAll.apply(
        local_permuted_hidden_states, cm, ep_rank, ep_size,
        symm_state["peer_fwd_ptrs"], symm_state["peer_bwd_ptrs"],
        symm_state["fwd_buf"], symm_state["bwd_buf"],
        symm_state["hdl_fwd"], symm_state["hdl_bwd"], True
    )

    # 5. Execute Sub-expert PyTorch compute
    expert_outputs = expert_forward(my_expert_input, gate_proj, up_proj, down_proj)

    # 6. Phase B: Remote Expert -> Local Sender UVA P2P All2All
    unpermute_outputs = UvaAllToAll.apply(
        expert_outputs, cm, ep_rank, ep_size,
        symm_state["peer_fwd_ptrs"], symm_state["peer_bwd_ptrs"],
        symm_state["fwd_buf"], symm_state["bwd_buf"],
        symm_state["hdl_bwd"], symm_state["hdl_fwd"], False
    )

    # 7. Unpermute weighting
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=dtype, device=device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    tokens_weight = weights_idx.T.contiguous().masked_select(routing_map)
    unpermute_outputs = unpermute_outputs * tokens_weight.unsqueeze(-1)

    out = torch.zeros_like(hidden_states_flat)
    expanded_mapping = sorted_indices.unsqueeze(1).expand(-1, hidden_dim)
    out.scatter_add_(0, expanded_mapping, unpermute_outputs)

    return out.view_as(hidden_states)


def main() -> None:
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    num_experts = 8
    top_k = 2
    hidden_dim = 64
    intermediate_dim = 128
    batch, seq = 2, 16
    num_tokens = batch * seq
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"

    torch.manual_seed(42 + rank)
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    gate_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32)
    gate_bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim).to(device)
    up_proj = torch.nn.Linear(hidden_dim, intermediate_dim).to(device)
    down_proj = torch.nn.Linear(intermediate_dim, hidden_dim).to(device)

    out = solution(
        hidden_states,
        gate_weight,
        gate_bias,
        gate_proj,
        up_proj,
        down_proj,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
    )
    loss = out.sum()
    loss.backward()

    if rank == 0:
        print("MoE e2e forward + backward OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()