import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void gather_peer_queries_kernel(
    const scalar_t* __restrict__ peer_ptr,
    scalar_t* __restrict__ local_buf,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        local_buf[i] = peer_ptr[i];
    }
}

template <typename scalar_t>
__global__ void scatter_peer_topk_kernel(
    const scalar_t* __restrict__ local_sims,
    const int64_t* __restrict__ local_labels,
    scalar_t* __restrict__ peer_sims_buf,
    int64_t* __restrict__ peer_labels_buf,
    int64_t Q_peer,
    int64_t K,
    int my_rank,
    int64_t max_Q
) {
    int64_t numel = Q_peer * K;
    int64_t threads = blockDim.x * gridDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int64_t i = idx; i < numel; i += threads) {
        int64_t row = i / K;
        int64_t col = i % K;
        
        int64_t dst_idx = (my_rank * max_Q + row) * K + col;
        peer_sims_buf[dst_idx] = local_sims[i];
        peer_labels_buf[dst_idx] = local_labels[i];
    }
}

void gather_peer_queries(
    int64_t peer_ptr_val,
    torch::Tensor local_buf,
    int64_t numel
) {
    if (numel == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = std::min<int64_t>((int64_t)65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_buf.scalar_type(), "gather_peer", [&] {
        const scalar_t* peer_ptr = reinterpret_cast<const scalar_t*>(peer_ptr_val);
        gather_peer_queries_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            peer_ptr,
            local_buf.data_ptr<scalar_t>(),
            numel
        );
    });
}

void scatter_peer_topk(
    torch::Tensor local_sims,
    torch::Tensor local_labels,
    int64_t peer_sim_ptr_val,
    int64_t peer_label_ptr_val,
    int64_t Q_peer,
    int64_t K,
    int my_rank,
    int64_t max_Q
) {
    if (Q_peer == 0 || K == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t numel = Q_peer * K;
    int threads = 256;
    int blocks = std::min<int64_t>((int64_t)65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_sims.scalar_type(), "scatter_peer", [&] {
        scalar_t* peer_sims_buf = reinterpret_cast<scalar_t*>(peer_sim_ptr_val);
        int64_t* peer_labels_buf = reinterpret_cast<int64_t*>(peer_label_ptr_val);
        
        scatter_peer_topk_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            local_sims.data_ptr<scalar_t>(),
            local_labels.data_ptr<int64_t>(),
            peer_sims_buf,
            peer_labels_buf,
            Q_peer,
            K,
            my_rank,
            max_Q
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_peer_queries", &gather_peer_queries, "Gather peer queries");
    m.def("scatter_peer_topk", &scatter_peer_topk, "Scatter peer topk");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_knn_overlap", CUDA_SRC)
    return _ext

_symm_cache = None
def _get_symm_state(max_Q, D, max_k, w, dtype_q, dtype_l, device, group):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if (c["max_Q"] >= max_Q and c["D"] == D and 
            c["max_k"] == max_k and c["w"] == w and 
            c["dtype_q"] == dtype_q and c["dtype_l"] == dtype_l and
            c["group"] == group):
            return c["sq"], c["hq"], c["ss"], c["hs"], c["sl"], c["hl"], c["max_Q"]

    alloc_max_Q = max(max_Q, 1)
    sq = symm_mem.empty((alloc_max_Q, D), dtype=dtype_q, device=device)
    hq = symm_mem.rendezvous(sq, group)
    
    ss = symm_mem.empty((w, alloc_max_Q, max_k), dtype=dtype_q, device=device)
    hs = symm_mem.rendezvous(ss, group)
    
    sl = symm_mem.empty((w, alloc_max_Q, max_k), dtype=dtype_l, device=device)
    hl = symm_mem.rendezvous(sl, group)
    
    _symm_cache = {
        "max_Q": alloc_max_Q, "D": D, "max_k": max_k, "w": w,
        "dtype_q": dtype_q, "dtype_l": dtype_l, "group": group,
        "sq": sq, "hq": hq, "ss": ss, "hs": hs, "sl": sl, "hl": hl
    }
    return sq, hq, ss, hs, sl, hl, alloc_max_Q


_streams = None
def _get_streams(world_size):
    global _streams
    if _streams is None or len(_streams) < world_size:
        _streams = [torch.cuda.Stream() for _ in range(world_size)]
    return _streams[:world_size]


@torch.no_grad()
def solution(
    test_features_rank: torch.Tensor,
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    max_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    device = test_features_rank.device
    
    if max_k > train_features_rank_T.shape[1]:
        raise ValueError("max_k must not exceed the local train shard size")

    # Serialize compilation so only rank 0 builds
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)

    Q_r = test_features_rank.shape[0]
    D = test_features_rank.shape[1]

    # Quick All-gather to discover peers' query sizes
    q_tensor = torch.tensor([Q_r], dtype=torch.int64, device=device)
    q_sizes_list = [torch.empty_like(q_tensor) for _ in range(world_size)]
    dist.all_gather(q_sizes_list, q_tensor, group=group)
    q_sizes = torch.cat(q_sizes_list).cpu().tolist()
    max_Q = max(q_sizes)

    # Allocate / Reuse symmetric memory
    sq, hq, ss, hs, sl, hl, alloc_max_Q = _get_symm_state(
        max_Q, D, max_k, world_size, 
        test_features_rank.dtype, train_labels_rank.dtype, 
        device, group
    )
    
    ptr_q = hq.buffer_ptrs
    ptr_s = hs.buffer_ptrs
    ptr_l = hl.buffer_ptrs

    # Expose local queries into symmetric memory 
    if Q_r > 0:
        sq[:Q_r, :].copy_(test_features_rank)
    dist.barrier(group=group)

    current_stream = torch.cuda.current_stream()
    streams = _get_streams(world_size)

    # Launch fully pipelined peer GEMMs and Top-K using separate streams
    for peer in range(world_size):
        Q_peer = q_sizes[peer]
        stream = streams[peer]
        stream.wait_stream(current_stream)
        
        with torch.cuda.stream(stream):
            if Q_peer > 0:
                local_peer_queries = torch.empty((Q_peer, D), dtype=test_features_rank.dtype, device=device)
                
                _get_ext().gather_peer_queries(int(ptr_q[peer]), local_peer_queries, Q_peer * D)
                peer_sims = torch.matmul(local_peer_queries, train_features_rank_T)
                
                peer_topk_sims, indices = peer_sims.topk(max_k, dim=1, largest=True, sorted=True)
                peer_topk_labels = torch.gather(train_labels_rank.expand(Q_peer, -1), 1, indices)
                
                _get_ext().scatter_peer_topk(
                    peer_topk_sims, peer_topk_labels,
                    int(ptr_s[peer]), int(ptr_l[peer]),
                    Q_peer, max_k, rank, alloc_max_Q
                )

    # Re-join concurrent streams before final Top-K
    for stream in streams[:world_size]:
        current_stream.wait_stream(stream)

    dist.barrier(group=group)

    # Final Merge phase: Top-K across gathered metrics 
    if Q_r == 0:
        return torch.empty((0, max_k), dtype=test_features_rank.dtype, device=device), \
               torch.empty((0, max_k), dtype=train_labels_rank.dtype, device=device)

    # Sub-slice perfectly bound items written by peers into our memory
    valid_sims = ss[:, :Q_r, :]
    valid_labels = sl[:, :Q_r, :]

    # Concatenate W sets of Top-K outcomes logically equivalent to all peers' contributions
    valid_sims = valid_sims.permute(1, 0, 2).reshape(Q_r, world_size * max_k)
    valid_labels = valid_labels.permute(1, 0, 2).reshape(Q_r, world_size * max_k)

    final_topk_sims, final_indices = valid_sims.topk(max_k, dim=1, largest=True, sorted=True)
    final_topk_labels = torch.gather(valid_labels, 1, final_indices)

    return final_topk_sims, final_topk_labels