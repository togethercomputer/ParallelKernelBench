import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void scatter_counts_kernel(
    const int64_t* send_counts,
    const int64_t* dest_ptrs,
    int world_size,
    int my_rank
) {
    int dst_rank = blockIdx.x;
    int idx = threadIdx.x;
    if (idx < world_size && dst_rank < world_size) {
        int64_t* dst = (int64_t*)dest_ptrs[dst_rank];
        dst[my_rank * world_size + idx] = send_counts[idx];
    }
}

template<typename scalar_t>
__global__ void all_to_all_write_kernel(
    const scalar_t* send_data,
    const int64_t* send_offsets,
    const int64_t* send_counts,
    const int64_t* dest_ptrs,
    const int64_t* dest_offsets,
    int world_size
) {
    int dst_rank = blockIdx.y;
    int64_t count = send_counts[dst_rank];
    int64_t src_offset = send_offsets[dst_rank];
    int64_t dst_offset = dest_offsets[dst_rank];
    scalar_t* dst = (scalar_t*)dest_ptrs[dst_rank];
    
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[dst_offset + i] = send_data[src_offset + i];
    }
}

__global__ void compute_take_kernel(
    const int64_t* input_nodes,
    const int64_t* colptr,
    int64_t* counts,
    int k,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t v = input_nodes[idx];
        int64_t start = colptr[v];
        int64_t end = colptr[v + 1];
        int64_t deg = end - start;
        int64_t take = (k >= 0 && k < deg) ? k : deg;
        counts[idx] = take;
    }
}

__device__ uint32_t gcd(uint32_t a, uint32_t b) {
    while (b != 0) {
        uint32_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__global__ void sample_and_write_kernel(
    const int64_t* input_nodes,
    const int64_t* counts,
    const int64_t* counts_prefix_sum,
    const int64_t* colptr,
    const int64_t* row,
    const int64_t* dest_ptrs_nodes,
    const int64_t* dest_ptrs_edges,
    const int64_t* dest_offsets,
    const int64_t* req_recv_counts_prefix,
    int n,
    int world_size,
    bool replace,
    int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t v = input_nodes[idx];
        int64_t take = counts[idx];
        if (take == 0) return;
        
        int peer = 0;
        for (int p = 0; p < world_size; ++p) {
            if (idx >= req_recv_counts_prefix[p] && idx < req_recv_counts_prefix[p + 1]) {
                peer = p;
                break;
            }
        }
        
        int64_t start = colptr[v];
        int64_t end = colptr[v + 1];
        int64_t deg = end - start;
        
        int64_t local_offset = counts_prefix_sum[idx] - counts_prefix_sum[req_recv_counts_prefix[peer]];
        int64_t global_dest_offset = dest_offsets[peer] + local_offset;
        
        int64_t* dst_nodes = (int64_t*)dest_ptrs_nodes[peer];
        int64_t* dst_edges = (int64_t*)dest_ptrs_edges[peer];
        
        if (replace) {
            for (int64_t j = 0; j < take; ++j) {
                uint32_t hash = seed ^ (idx * 1337) ^ (j * 73);
                hash ^= hash >> 16;
                hash *= 0x85ebca6b;
                hash ^= hash >> 13;
                int64_t r = hash % deg;
                dst_nodes[global_dest_offset + j] = row[start + r];
                dst_edges[global_dest_offset + j] = start + r;
            }
        } else {
            if (take == deg) {
                for (int64_t j = 0; j < take; ++j) {
                    dst_nodes[global_dest_offset + j] = row[start + j];
                    dst_edges[global_dest_offset + j] = start + j;
                }
            } else {
                uint32_t hash = seed ^ (idx * 1337);
                uint32_t stride = (hash % (deg - 1)) + 1;
                while (gcd(stride, deg) != 1 && stride < deg) {
                    stride++;
                }
                if (stride >= deg) stride = 1;
                uint32_t start_r = (hash >> 4) % deg;
                
                for (int64_t j = 0; j < take; ++j) {
                    int64_t r = (start_r + j * stride) % deg;
                    dst_nodes[global_dest_offset + j] = row[start + r];
                    dst_edges[global_dest_offset + j] = start + r;
                }
            }
        }
    }
}

void scatter_counts(
    torch::Tensor send_counts,
    torch::Tensor dest_ptrs,
    int world_size,
    int my_rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_counts_kernel<<<world_size, world_size, 0, stream>>>(
        send_counts.data_ptr<int64_t>(),
        dest_ptrs.data_ptr<int64_t>(),
        world_size,
        my_rank
    );
}

void all_to_all_write(
    torch::Tensor send_data,
    torch::Tensor send_offsets,
    torch::Tensor send_counts,
    torch::Tensor dest_ptrs,
    torch::Tensor dest_offsets,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks_x = 32;
    dim3 grid(blocks_x, world_size);
    dim3 block(256);
    
    all_to_all_write_kernel<int64_t><<<grid, block, 0, stream>>>(
        send_data.data_ptr<int64_t>(),
        send_offsets.data_ptr<int64_t>(),
        send_counts.data_ptr<int64_t>(),
        dest_ptrs.data_ptr<int64_t>(),
        dest_offsets.data_ptr<int64_t>(),
        world_size
    );
}

void compute_take(
    torch::Tensor input_nodes,
    torch::Tensor colptr,
    torch::Tensor counts,
    int k,
    int n
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compute_take_kernel<<<blocks, threads, 0, stream>>>(
        input_nodes.data_ptr<int64_t>(),
        colptr.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        k, n
    );
}

void sample_and_write(
    torch::Tensor input_nodes,
    torch::Tensor counts,
    torch::Tensor counts_prefix_sum,
    torch::Tensor colptr,
    torch::Tensor row,
    torch::Tensor dest_ptrs_nodes,
    torch::Tensor dest_ptrs_edges,
    torch::Tensor dest_offsets,
    torch::Tensor req_recv_counts_prefix,
    int n,
    int world_size,
    bool replace,
    int seed
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sample_and_write_kernel<<<blocks, threads, 0, stream>>>(
        input_nodes.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        counts_prefix_sum.data_ptr<int64_t>(),
        colptr.data_ptr<int64_t>(),
        row.data_ptr<int64_t>(),
        dest_ptrs_nodes.data_ptr<int64_t>(),
        dest_ptrs_edges.data_ptr<int64_t>(),
        dest_offsets.data_ptr<int64_t>(),
        req_recv_counts_prefix.data_ptr<int64_t>(),
        n, world_size, replace, seed
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_counts", &scatter_counts);
    m.def("all_to_all_write", &all_to_all_write);
    m.def("compute_take", &compute_take);
    m.def("sample_and_write", &sample_and_write);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_gnn_sample_ext", CUDA_SRC)
    return _ext

class SymmMemAllocator:
    def __init__(self, group, device, world_size):
        self.group = group
        self.device = device
        self.world_size = world_size
        self.buffers = {}

    def get_buffer(self, name, min_size, dtype):
        if name not in self.buffers or self.buffers[name]['size'] < min_size:
            new_size = int(max(min_size * 1.2 + 1024, 4096))
            buf = symm_mem.empty(new_size, dtype=dtype, device=self.device)
            hdl = symm_mem.rendezvous(buf, self.group)
            ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.long, device=self.device)
            self.buffers[name] = {'buf': buf, 'hdl': hdl, 'size': new_size, 'ptrs': ptrs}
        return self.buffers[name]

class SymmAllToAll:
    def __init__(self, group, device):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = device
        
        self.counts_buf = symm_mem.empty((self.world_size, self.world_size), dtype=torch.long, device=device)
        self.counts_hdl = symm_mem.rendezvous(self.counts_buf, group)
        self.counts_ptrs = torch.tensor(self.counts_hdl.buffer_ptrs, dtype=torch.long, device=device)
        
        self.allocator = SymmMemAllocator(group, device, self.world_size)
        self.ext = _get_ext()

    def exchange_requests(self, send_nodes_list):
        send_counts = torch.tensor([x.numel() for x in send_nodes_list], dtype=torch.long, device=self.device)
        send_nodes = torch.cat(send_nodes_list) if send_nodes_list else torch.empty(0, dtype=torch.long, device=self.device)
        
        self.ext.scatter_counts(send_counts, self.counts_ptrs, self.world_size, self.rank)
        self.counts_hdl.barrier()
        
        send_matrix = self.counts_buf.clone()
        recv_counts = send_matrix[:, self.rank]
        total_recv = recv_counts.sum().item()
        dest_offsets = send_matrix[:self.rank, :].sum(dim=0)
        
        send_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), send_counts.cumsum(0)[:-1]])
        
        max_recv = send_matrix.sum(dim=0).max().item()
        buf_info = self.allocator.get_buffer("req_nodes", max_recv, torch.long)
        
        if send_nodes.numel() > 0:
            self.ext.all_to_all_write(send_nodes, send_offsets, send_counts, buf_info['ptrs'], dest_offsets, self.world_size)
        buf_info['hdl'].barrier()
        
        recv_nodes = buf_info['buf'][:total_recv].clone()
        return recv_nodes, send_counts, recv_counts, send_matrix

    def exchange_replies_fused(self, recv_nodes, sampled_counts, colptr, row, req_recv_counts, req_send_matrix, fanout, replace, seed):
        req_recv_prefix = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), req_recv_counts.cumsum(0)])
        counts_prefix = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), sampled_counts.cumsum(0)[:-1]]) if sampled_counts.numel() > 0 else torch.zeros(1, dtype=torch.long, device=self.device)
        
        send_node_counts = torch.empty(self.world_size, dtype=torch.long, device=self.device)
        for p in range(self.world_size):
            start = req_recv_prefix[p].item()
            end = req_recv_prefix[p+1].item()
            send_node_counts[p] = sampled_counts[start:end].sum() if end > start else 0
            
        send_count_counts = req_recv_counts
        
        self.ext.scatter_counts(send_node_counts, self.counts_ptrs, self.world_size, self.rank)
        self.counts_hdl.barrier()
        
        reply_node_matrix = self.counts_buf.clone()
        reply_node_counts = reply_node_matrix[:, self.rank]
        total_reply_nodes = reply_node_counts.sum().item()
        dest_node_offsets = reply_node_matrix[:self.rank, :].sum(dim=0)
        
        self.ext.scatter_counts(send_count_counts, self.counts_ptrs, self.world_size, self.rank)
        self.counts_hdl.barrier()
        
        reply_count_matrix = self.counts_buf.clone()
        reply_count_counts = reply_count_matrix[:, self.rank]
        total_reply_counts = reply_count_counts.sum().item()
        dest_count_offsets = reply_count_matrix[:self.rank, :].sum(dim=0)
        send_count_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), send_count_counts.cumsum(0)[:-1]])
        
        max_reply_nodes = reply_node_matrix.sum(dim=0).max().item()
        max_reply_counts = reply_count_matrix.sum(dim=0).max().item()
        
        buf_nodes = self.allocator.get_buffer("rep_nodes", max_reply_nodes, torch.long)
        buf_edges = self.allocator.get_buffer("rep_edges", max_reply_nodes, torch.long)
        buf_counts = self.allocator.get_buffer("rep_counts", max_reply_counts, torch.long)
        
        if total_reply_counts > 0 or send_count_counts.sum().item() > 0:
            if sampled_counts.numel() > 0:
                self.ext.all_to_all_write(sampled_counts, send_count_offsets, send_count_counts, buf_counts['ptrs'], dest_count_offsets, self.world_size)
            
        if recv_nodes.numel() > 0:
            self.ext.sample_and_write(
                recv_nodes, sampled_counts, counts_prefix, colptr, row, 
                buf_nodes['ptrs'], buf_edges['ptrs'], dest_node_offsets, req_recv_prefix, 
                recv_nodes.numel(), self.world_size, replace, seed
            )
            
        buf_nodes['hdl'].barrier()
        buf_edges['hdl'].barrier()
        buf_counts['hdl'].barrier()
        
        recv_rep_nodes = buf_nodes['buf'][:total_reply_nodes].clone()
        recv_rep_edges = buf_edges['buf'][:total_reply_nodes].clone()
        recv_rep_counts = buf_counts['buf'][:total_reply_counts].clone()
        
        return recv_rep_nodes, recv_rep_edges, recv_rep_counts

def _remove_duplicates(out_node: torch.Tensor, node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = node.numel()
    node_combined = torch.cat([node, out_node])
    _, idx = np.unique(node_combined.cpu().numpy(), return_index=True)
    idx = torch.from_numpy(idx).to(node.device).sort().values
    node = node_combined[idx]
    src = node[num_nodes:]
    return src, node

def _relabel_neighborhood(
    node: torch.Tensor,
    dst_with_dupl: torch.Tensor,
    node_with_dupl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if node_with_dupl.numel() == 0:
        return node.new_empty(0), node.new_empty(0)

    assoc = torch.full(
        (int(node.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=node.device,
    )
    assoc[node] = torch.arange(node.numel(), device=node.device)
    row = assoc[node_with_dupl]
    col = assoc[dst_with_dupl]
    return row, col

@torch.no_grad()
def solution(
    seed_nodes: torch.Tensor,
    fanouts: List[int],
    local_adj_row_ptr: torch.Tensor,
    local_adj_col: torch.Tensor,
    node_to_rank: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    device = seed_nodes.device
    rank = dist.get_rank(group)

    seed = seed_nodes.to(dtype=torch.long, device=device)
    src = seed.clone()
    node = src.clone()
    node_with_dupl = [seed.new_empty(0)]
    dst_with_dupl = [seed.new_empty(0)]
    edge = [seed.new_empty(0)]
    
    if not hasattr(solution, 'uva_manager'):
        solution.uva_manager = SymmAllToAll(group, device)
    uva = solution.uva_manager

    import random
    
    for fanout in fanouts:
        if src.numel() == 0:
            break

        partition_ids = node_to_rank[src].to(torch.long)
        partition_orders = torch.empty_like(partition_ids)
        send_nodes_list = []
        
        for r in range(world_size):
            pos = (partition_ids == r).nonzero(as_tuple=False).flatten()
            partition_orders[pos] = torch.arange(pos.numel(), dtype=torch.long, device=device)
            send_nodes_list.append(src[pos])

        recv_nodes, send_counts, req_recv_counts, req_send_matrix = uva.exchange_requests(send_nodes_list)

        sampled_counts = torch.empty_like(recv_nodes)
        if recv_nodes.numel() > 0:
            uva.ext.compute_take(recv_nodes, local_adj_row_ptr, sampled_counts, int(fanout), recv_nodes.numel())
            
        reply_nodes, reply_edges, reply_counts = uva.exchange_replies_fused(
            recv_nodes, sampled_counts, local_adj_row_ptr, local_adj_col, 
            req_recv_counts, req_send_matrix, int(fanout), replace, random.randint(0, 1000000)
        )

        rank_offsets = torch.cat(
            [send_counts.new_zeros(1), torch.cumsum(send_counts, dim=0)[:-1]]
        )
        grouped_index = rank_offsets[partition_ids] + partition_orders
        
        node_chunks = list(torch.split(reply_nodes, reply_counts.cpu().tolist()))
        edge_chunks = list(torch.split(reply_edges, reply_counts.cpu().tolist()))

        ordered_nodes = []
        ordered_edges = []
        ordered_dst = []
        for idx in grouped_index.tolist():
            ordered_nodes.append(node_chunks[idx])
            ordered_edges.append(edge_chunks[idx])
        for dst_node, count in zip(src, reply_counts[grouped_index]):
            ordered_dst.append(dst_node.repeat(int(count.item())))

        out_node = torch.cat(ordered_nodes) if ordered_nodes else seed.new_empty(0)
        out_edge = torch.cat(ordered_edges) if ordered_edges else seed.new_empty(0)
        out_dst = torch.cat(ordered_dst) if ordered_dst else seed.new_empty(0)
        
        if out_node.numel() == 0:
            break

        src, node = _remove_duplicates(out_node, node)
        node_with_dupl.append(out_node)
        dst_with_dupl.append(out_dst)
        edge.append(out_edge)

    node_dupl = torch.cat(node_with_dupl)
    dst_dupl = torch.cat(dst_with_dupl)
    row, col = _relabel_neighborhood(node, dst_dupl, node_dupl)
    return node, row, col, torch.cat(edge)