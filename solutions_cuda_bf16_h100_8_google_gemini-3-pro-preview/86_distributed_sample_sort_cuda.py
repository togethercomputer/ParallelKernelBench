from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void extract_samples_kernel(
    const __nv_bfloat16* __restrict__ sorted_local,
    int64_t local_n,
    int64_t* __restrict__ my_meta,
    int sort_rank,
    int active_count)
{
    int i = threadIdx.x;
    if (i < active_count) {
        float val = __int_as_float(0x7f800000); // +inf
        int s_rank = -1;
        int s_pos = -1;
        if (sort_rank >= 0 && local_n > 0) {
            int valid_count = min(active_count, (int)local_n);
            if (i < valid_count) {
                int64_t pos;
                if (active_count < local_n) {
                    pos = ((int64_t)(i + 1) * local_n) / active_count - 1;
                } else {
                    pos = i;
                }
                val = __bfloat162float(sorted_local[pos]);
                s_rank = sort_rank;
                s_pos = pos;
            }
        }
        my_meta[10 + i*3 + 0] = __float_as_int(val);
        my_meta[10 + i*3 + 1] = s_rank;
        my_meta[10 + i*3 + 2] = s_pos;
    }
}

__global__ void compute_boundaries_kernel(
    const __nv_bfloat16* __restrict__ sorted_local,
    int64_t local_n,
    const uint64_t* __restrict__ meta_ptrs,
    int world_size,
    int rank,
    int sort_rank,
    int active_count,
    const int* __restrict__ active_ranks)
{
    if (threadIdx.x != 0) return;
    
    float s_vals[64];
    int s_ranks[64];
    int s_poses[64];
    int s_count = 0;
    
    // Gather all samples from active peers
    for (int r = 0; r < active_count; ++r) {
        int peer_rank = active_ranks[r];
        int64_t* peer_meta = (int64_t*)meta_ptrs[peer_rank];
        for (int i = 0; i < active_count; ++i) {
            int s_r = peer_meta[10 + i*3 + 1];
            if (s_r >= 0) {
                s_vals[s_count] = __int_as_float(peer_meta[10 + i*3 + 0]);
                s_ranks[s_count] = s_r;
                s_poses[s_count] = peer_meta[10 + i*3 + 2];
                s_count++;
            }
        }
    }
    
    // Sort samples
    for (int i = 1; i < s_count; ++i) {
        float v = s_vals[i];
        int sr = s_ranks[i];
        int sp = s_poses[i];
        int j = i - 1;
        while (j >= 0) {
            bool swap = false;
            if (s_vals[j] > v) swap = true;
            else if (s_vals[j] == v && s_ranks[j] > sr) swap = true;
            else if (s_vals[j] == v && s_ranks[j] == sr && s_poses[j] > sp) swap = true;
            
            if (swap) {
                s_vals[j+1] = s_vals[j];
                s_ranks[j+1] = s_ranks[j];
                s_poses[j+1] = s_poses[j];
                j--;
            } else {
                break;
            }
        }
        s_vals[j+1] = v;
        s_ranks[j+1] = sr;
        s_poses[j+1] = sp;
    }
    
    // Pick splitters
    float split_vals[8];
    int split_ranks[8];
    int split_poses[8];
    for (int k = 0; k < active_count - 1; ++k) {
        int index = (k + 1) * s_count / active_count - 1;
        if (index < 0) index = 0;
        if (index >= s_count) index = s_count - 1;
        split_vals[k] = s_vals[index];
        split_ranks[k] = s_ranks[index];
        split_poses[k] = s_poses[index];
    }
    
    // Binary search boundaries
    int64_t boundaries[9];
    boundaries[0] = 0;
    boundaries[active_count] = local_n;
    
    for (int k = 0; k < active_count - 1; ++k) {
        float val = split_vals[k];
        int s_r = split_ranks[k];
        int s_p = split_poses[k];
        
        int64_t low = 0;
        int64_t high = local_n;
        if (sort_rank > s_r) {
            while (low < high) {
                int64_t mid = low + (high - low) / 2;
                if (__bfloat162float(sorted_local[mid]) < val) low = mid + 1;
                else high = mid;
            }
        } else if (sort_rank < s_r) {
            while (low < high) {
                int64_t mid = low + (high - low) / 2;
                if (__bfloat162float(sorted_local[mid]) <= val) low = mid + 1;
                else high = mid;
            }
        } else {
            low = s_p + 1;
        }
        boundaries[k + 1] = low;
    }
    
    // Monotonicity fix
    for (int k = 1; k <= active_count; ++k) {
        if (boundaries[k] < boundaries[k-1]) boundaries[k] = boundaries[k-1];
        if (boundaries[k] > local_n) boundaries[k] = local_n;
    }
    
    // Save boundaries internally
    int64_t* my_meta = (int64_t*)meta_ptrs[rank];
    for (int k = 0; k <= active_count; ++k) {
        my_meta[200 + k] = boundaries[k];
    }
    
    // Push send counts
    for (int k = 0; k < active_count; ++k) {
        int dest_rank = active_ranks[k];
        int64_t count = boundaries[k+1] - boundaries[k];
        int64_t* dest_meta = (int64_t*)meta_ptrs[dest_rank];
        dest_meta[400 + rank] = count;
    }
}

__global__ void compute_recv_offsets_kernel(
    int64_t* __restrict__ my_meta,
    int world_size)
{
    if (threadIdx.x != 0) return;
    int64_t offset = 0;
    for (int r = 0; r < world_size; ++r) {
        my_meta[500 + r] = offset;
        offset += my_meta[400 + r];
    }
    my_meta[1] = offset; // Store total_recv
}

__global__ void gather_merged_sizes_kernel(
    const uint64_t* __restrict__ meta_ptrs,
    int64_t* __restrict__ my_meta,
    int world_size)
{
    if (threadIdx.x != 0) return;
    for (int r = 0; r < world_size; ++r) {
        my_meta[800 + r] = ((int64_t*)meta_ptrs[r])[1];
    }
}

__global__ void push_a2a_payload_kernel(
    const __nv_bfloat16* __restrict__ sorted_local,
    const uint64_t* __restrict__ a2a_ptrs,
    const uint64_t* __restrict__ meta_ptrs,
    int rank,
    int active_count,
    const int* __restrict__ active_ranks)
{
    int bucket = blockIdx.y;
    int dest_rank = active_ranks[bucket];
    
    int64_t* my_meta = (int64_t*)meta_ptrs[rank];
    int64_t bucket_start = my_meta[200 + bucket];
    int64_t bucket_end = my_meta[200 + bucket + 1];
    int64_t count = bucket_end - bucket_start;
    
    int64_t dest_offset = ((int64_t*)meta_ptrs[dest_rank])[500 + rank];
    __nv_bfloat16* dest_buf = (__nv_bfloat16*)a2a_ptrs[dest_rank];
    
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dest_buf[dest_offset + i] = sorted_local[bucket_start + i];
    }
}

__global__ void push_final_kernel(
    const __nv_bfloat16* __restrict__ merged,
    const uint64_t* __restrict__ final_ptrs,
    int64_t bucket_start,
    int64_t bucket_end,
    int64_t target_start,
    int64_t target_end,
    int dest_rank)
{
    int64_t start = max(bucket_start, target_start);
    int64_t end = min(bucket_end, target_end);
    if (start >= end) return;
    
    int64_t count = end - start;
    int64_t src_offset = start - bucket_start;
    int64_t dst_offset = start - target_start;
    
    __nv_bfloat16* dest_buf = (__nv_bfloat16*)final_ptrs[dest_rank];
    
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dest_buf[dst_offset + i] = merged[src_offset + i];
    }
}

void extract_samples(torch::Tensor sorted_local, int64_t local_n, torch::Tensor my_meta, int sort_rank, int active_count) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    extract_samples_kernel<<<1, 32, 0, stream>>>(
        (__nv_bfloat16*)sorted_local.data_ptr<at::BFloat16>(), local_n, my_meta.data_ptr<int64_t>(), sort_rank, active_count);
}

void compute_boundaries(torch::Tensor sorted_local, int64_t local_n, torch::Tensor meta_ptrs, int world_size, int rank, int sort_rank, int active_count, torch::Tensor active_ranks) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    compute_boundaries_kernel<<<1, 32, 0, stream>>>(
        (__nv_bfloat16*)sorted_local.data_ptr<at::BFloat16>(), local_n, (const uint64_t*)meta_ptrs.data_ptr<int64_t>(),
        world_size, rank, sort_rank, active_count, active_ranks.data_ptr<int>());
}

void compute_recv_offsets(torch::Tensor my_meta, int world_size) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    compute_recv_offsets_kernel<<<1, 32, 0, stream>>>(my_meta.data_ptr<int64_t>(), world_size);
}

void gather_merged_sizes(torch::Tensor meta_ptrs, torch::Tensor my_meta, int world_size) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    gather_merged_sizes_kernel<<<1, 32, 0, stream>>>((const uint64_t*)meta_ptrs.data_ptr<int64_t>(), my_meta.data_ptr<int64_t>(), world_size);
}

void push_a2a_payload(torch::Tensor sorted_local, torch::Tensor a2a_ptrs, torch::Tensor meta_ptrs, int rank, int active_count, torch::Tensor active_ranks) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(128, active_count);
    dim3 block(256);
    push_a2a_payload_kernel<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)sorted_local.data_ptr<at::BFloat16>(), (const uint64_t*)a2a_ptrs.data_ptr<int64_t>(),
        (const uint64_t*)meta_ptrs.data_ptr<int64_t>(), rank, active_count, active_ranks.data_ptr<int>());
}

void push_final(torch::Tensor merged, torch::Tensor final_ptrs, int64_t bucket_start, int64_t bucket_end, int64_t target_start, int64_t target_end, int dest_rank) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    push_final_kernel<<<128, 256, 0, stream>>>(
        (__nv_bfloat16*)merged.data_ptr<at::BFloat16>(), (const uint64_t*)final_ptrs.data_ptr<int64_t>(),
        bucket_start, bucket_end, target_start, target_end, dest_rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_samples", &extract_samples);
    m.def("compute_boundaries", &compute_boundaries);
    m.def("compute_recv_offsets", &compute_recv_offsets);
    m.def("gather_merged_sizes", &gather_merged_sizes);
    m.def("push_a2a_payload", &push_a2a_payload);
    m.def("push_final", &push_final);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_sample_sort_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def get_symm_buffer(name: str, min_size: int, dtype: torch.dtype, device: torch.device):
    if name in _symm_cache:
        buf, hdl = _symm_cache[name]
        if buf.numel() >= min_size and buf.dtype == dtype:
            return buf, hdl
    alloc_size = max(min_size, 1024 * 1024) if name != "meta" else min_size
    buf = symm_mem.empty(alloc_size, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[name] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(local_shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = local_shard.device
    local_n = local_shard.numel()

    # Initial sizes exchange via tight gather buffer
    size_t = torch.tensor([local_n], dtype=torch.long, device=device)
    gathered_sizes_t = torch.empty((world_size,), dtype=torch.long, device=device)
    dist.all_gather_into_tensor(gathered_sizes_t, size_t, group=group)
    sizes = gathered_sizes_t.cpu().tolist()

    total_N = sum(sizes)
    if total_N == 0:
        return local_shard.new_empty(0)

    active_ranks = [i for i, s in enumerate(sizes) if s > 0]
    sort_rank = active_ranks.index(rank) if rank in active_ranks else -1
    active_count = len(active_ranks)
    
    # Pre-allocate fully overlapping max-span device-side buffers
    meta_buf, meta_hdl = get_symm_buffer("meta", 1024, torch.int64, device)
    a2a_buf, a2a_hdl = get_symm_buffer("a2a", total_N, torch.bfloat16, device)
    
    base = total_N // world_size
    extra = total_N % world_size
    my_target_size = base + (1 if rank < extra else 0)
    final_buf, final_hdl = get_symm_buffer("final", base + 1, torch.bfloat16, device)

    ext = _get_ext()
    sorted_local = local_shard.sort().values
    active_ranks_t = torch.tensor(active_ranks, dtype=torch.int32, device=device)
    meta_ptrs = torch.tensor(meta_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    a2a_ptrs = torch.tensor(a2a_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    final_ptrs = torch.tensor(final_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    # Initialize / synchronize state buffers for deterministic offsets and exchanges
    meta_buf.zero_()
    dist.barrier(group)

    ext.extract_samples(sorted_local, local_n, meta_buf, sort_rank, active_count)
    dist.barrier(group)

    ext.compute_boundaries(sorted_local, local_n, meta_ptrs, world_size, rank, sort_rank, active_count, active_ranks_t)
    dist.barrier(group)

    ext.compute_recv_offsets(meta_buf, world_size)
    dist.barrier(group)

    # Overlap exact destination footprint sizes retrieval with asynchronous payload exchange
    ext.gather_merged_sizes(meta_ptrs, meta_buf, world_size)
    if active_count > 0 and local_n > 0:
        ext.push_a2a_payload(sorted_local, a2a_ptrs, meta_ptrs, rank, active_count, active_ranks_t)
    
    dist.barrier(group)
    meta_cpu = meta_buf.cpu()
    total_recv = int(meta_cpu[1].item())
    merged_sizes = meta_cpu[800 : 800 + world_size].tolist()

    # Intermediate variable sizes merged sort
    if total_recv > 0:
        merged = a2a_buf[:total_recv].sort().values
    else:
        merged = torch.empty(0, dtype=torch.bfloat16, device=device)

    bucket_start = sum(merged_sizes[:rank])
    bucket_end = bucket_start + merged.numel()

    # Exact redistribution kernel sweeps exact offset bounds concurrently directly via symmetric P2P writes
    for dest in range(world_size):
        target_start = dest * base + min(dest, extra)
        target_end = target_start + base + (1 if dest < extra else 0)
        ext.push_final(merged, final_ptrs, bucket_start, bucket_end, target_start, target_end, dest)

    dist.barrier(group)
    return final_buf[:my_target_size].clone()