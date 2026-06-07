import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

__global__ void gather_and_init_kernel_2d(
    const long long* peer_ptrs_int64,
    float* __restrict__ out_logits,
    int* __restrict__ out_indices,
    int local_vocab,
    int world_size,
    int rank,
    int local_tokens,
    int start_token,
    int chunk_size
) {
    int chunk_token_idx = blockIdx.x;
    int vocab_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int V = world_size * local_vocab;
    
    if (vocab_idx < V) {
        int peer = vocab_idx / local_vocab;
        int peer_vocab_idx = vocab_idx % local_vocab;
        
        int global_token_idx = rank * local_tokens + start_token + chunk_token_idx;
        
        const __nv_bfloat16* peer_ptr = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs_int64[peer]);
        __nv_bfloat16 val = peer_ptr[global_token_idx * local_vocab + peer_vocab_idx];
        
        int out_idx = chunk_token_idx * V + vocab_idx;
        out_logits[out_idx] = __bfloat162float(val);
        out_indices[out_idx] = vocab_idx;
    }
}

__global__ void init_offsets_kernel(int* offsets, int num_segments, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_segments) {
        offsets[idx] = idx * V;
    }
}

__global__ void filter_and_logprob_kernel(
    const float* __restrict__ sorted_logits,
    const int* __restrict__ sorted_indices,
    const int64_t* __restrict__ target_local,
    float* __restrict__ out_logprobs,
    int V,
    int top_k,
    float top_p
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* token_logits = sorted_logits + token_idx * V;
    const int* token_indices = sorted_indices + token_idx * V;
    int target = target_local[token_idx];
    
    typedef cub::BlockReduce<float, 512> BlockReduce;
    typedef cub::BlockScan<float, 512> BlockScan;
    __shared__ union {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    } temp_storage;
    
    __shared__ float s_max_val;
    __shared__ float s_threshold_k;
    __shared__ float s_block_sum;
    __shared__ float s_scan_carry;
    __shared__ float s_target_val;
    __shared__ bool s_target_kept;
    
    if (tid == 0) {
        s_max_val = token_logits[V - 1]; // Sorted ascending, max is at the end
        s_threshold_k = -INFINITY;
        if (top_k > 0 && top_k <= V) {
            s_threshold_k = token_logits[V - top_k];
        }
        s_scan_carry = 0;
        s_target_val = -INFINITY;
        s_target_kept = false;
    }
    __syncthreads();
    
    float max_val = s_max_val;
    float threshold_k = s_threshold_k;
    
    // Pass 1: Sum exponentials for top-k elements
    float thread_sum = 0;
    for (int i = tid; i < V; i += blockDim.x) {
        float val = token_logits[i];
        if (val >= threshold_k) {
            thread_sum += expf(val - max_val);
        }
    }
    
    float sum_exp = BlockReduce(temp_storage.reduce).Sum(thread_sum);
    if (tid == 0) {
        s_block_sum = sum_exp;
    }
    __syncthreads();
    
    sum_exp = s_block_sum;
    
    // Pass 2: Block-level scan for top-p and final sum_exp
    float thread_final_sum = 0;
    
    for (int chunk = 0; chunk < V; chunk += blockDim.x) {
        int i = chunk + tid;
        float val = -INFINITY;
        float prob = 0;
        bool valid_k = false;
        int orig_idx = -1;
        
        if (i < V) {
            val = token_logits[i];
            orig_idx = token_indices[i];
            if (val >= threshold_k) {
                valid_k = true;
                prob = expf(val - max_val) / sum_exp;
            }
        }
        
        float chunk_cumsum;
        BlockScan(temp_storage.scan).InclusiveSum(prob, chunk_cumsum);
        __syncthreads();
        
        float global_cumsum = s_scan_carry + chunk_cumsum;
        
        if (i < V && valid_k) {
            bool keep_p = true;
            if (top_p < 1.0f) {
                keep_p = (global_cumsum > 1.0f - top_p) || (i == V - 1);
            }
            if (keep_p) {
                thread_final_sum += expf(val - max_val);
            }
            if (orig_idx == target) {
                if (keep_p) {
                    s_target_val = val;
                    s_target_kept = true;
                }
            }
        }
        
        if (tid == blockDim.x - 1) {
            s_scan_carry += chunk_cumsum;
        }
        __syncthreads();
    }
    
    float final_sum_exp = BlockReduce(temp_storage.reduce).Sum(thread_final_sum);
    __shared__ float s_final_sum_exp;
    if (tid == 0) {
        s_final_sum_exp = final_sum_exp;
    }
    __syncthreads();
    
    if (tid == 0) {
        if (s_target_kept) {
            out_logprobs[token_idx] = s_target_val - (max_val + logf(s_final_sum_exp));
        } else {
            out_logprobs[token_idx] = -INFINITY;
        }
    }
}

__global__ void gather_logprobs_kernel(
    const long long* peer_ptrs_int64,
    float* __restrict__ out,
    int local_tokens,
    int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * local_tokens;
    if (idx < total) {
        int peer = idx / local_tokens;
        int token_idx = idx % local_tokens;
        const float* peer_ptr = reinterpret_cast<const float*>(peer_ptrs_int64[peer]);
        out[idx] = peer_ptr[token_idx];
    }
}

void gather_and_init(
    torch::Tensor peer_ptrs,
    torch::Tensor out_logits,
    torch::Tensor out_indices,
    int local_vocab,
    int world_size,
    int rank,
    int local_tokens,
    int start_token,
    int chunk_size
) {
    const long long* d_peers = (const long long*)peer_ptrs.data_ptr<int64_t>();
    float* d_out_logits = out_logits.data_ptr<float>();
    int* d_out_indices = out_indices.data_ptr<int>();
    
    int V = world_size * local_vocab;
    dim3 block(256);
    dim3 grid(chunk_size, (V + 255)/256);
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    gather_and_init_kernel_2d<<<grid, block, 0, stream>>>(
        d_peers, d_out_logits, d_out_indices, local_vocab, world_size,
        rank, local_tokens, start_token, chunk_size
    );
}

void init_offsets(torch::Tensor offsets, int num_segments, int V) {
    int* d_offsets = offsets.data_ptr<int>();
    int threads = 256;
    int blocks = (num_segments + 1 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    init_offsets_kernel<<<blocks, threads, 0, stream>>>(d_offsets, num_segments, V);
}

void cub_segmented_sort(
    torch::Tensor keys_in,
    torch::Tensor keys_out,
    torch::Tensor vals_in,
    torch::Tensor vals_out,
    torch::Tensor offsets,
    int num_segments
) {
    size_t temp_storage_bytes = 0;
    float* d_keys_in = keys_in.data_ptr<float>();
    float* d_keys_out = keys_out.data_ptr<float>();
    int* d_vals_in = vals_in.data_ptr<int>();
    int* d_vals_out = vals_out.data_ptr<int>();
    int* d_offsets = offsets.data_ptr<int>();
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Dry-run (O(1) host overhead)
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        keys_in.numel(), num_segments, d_offsets, d_offsets + 1,
        0, sizeof(float)*8, stream
    );
    
    auto temp_storage = torch::empty({(long)temp_storage_bytes}, keys_in.options().dtype(torch::kUInt8));
    
    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        keys_in.numel(), num_segments, d_offsets, d_offsets + 1,
        0, sizeof(float)*8, stream
    );
}

void filter_and_logprob(
    torch::Tensor sorted_logits,
    torch::Tensor sorted_indices,
    torch::Tensor target_local,
    torch::Tensor out_logprobs,
    int V,
    int top_k,
    float top_p
) {
    int chunk_size = out_logprobs.numel();
    if (chunk_size == 0) return;
    
    dim3 block(512);
    dim3 grid(chunk_size);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    filter_and_logprob_kernel<<<grid, block, 0, stream>>>(
        sorted_logits.data_ptr<float>(),
        sorted_indices.data_ptr<int>(),
        target_local.data_ptr<int64_t>(),
        out_logprobs.data_ptr<float>(),
        V, top_k, top_p
    );
}

void gather_logprobs(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int local_tokens,
    int world_size
) {
    const long long* d_peers = (const long long*)peer_ptrs.data_ptr<int64_t>();
    float* d_out = out.data_ptr<float>();
    
    int total = world_size * local_tokens;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    gather_logprobs_kernel<<<blocks, threads, 0, stream>>>(
        d_peers, d_out, local_tokens, world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_and_init", &gather_and_init);
    m.def("init_offsets", &init_offsets);
    m.def("cub_segmented_sort", &cub_segmented_sort);
    m.def("filter_and_logprob", &filter_and_logprob);
    m.def("gather_logprobs", &gather_logprobs);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vp_logprob_topk_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(name, shape, dtype, device, group):
    key = (name, tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs)
    _symm_cache[key] = res
    return res

@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    if num_tokens % world_size != 0:
        raise ValueError(f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}")
    
    local_tokens = num_tokens // world_size
    V = world_size * local_vocab
    device = vocab_parallel_logits.device
    
    target_local = target.reshape(-1)[rank * local_tokens : (rank + 1) * local_tokens].contiguous()
    target_local = target_local.to(device=device, dtype=torch.int64)

    buf_logits, hdl_logits, ptrs_logits = _get_symm_state("logits", vocab_parallel_logits.shape, vocab_parallel_logits.dtype, device, tp_group)
    buf_logprobs, hdl_logprobs, ptrs_logprobs = _get_symm_state("logprobs", (local_tokens,), torch.float32, device, tp_group)
    
    buf_logits.copy_(vocab_parallel_logits)
    hdl_logits.barrier(channel=0)

    # 2-stage chunking over execution streams for overlapping execution & UVA reads
    num_chunks = 2
    if local_tokens < num_chunks or local_tokens % num_chunks != 0:
        num_chunks = 1
    chunk_size = local_tokens // num_chunks
    
    ext = _get_ext()
    if not hasattr(ext, 'streams'):
        ext.streams = [torch.cuda.Stream() for _ in range(num_chunks)]
    streams = ext.streams[:num_chunks]

    local_logits_keys = torch.empty((local_tokens, V), device=device, dtype=torch.float32)
    local_indices_vals = torch.empty((local_tokens, V), device=device, dtype=torch.int32)
    sorted_keys = torch.empty_like(local_logits_keys)
    sorted_vals = torch.empty_like(local_indices_vals)
    
    offsets = torch.empty(chunk_size + 1, device=device, dtype=torch.int32)
    ext.init_offsets(offsets, chunk_size, V)

    default_stream = torch.cuda.current_stream()
    
    for i in range(num_chunks):
        with torch.cuda.stream(streams[i]):
            streams[i].wait_stream(default_stream)
            start = i * chunk_size
            end = start + chunk_size
            
            keys_in = local_logits_keys[start:end]
            vals_in = local_indices_vals[start:end]
            keys_out = sorted_keys[start:end]
            vals_out = sorted_vals[start:end]
            tgt_in = target_local[start:end]
            lp_out = buf_logprobs[start:end]
            
            ext.gather_and_init(ptrs_logits, keys_in, vals_in, local_vocab, world_size, rank, local_tokens, start, chunk_size)
            ext.cub_segmented_sort(keys_in, keys_out, vals_in, vals_out, offsets, chunk_size)
            ext.filter_and_logprob(keys_out, vals_out, tgt_in, lp_out, V, top_k if top_k is not None else 0, top_p)

    for s in streams:
        default_stream.wait_stream(s)
        
    hdl_logprobs.barrier(channel=0)
    
    out_logprobs = torch.empty(world_size * local_tokens, device=device, dtype=torch.float32)
    ext.gather_logprobs(ptrs_logprobs, out_logprobs, local_tokens, world_size)
    
    return out_logprobs.reshape(batch, seq_len)