import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// 1. QKV Scatter Kernel
// Reads local chunked QKV [B, chunk_len, 3, num_heads, head_dim]
// Writes directly to peers' gathered buffer [3, B, S_full, num_heads_local, head_dim]
// ---------------------------------------------------------------------------
__global__ void qkv_alltoall_kernel_flat(
    const uint4* __restrict__ qkv,
    const uint64_t* __restrict__ dest_ptrs,
    int B, int chunk_len, int num_heads, int head_dim_vec,
    int rank, int world_size, int S_local, int start_s,
    int64_t total_vecs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;
    
    int d_v = idx % head_dim_vec;
    int64_t tmp = idx / head_dim_vec;
    int h = tmp % num_heads;
    tmp /= num_heads;
    int qkv_idx = tmp % 3;
    tmp /= 3;
    int s = tmp % chunk_len;
    int b = tmp / chunk_len;
    
    int num_heads_local = num_heads / world_size;
    int p = h / num_heads_local;
    int h_dst = h % num_heads_local;
    int S_full = S_local * world_size;
    int s_dst = rank * S_local + start_s + s;
    
    // dest shape: [3, B, S_full, num_heads_local, head_dim_vec]
    int64_t dest_offset = ((((int64_t)(qkv_idx * B + b) * S_full + s_dst) * num_heads_local) + h_dst) * head_dim_vec + d_v;
    
    uint4* peer_dest = (uint4*)dest_ptrs[p];
    peer_dest[dest_offset] = qkv[idx];
}

// ---------------------------------------------------------------------------
// 2. Attention Output Scatter Kernel
// Reads local attention output [B, S_full, num_heads_local, head_dim]
// Writes directly to peers' buffer [B, S_local, num_heads, head_dim]
// ---------------------------------------------------------------------------
__global__ void attn_out_alltoall_kernel_flat(
    const uint4* __restrict__ attn_out,
    const uint64_t* __restrict__ dest_ptrs,
    int B, int S_full, int num_heads_local, int head_dim_vec,
    int rank, int world_size,
    int S_local, int start_s_dst, int chunk_len, int64_t total_vecs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;
    
    int d_v = idx % head_dim_vec;
    int64_t tmp = idx / head_dim_vec;
    int h_local = tmp % num_heads_local;
    tmp /= num_heads_local;
    int s_in_chunk = tmp % chunk_len;
    tmp /= chunk_len;
    int p = tmp % world_size;
    int b = tmp / world_size;
    
    int s_src = p * S_local + start_s_dst + s_in_chunk;
    int64_t src_offset = (((int64_t)(b * S_full + s_src) * num_heads_local) + h_local) * head_dim_vec + d_v;
    
    int s_dst = start_s_dst + s_in_chunk;
    int h_dst = rank * num_heads_local + h_local;
    int num_heads = num_heads_local * world_size;
    
    // dest shape: [B, S_local, num_heads, head_dim_vec]
    int64_t dest_offset = (((int64_t)(b * S_local + s_dst) * num_heads) + h_dst) * head_dim_vec + d_v;
    
    uint4* peer_dest = (uint4*)dest_ptrs[p];
    peer_dest[dest_offset] = attn_out[src_offset];
}

// ---------------------------------------------------------------------------
// 3. Device-Side Synchronization Kernels
// ---------------------------------------------------------------------------
__global__ void signal_peers_kernel_relaxed(
    const uint64_t* __restrict__ signal_ptrs,
    int rank, int c, int world_size
) {
    int p = threadIdx.x;
    if (p < world_size) {
        volatile int* peer_signal = (volatile int*)signal_ptrs[p];
        peer_signal[c * world_size + rank] = 1;
    }
}

__global__ void wait_signal_kernel_relaxed(
    volatile int* __restrict__ my_signal,
    int c, int world_size
) {
    int p = threadIdx.x;
    if (p < world_size) {
        while (my_signal[c * world_size + p] == 0) {
            // Spin waiting for peer 'p' to flag completion for chunk 'c'
        }
    }
}

// ---------------------------------------------------------------------------
// C++ Bindings
// ---------------------------------------------------------------------------
void launch_qkv_alltoall(
    torch::Tensor qkv,
    torch::Tensor dest_ptrs,
    int B, int chunk_len, int num_heads, int head_dim,
    int rank, int world_size, int S_local, int start_s
) {
    TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8 for BF16 vectorization");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int head_dim_vec = head_dim / 8;
    int64_t total_vecs = (int64_t)B * chunk_len * 3 * num_heads * head_dim_vec;
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    qkv_alltoall_kernel_flat<<<blocks, threads, 0, stream>>>(
        (const uint4*)qkv.data_ptr(),
        (const uint64_t*)dest_ptrs.data_ptr(),
        B, chunk_len, num_heads, head_dim_vec,
        rank, world_size, S_local, start_s,
        total_vecs
    );
}

void launch_attn_out_alltoall(
    torch::Tensor attn_out,
    torch::Tensor dest_ptrs,
    int B, int S_full, int num_heads_local, int head_dim,
    int rank, int world_size, int S_local, int start_s_dst, int chunk_len
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int head_dim_vec = head_dim / 8;
    int64_t total_vecs = (int64_t)B * world_size * chunk_len * num_heads_local * head_dim_vec;
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    attn_out_alltoall_kernel_flat<<<blocks, threads, 0, stream>>>(
        (const uint4*)attn_out.data_ptr(),
        (const uint64_t*)dest_ptrs.data_ptr(),
        B, S_full, num_heads_local, head_dim_vec,
        rank, world_size, S_local, start_s_dst, chunk_len,
        total_vecs
    );
}

void launch_signal_peers(torch::Tensor signal_ptrs, int rank, int c, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    signal_peers_kernel_relaxed<<<1, 32, 0, stream>>>((const uint64_t*)signal_ptrs.data_ptr(), rank, c, world_size);
}

void launch_wait_signal(torch::Tensor my_signal, int c, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    wait_signal_kernel_relaxed<<<1, 32, 0, stream>>>((volatile int*)my_signal.data_ptr(), c, world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_qkv_alltoall", &launch_qkv_alltoall);
    m.def("launch_attn_out_alltoall", &launch_attn_out_alltoall);
    m.def("launch_signal_peers", &launch_signal_peers);
    m.def("launch_wait_signal", &launch_wait_signal);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_attn_overlap_ext", CUDA_SRC)
    return _ext

_resource_cache = {}
def _get_resources(B, S_local, S_full, num_heads, num_heads_local, head_dim, world_size, dtype, device):
    key = (B, S_local, S_full, num_heads, num_heads_local, head_dim, world_size, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
        
    qkv_gathered = symm_mem.empty((3, B, S_full, num_heads_local, head_dim), dtype=dtype, device=device)
    attn_gathered = symm_mem.empty((B, S_local, num_heads, head_dim), dtype=dtype, device=device)
    signal_pad = symm_mem.empty((4, world_size), dtype=torch.int32, device=device) # Supports up to 4 chunks
    
    hdl_qkv = symm_mem.rendezvous(qkv_gathered, dist.group.WORLD)
    hdl_attn = symm_mem.rendezvous(attn_gathered, dist.group.WORLD)
    hdl_signal = symm_mem.rendezvous(signal_pad, dist.group.WORLD)
    
    dest_ptrs_qkv = torch.tensor(hdl_qkv.buffer_ptrs, dtype=torch.int64, device=device)
    dest_ptrs_attn = torch.tensor(hdl_attn.buffer_ptrs, dtype=torch.int64, device=device)
    dest_ptrs_signal = torch.tensor(hdl_signal.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (qkv_gathered, attn_gathered, signal_pad, dest_ptrs_qkv, dest_ptrs_attn, dest_ptrs_signal)
    _resource_cache[key] = res
    return res

_streams = None
def _get_streams(n):
    global _streams
    if _streams is None:
        _streams = [torch.cuda.Stream() for _ in range(4)]
    return _streams[:n]

def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    """Minimal scaled dot-product attention logic. Kept identical for exact reference parity."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal and q.size(1) > 1:
        S = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    num_heads: int = 8,
    causal: bool = False,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    
    B, S_local, H = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads

    if world_size == 1:
        qkv = F.linear(hidden_states, w_qkv)
        qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        scale = head_dim**-0.5
        attn_out = _local_attention(q, k, v, scale, causal=causal)
        out = attn_out.reshape(B, S_local, -1)
        return F.linear(out, w_o)

    ext = _get_ext()
    rank = dist.get_rank(group)
    num_heads_local = num_heads // world_size
    S_full = S_local * world_size

    # Establish UVA pointers and unified target buffers
    qkv_gathered, attn_gathered, signal_pad, dest_ptrs_qkv, dest_ptrs_attn, dest_ptrs_signal = _get_resources(
        B, S_local, S_full, num_heads, num_heads_local, head_dim, world_size, hidden_states.dtype, hidden_states.device
    )

    num_chunks = 2 if S_local >= 2 else 1
    chunk_size = S_local // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = S_local if i == num_chunks - 1 else (i + 1) * chunk_size
        chunks.append((start, end - start))

    streams = _get_streams(num_chunks)
    current_stream = torch.cuda.current_stream()
    
    for s in streams:
        s.wait_stream(current_stream)

    # 1. Pipeline QKV Matmul and P2P Scatter kernel
    for c, (start_s, chunk_len) in enumerate(chunks):
        with torch.cuda.stream(streams[c % len(streams)]):
            hs_chunk = hidden_states[:, start_s:start_s+chunk_len, :]
            qkv_chunk = F.linear(hs_chunk, w_qkv)
            qkv_chunk = qkv_chunk.view(B, chunk_len, 3, num_heads, head_dim)
            ext.launch_qkv_alltoall(
                qkv_chunk, dest_ptrs_qkv,
                B, chunk_len, num_heads, head_dim,
                rank, world_size, S_local, start_s
            )

    for s in streams:
        current_stream.wait_stream(s)

    # Await global reception of all query/key/value slices prior to attention compute
    dist.barrier(group=group)

    # 2. Complete Local Attention computation locally
    q = qkv_gathered[0]
    k = qkv_gathered[1]
    v = qkv_gathered[2]
    
    scale = head_dim**-0.5
    attn_out = _local_attention(q, k, v, scale, causal=causal)

    # Reset signal pads before triggering the final P2P stage
    signal_pad.zero_()
    dist.barrier(group=group)
    
    out = torch.empty(B, S_local, w_o.shape[0], device=hidden_states.device, dtype=hidden_states.dtype)

    for s in streams:
        s.wait_stream(current_stream)

    # 3. Pipeline Attn Scatter and Final Projection using Device Spinlocks
    for c, (start_s, chunk_len) in enumerate(chunks):
        with torch.cuda.stream(streams[c % len(streams)]):
            ext.launch_attn_out_alltoall(
                attn_out, dest_ptrs_attn,
                B, S_full, num_heads_local, head_dim,
                rank, world_size, S_local, start_s, chunk_len
            )
            ext.launch_signal_peers(dest_ptrs_signal, rank, c, world_size)
            ext.launch_wait_signal(signal_pad, c, world_size)
            
            attn_gathered_chunk = attn_gathered[:, start_s:start_s+chunk_len, :, :].reshape(B, chunk_len, -1)
            out_chunk = F.linear(attn_gathered_chunk, w_o)
            out[:, start_s:start_s+chunk_len, :] = out_chunk

    for s in streams:
        current_stream.wait_stream(s)

    return out