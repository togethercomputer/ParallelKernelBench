"""
Strategy:
- **Device-side P2P Communication**: Bypasses `all_to_all_single` intermediate steps by pushing and pulling data directly to/from peers via `torch.distributed._symmetric_memory` and NVLink pointers.
- **Upfront KV Fetch**: Uses a single custom kernel to asynchronously construct the fully redistributed KV tensor locally from peer memory.
- **Overlap & Pipelining**: Hides sequence gathering and output scattering behind SDPA computation. We double-buffer Queries and use three independent streams (Input Comm, Compute, Output Comm). This ensures `Q_{d+1}` is fetched and `Out_{d-1}` is scattered while SDPA processes `Q_d`.
- **Vectorized Copy**: Employs `uint4` (128-bit) vectorized loads/stores where the inner head dimension allows, maximizing 16-bit (bf16/fp16) memory bandwidth utilization.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// KV Gather Kernel
// ---------------------------------------------------------------------------

__global__ void gather_kv_kernel_uint4(
    const int64_t* __restrict__ symm_kv_ptrs,
    uint4* __restrict__ local_kv,
    int D, int S, int W, int Hkv, int Hd2_vec, int clip_token_nums, int rank
) {
    int64_t total_elements = (int64_t)D * clip_token_nums * Hkv * Hd2_vec;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd2_vec;
    int h_idx = (idx / Hd2_vec) % Hkv;
    int t_idx = idx / (Hd2_vec * Hkv);

    int d = t_idx / clip_token_nums;
    int c = t_idx % clip_token_nums;

    int r = c / S;
    int seq_idx = d * S + (c % S);
    int h_global = rank * Hkv + h_idx;

    const uint4* src_ptr = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(symm_kv_ptrs[r]));
    int64_t src_idx = ((int64_t)seq_idx * (W * Hkv) + h_global) * Hd2_vec + hd_idx;

    local_kv[idx] = src_ptr[src_idx];
}

__global__ void gather_kv_kernel_scalar(
    const int64_t* __restrict__ symm_kv_ptrs,
    uint16_t* __restrict__ local_kv,
    int D, int S, int W, int Hkv, int Hd2, int clip_token_nums, int rank
) {
    int64_t total_elements = (int64_t)D * clip_token_nums * Hkv * Hd2;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd2;
    int h_idx = (idx / Hd2) % Hkv;
    int t_idx = idx / (Hd2 * Hkv);

    int d = t_idx / clip_token_nums;
    int c = t_idx % clip_token_nums;

    int r = c / S;
    int seq_idx = d * S + (c % S);
    int h_global = rank * Hkv + h_idx;

    const uint16_t* src_ptr = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(symm_kv_ptrs[r]));
    int64_t src_idx = ((int64_t)seq_idx * (W * Hkv) + h_global) * Hd2 + hd_idx;

    local_kv[idx] = src_ptr[src_idx];
}

// ---------------------------------------------------------------------------
// Query Gather Kernel
// ---------------------------------------------------------------------------

__global__ void gather_q_kernel_uint4(
    const int64_t* __restrict__ symm_q_ptrs,
    uint4* __restrict__ local_q,
    int d, int S, int W, int Hq, int Hd_vec, int rank
) {
    int64_t total_elements = (int64_t)W * S * Hq * Hd_vec;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd_vec;
    int h_idx = (idx / Hd_vec) % Hq;
    int t_idx = idx / (Hd_vec * Hq);

    int r = t_idx / S;
    int seq_idx = d * S + (t_idx % S);
    int h_global = rank * Hq + h_idx;

    const uint4* src_ptr = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(symm_q_ptrs[r]));
    int64_t src_idx = ((int64_t)seq_idx * (W * Hq) + h_global) * Hd_vec + hd_idx;

    local_q[idx] = src_ptr[src_idx];
}

__global__ void gather_q_kernel_scalar(
    const int64_t* __restrict__ symm_q_ptrs,
    uint16_t* __restrict__ local_q,
    int d, int S, int W, int Hq, int Hd, int rank
) {
    int64_t total_elements = (int64_t)W * S * Hq * Hd;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd;
    int h_idx = (idx / Hd) % Hq;
    int t_idx = idx / (Hd * Hq);

    int r = t_idx / S;
    int seq_idx = d * S + (t_idx % S);
    int h_global = rank * Hq + h_idx;

    const uint16_t* src_ptr = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(symm_q_ptrs[r]));
    int64_t src_idx = ((int64_t)seq_idx * (W * Hq) + h_global) * Hd + hd_idx;

    local_q[idx] = src_ptr[src_idx];
}

// ---------------------------------------------------------------------------
// Output Scatter Kernel
// ---------------------------------------------------------------------------

__global__ void scatter_out_kernel_uint4(
    const uint4* __restrict__ local_out,
    const int64_t* __restrict__ symm_out_ptrs,
    int d, int S, int W, int Hq, int Hd_vec, int rank
) {
    int64_t total_elements = (int64_t)W * S * Hq * Hd_vec;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd_vec;
    int h_idx = (idx / Hd_vec) % Hq;
    int t_idx = idx / (Hd_vec * Hq);

    int r = t_idx / S;
    int seq_idx = d * S + (t_idx % S);
    int h_global = rank * Hq + h_idx;

    uint4* dst_ptr = reinterpret_cast<uint4*>(static_cast<uintptr_t>(symm_out_ptrs[r]));
    int64_t dst_idx = ((int64_t)seq_idx * (W * Hq) + h_global) * Hd_vec + hd_idx;

    dst_ptr[dst_idx] = local_out[idx];
}

__global__ void scatter_out_kernel_scalar(
    const uint16_t* __restrict__ local_out,
    const int64_t* __restrict__ symm_out_ptrs,
    int d, int S, int W, int Hq, int Hd, int rank
) {
    int64_t total_elements = (int64_t)W * S * Hq * Hd;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int hd_idx = idx % Hd;
    int h_idx = (idx / Hd) % Hq;
    int t_idx = idx / (Hd * Hq);

    int r = t_idx / S;
    int seq_idx = d * S + (t_idx % S);
    int h_global = rank * Hq + h_idx;

    uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(symm_out_ptrs[r]));
    int64_t dst_idx = ((int64_t)seq_idx * (W * Hq) + h_global) * Hd + hd_idx;

    dst_ptr[dst_idx] = local_out[idx];
}

// ---------------------------------------------------------------------------
// Python Bindings
// ---------------------------------------------------------------------------

void launch_gather_kv(
    torch::Tensor symm_kv_ptrs_tensor,
    torch::Tensor local_kv,
    int D, int S, int W, int Hkv, int Hd2, int clip_token_nums, int rank,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int64_t* ptrs = symm_kv_ptrs_tensor.data_ptr<int64_t>();
    
    if (Hd2 % 8 == 0) {
        int Hd2_vec = Hd2 / 8;
        int64_t total_elements = (int64_t)D * clip_token_nums * Hkv * Hd2_vec;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        gather_kv_kernel_uint4<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint4*>(local_kv.data_ptr()),
            D, S, W, Hkv, Hd2_vec, clip_token_nums, rank);
    } else {
        int64_t total_elements = (int64_t)D * clip_token_nums * Hkv * Hd2;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        gather_kv_kernel_scalar<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint16_t*>(local_kv.data_ptr()),
            D, S, W, Hkv, Hd2, clip_token_nums, rank);
    }
}

void launch_gather_q(
    torch::Tensor symm_q_ptrs_tensor,
    torch::Tensor local_q,
    int d, int S, int W, int Hq, int Hd, int rank,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int64_t* ptrs = symm_q_ptrs_tensor.data_ptr<int64_t>();

    if (Hd % 8 == 0) {
        int Hd_vec = Hd / 8;
        int64_t total_elements = (int64_t)W * S * Hq * Hd_vec;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        gather_q_kernel_uint4<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint4*>(local_q.data_ptr()),
            d, S, W, Hq, Hd_vec, rank);
    } else {
        int64_t total_elements = (int64_t)W * S * Hq * Hd;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        gather_q_kernel_scalar<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint16_t*>(local_q.data_ptr()),
            d, S, W, Hq, Hd, rank);
    }
}

void launch_scatter_out(
    torch::Tensor local_out,
    torch::Tensor symm_out_ptrs_tensor,
    int d, int S, int W, int Hq, int Hd, int rank,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int64_t* ptrs = symm_out_ptrs_tensor.data_ptr<int64_t>();

    if (Hd % 8 == 0) {
        int Hd_vec = Hd / 8;
        int64_t total_elements = (int64_t)W * S * Hq * Hd_vec;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        scatter_out_kernel_uint4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(local_out.data_ptr()), ptrs,
            d, S, W, Hq, Hd_vec, rank);
    } else {
        int64_t total_elements = (int64_t)W * S * Hq * Hd;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        scatter_out_kernel_scalar<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(local_out.data_ptr()), ptrs,
            d, S, W, Hq, Hd, rank);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_kv", &launch_gather_kv);
    m.def("launch_gather_q", &launch_gather_q);
    m.def("launch_scatter_out", &launch_scatter_out);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi_cso_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(q_shape, kv_shape, dtype, device, group):
    global _symm_cache
    key = (q_shape, kv_shape, dtype, device, id(group))
    if key in _symm_cache:
        return _symm_cache[key]
    
    q_buf = symm_mem.empty(q_shape, dtype=dtype, device=device)
    q_hdl = symm_mem.rendezvous(q_buf, group)
    
    kv_buf = symm_mem.empty(kv_shape, dtype=dtype, device=device)
    kv_hdl = symm_mem.rendezvous(kv_buf, group)
    
    out_buf = symm_mem.empty(q_shape, dtype=dtype, device=device)
    out_hdl = symm_mem.rendezvous(out_buf, group)
    
    q_ptrs = torch.tensor(q_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    kv_ptrs = torch.tensor(kv_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    out_ptrs = torch.tensor(out_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (q_buf, q_hdl, q_ptrs, kv_buf, kv_hdl, kv_ptrs, out_buf, out_hdl, out_ptrs)
    _symm_cache[key] = res
    return res

class PipelineState:
    def __init__(self, W, D, S, Hq, Hkv, Hd, clip_token_nums, dtype, device):
        self.stream_in = torch.cuda.Stream(device=device)
        self.stream_out = torch.cuda.Stream(device=device)
        self.ev_kv_ready = torch.cuda.Event()
        self.ev_q_ready = [torch.cuda.Event() for _ in range(D)]
        self.ev_sdpa_done = [torch.cuda.Event() for _ in range(D)]
        self.local_kv = torch.empty((D * clip_token_nums, Hkv, 2 * Hd), dtype=dtype, device=device)
        self.local_q_buf = [torch.empty((W * S, Hq, Hd), dtype=dtype, device=device) for _ in range(2)]

_state_cache = {}
def _get_pipeline_state(W, D, S, Hq, Hkv, Hd, clip_token_nums, dtype, device):
    key = (W, D, S, Hq, Hkv, Hd, clip_token_nums, dtype, device)
    if key not in _state_cache:
        _state_cache[key] = PipelineState(W, D, S, Hq, Hkv, Hd, clip_token_nums, dtype, device)
    return _state_cache[key]

def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)
    if k.shape[1] < q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).contiguous()


@torch.no_grad()
def solution(
    query: torch.Tensor,
    key_value: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_shuffle_num: int,
    clip_token_nums: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert query.element_size() == 2, "Kernels designed for 16-bit dtypes (bf16/fp16)"
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    
    D = cp_shuffle_num
    W = world_size
    
    # Handle implicit repeated heads for KV upfront
    if key_value.shape[1] < W and W % key_value.shape[1] == 0:
        key_value = key_value.repeat_interleave(W // key_value.shape[1], dim=1)
        
    tokens, heads_q, Hd = query.shape
    _, heads_kv, Hd2 = key_value.shape
    
    if tokens % D != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    if heads_q % W != 0 or heads_kv % W != 0:
        raise ValueError("heads must divide evenly across context ranks")
        
    S = tokens // D
    Hq = heads_q // W
    Hkv = heads_kv // W
    
    clip_token_nums = min(int(clip_token_nums or W * S), W * S)
    
    q_buf, q_hdl, q_ptrs, kv_buf, kv_hdl, kv_ptrs, out_buf, out_hdl, out_ptrs = _get_symm_state(
        query.shape, key_value.shape, query.dtype, query.device, group
    )
    
    # Push inputs into contiguous symmetric memory for fast peer access
    q_buf.copy_(query)
    kv_buf.copy_(key_value)
    q_hdl.barrier(channel=0)
    
    state = _get_pipeline_state(W, D, S, Hq, Hkv, Hd, clip_token_nums, query.dtype, query.device)
    ext = _get_ext()
    comp_stream = torch.cuda.current_stream()
    
    # Launch overlapped communication pipeline
    with torch.cuda.stream(state.stream_in):
        # 1. Fetch entire sequence KV async
        ext.launch_gather_kv(
            kv_ptrs, state.local_kv, D, S, W, Hkv, Hd2, clip_token_nums, rank, state.stream_in.cuda_stream
        )
        state.ev_kv_ready.record(stream=state.stream_in)
        
        # 2. Pipeline fetching Query chunks
        for d in range(D):
            if d >= 2:
                # Prevent overwriting Q buffers before SDPA finishes reading
                state.ev_sdpa_done[d - 2].wait(stream=state.stream_in)
            
            ext.launch_gather_q(
                q_ptrs, state.local_q_buf[d % 2], d, S, W, Hq, Hd, rank, state.stream_in.cuda_stream
            )
            state.ev_q_ready[d].record(stream=state.stream_in)

    local_out_res = []
    
    # Execute Compute (SDPA)
    for d in range(D):
        if d == 0:
            state.ev_kv_ready.wait(stream=comp_stream)
        state.ev_q_ready[d].wait(stream=comp_stream)
        
        q = state.local_q_buf[d % 2]
        start = int(k_ranges[d, 0])
        end = int(k_ranges[d, 1])
        k = state.local_kv[start:end, :, :Hd]
        v = state.local_kv[start:end, :, Hd:]
        
        # Compute exact SDPA reference, capturing resulting tensor
        out = _sdpa(q, k, v)
        local_out_res.append(out.contiguous())
        
        state.ev_sdpa_done[d].record(stream=comp_stream)
        
    # Scatter Pipeline (pushes outputs directly into peers' buffers)
    with torch.cuda.stream(state.stream_out):
        for d in range(D):
            state.ev_sdpa_done[d].wait(stream=state.stream_out)
            ext.launch_scatter_out(
                local_out_res[d], out_ptrs, d, S, W, Hq, Hd, rank, state.stream_out.cuda_stream
            )

    comp_stream.wait_stream(state.stream_out)
    
    # Prevent early exit before all scattered segments have arrived
    q_hdl.barrier(channel=1)
    
    return out_buf.clone()