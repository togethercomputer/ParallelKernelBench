from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// 1. CP Ring P2P Copy
// ---------------------------------------------------------------------------

__global__ void p2p_copy_kv_kernel_128(
    const int64_t remote_ptr,
    void* __restrict__ next_k,
    void* __restrict__ next_v,
    int64_t numel_128
) {
    const uint4* src_k = reinterpret_cast<const uint4*>(remote_ptr);
    const uint4* src_v = src_k + numel_128;
    
    uint4* dst_k = reinterpret_cast<uint4*>(next_k);
    uint4* dst_v = reinterpret_cast<uint4*>(next_v);
    
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel_128) {
        dst_k[idx] = src_k[idx];
        dst_v[idx] = src_v[idx];
    }
}

__global__ void p2p_copy_kv_kernel(
    const int64_t remote_ptr,
    __nv_bfloat16* __restrict__ next_k,
    __nv_bfloat16* __restrict__ next_v,
    int64_t numel
) {
    const __nv_bfloat16* src_k = reinterpret_cast<const __nv_bfloat16*>(remote_ptr);
    const __nv_bfloat16* src_v = src_k + numel;
    
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        next_k[idx] = src_k[idx];
        next_v[idx] = src_v[idx];
    }
}

void launch_p2p_copy_kv(
    int64_t remote_ptr,
    torch::Tensor next_k,
    torch::Tensor next_v
) {
    int64_t numel = next_k.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (numel % 8 == 0) {
        int64_t numel_128 = numel / 8;
        int threads = 256;
        int blocks = (numel_128 + threads - 1) / threads;
        p2p_copy_kv_kernel_128<<<blocks, threads, 0, stream>>>(
            remote_ptr,
            next_k.data_ptr(),
            next_v.data_ptr(),
            numel_128
        );
    } else {
        int threads = 256;
        int blocks = (numel + threads - 1) / threads;
        p2p_copy_kv_kernel<<<blocks, threads, 0, stream>>>(
            remote_ptr,
            (__nv_bfloat16*)next_k.data_ptr(),
            (__nv_bfloat16*)next_v.data_ptr(),
            numel
        );
    }
}

// ---------------------------------------------------------------------------
// 2. CP Ring Merge Out & LSE
// ---------------------------------------------------------------------------

__global__ void merge_out_lse_kernel_block(
    float* __restrict__ out,
    float* __restrict__ lse,
    const float* __restrict__ block_out,
    const float* __restrict__ block_lse,
    int64_t B, int64_t S, int64_t H, int64_t D
) {
    int64_t lse_idx = blockIdx.x;
    int64_t h = lse_idx % H;
    int64_t tmp = lse_idx / H;
    int64_t s = tmp % S;
    int64_t b = tmp / S;

    int64_t blse_idx = b * (H * S) + h * S + s;

    __shared__ float sh_sig;

    if (threadIdx.x == 0) {
        float curr_lse = lse[lse_idx];
        float b_lse = block_lse[blse_idx];
        
        float max_lse = fmaxf(curr_lse, b_lse);
        float exp_curr = expf(curr_lse - max_lse);
        float exp_b = expf(b_lse - max_lse);
        float sum_exp = exp_curr + exp_b;
        
        sh_sig = exp_b / sum_exp;
        
        // Write the new updated LSE exclusively
        lse[lse_idx] = max_lse + logf(sum_exp);
    }
    
    __syncthreads();
    
    float sig = sh_sig;
    int64_t out_base = lse_idx * D;
    
    // Process the inner dimension seamlessly via fast coalesced accesses 
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float curr_out = out[out_base + d];
        float b_out = block_out[out_base + d];
        out[out_base + d] = curr_out - sig * (curr_out - b_out);
    }
}

void launch_merge_out_lse(
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor block_out,
    torch::Tensor block_lse
) {
    int64_t B = out.size(0);
    int64_t S = out.size(1);
    int64_t H = out.size(2);
    int64_t D = out.size(3);
    
    int blocks = B * S * H;
    int threads = 128;
    if (D < 128) {
        threads = 32;
        while (threads < D) threads *= 2;
    } else if (D > 128) {
        threads = 256;
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    merge_out_lse_kernel_block<<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(),
        lse.data_ptr<float>(),
        block_out.data_ptr<float>(),
        block_lse.data_ptr<float>(),
        B, S, H, D
    );
}

// ---------------------------------------------------------------------------
// 3. TP Multimem Allreduce (from switch PTX limits)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do { asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do { asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do { asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do { asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ void blockwise_barrier_acq_rel(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride; block_start < numel_per_rank; block_start += (int64_t)num_programs * (int64_t)block_stride) {
        const int64_t offsets = block_start + (int64_t)tid;
        if (offsets >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

__global__ void allreduce_bf16_fallback_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride
) {
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride);
}

void launch_allreduce_bf16_fallback(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_fallback_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_p2p_copy_kv", &launch_p2p_copy_kv);
    m.def("launch_merge_out_lse", &launch_merge_out_lse);
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_bf16_fallback", &launch_allreduce_bf16_fallback);
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_optim_ext", CUDA_SRC)
    return _ext

WARP_SIZE = 32
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16

def _multimem_launch_config(numel: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16 assumes 2 bytes
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size

_tp_cache = {}
def get_tp_symm_resources(shape, dtype, device, group):
    key = (shape, dtype, device, id(group))
    if key in _tp_cache:
        return _tp_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, out, ptrs_tensor)
    _tp_cache[key] = res
    return res

_cp_cache = {}
def get_cp_symm_resources(shape_KV, dtype, device, group):
    key = (shape_KV, dtype, device, id(group))
    if key in _cp_cache:
        return _cp_cache[key]
    
    buf = symm_mem.empty(shape_KV, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _cp_cache[key] = (buf, hdl)
    return buf, hdl

def tp_allreduce(tensor, group):
    tp_size = dist.get_world_size(group)
    if tp_size == 1:
        return tensor
    
    n = tensor.numel()
    buf, hdl, out, ptrs_tensor = get_tp_symm_resources(tensor.shape, tensor.dtype, tensor.device, group)
    buf.copy_(tensor)
    
    numel_per_thread = BYTES_PER_THREAD // 2
    if n % numel_per_thread != 0:
        hdl.barrier(channel=0)
        _get_ext().launch_allreduce_bf16_fallback(ptrs_tensor, out, n)
        return out
        
    numel_128 = n // numel_per_thread
    num_blocks, block_size, block_stride = _multimem_launch_config(n, tp_size)
    
    dist.barrier(group=group)
    
    multicast_ptr = int(hdl.multicast_ptr)
    signal_dev = hdl.signal_pad_ptrs_dev
    _get_ext().launch_multimem_allreduce_bf16(
        multicast_ptr, signal_dev, numel_128, tp_size, dist.get_rank(group),
        num_blocks, block_size, block_stride
    )
    return buf.reshape_as(tensor).clone()

def _local_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qh = q.float().transpose(1, 2)
    kh = k.float().transpose(1, 2)
    vh = v.float().transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_lse = torch.logsumexp(scores, dim=-1)
    block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
    return block_out, block_lse

def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank Megatron-style CP+TP ring attention forward via Device Overlapped P2P/Multimem logic.
    """
    tp_group = tp_group or dist.group.WORLD
    cp_group = cp_group or dist.group.WORLD
    
    tp_size = dist.get_world_size(tp_group)
    cp_size = dist.get_world_size(cp_group)
    
    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
        
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
        
    B, S = hidden_states.shape[:2]
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, heads_local, head_dim)
    q, k, v = qkv.unbind(dim=2)
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    out = None
    lse = None
    
    if cp_size == 1:
        block_out, block_lse = _local_attn(q, k, v, float(softmax_scale), causal)
        out = block_out.to(q.dtype)
    else:
        # CP Buffer Setup and Remote Ring Allocation
        cp_rank = dist.get_rank(cp_group)
        shape_KV = (2, B, S, heads_local, head_dim)
        symm_KV_buf, cp_hdl = get_cp_symm_resources(shape_KV, k.dtype, k.device, cp_group)
        
        symm_KV_buf[0].copy_(k)
        symm_KV_buf[1].copy_(v)
        cp_hdl.barrier(channel=0)
        
        curr_K, curr_V = k, v
        next_K = torch.empty_like(k)
        next_V = torch.empty_like(v)
        
        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()
        copy_event = torch.cuda.Event()
        compute_event = torch.cuda.Event()
        
        for step in range(cp_size):
            # Fetch directly from the requisite peer skipping traditional ring passes 
            if step + 1 < cp_size:
                next_source = (cp_rank - step - 1) % cp_size
                remote_ptr = cp_hdl.buffer_ptrs[next_source]
                
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(compute_event)
                    _get_ext().launch_p2p_copy_kv(remote_ptr, next_K, next_V)
                    copy_event.record(copy_stream)
                    
            if (not causal) or step <= cp_rank:
                is_causal = causal and (step == 0)
                block_out, block_lse = _local_attn(q, curr_K, curr_V, float(softmax_scale), is_causal)
                
                if out is None:
                    out = block_out.clone()
                    lse = block_lse.transpose(-2, -1).contiguous()
                else:
                    _get_ext().launch_merge_out_lse(out, lse, block_out, block_lse)
                    
            compute_event.record(compute_stream)
            
            if step + 1 < cp_size:
                compute_stream.wait_event(copy_event)
                
            curr_K, next_K = next_K, curr_K
            curr_V, next_V = next_V, curr_V
            
        out = out.to(q.dtype)
        
    out = F.linear(out.view(B, S, -1), w_o)
    
    if tp_size > 1:
        out = tp_allreduce(out, tp_group)
        
    return out