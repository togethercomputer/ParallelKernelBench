import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// P2P Signal and Wait Kernels (Acquire/Release Semantics)
// ---------------------------------------------------------------------------

__global__ void p2p_signal_kernel(uint32_t* addr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t tmp;
        do {
            asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;" 
                         : "=r"(tmp) : "l"(addr) : "memory");
        } while (tmp != 0);
    }
}

__global__ void p2p_wait_kernel(uint32_t* addr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t tmp;
        do {
            asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;" 
                         : "=r"(tmp) : "l"(addr) : "memory");
        } while (tmp != 1);
    }
}

void p2p_signal(int64_t addr, int64_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    p2p_signal_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(addr));
}

void p2p_wait(int64_t addr, int64_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    p2p_wait_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(addr));
}

void p2p_memcpy_async(int64_t dst_ptr, int64_t src_ptr, int64_t size_bytes, int64_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    C10_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), 
                                   reinterpret_cast<const void*>(src_ptr), 
                                   size_bytes, cudaMemcpyDefault, stream));
}

// ---------------------------------------------------------------------------
// Fused Elementwise Kernel for Attention Backward
// ---------------------------------------------------------------------------

__global__ void fused_elementwise_bf16_kernel(
    __nv_bfloat16* __restrict__ scores,
    __nv_bfloat16* __restrict__ dP,
    const float* __restrict__ lse,
    const float* __restrict__ row_dot,
    int BH, int Sq, int Sk, bool causal, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BH * Sq * Sk;
    if (idx < total) {
        int sk_idx = idx % Sk;
        int tmp = idx / Sk;
        int sq_idx = tmp % Sq;
        int bh_idx = tmp / Sq;
        
        float score = __bfloat162float(scores[idx]) * scale;
        if (causal && sq_idx < sk_idx) {
            score = -INFINITY;
        }
        
        float cur_lse = lse[bh_idx * Sq + sq_idx];
        float prob = expf(score - cur_lse);
        
        float dp_val = __bfloat162float(dP[idx]);
        float rd_val = row_dot[bh_idx * Sq + sq_idx];
        float ds_val = prob * (dp_val - rd_val);
        
        scores[idx] = __float2bfloat16(prob);
        dP[idx] = __float2bfloat16(ds_val * scale);
    }
}

void fused_elementwise_bf16(
    torch::Tensor scores, torch::Tensor dP,
    torch::Tensor lse, torch::Tensor row_dot,
    int BH, int Sq, int Sk, bool causal, float scale, int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int total = BH * Sq * Sk;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_elementwise_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dP.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        row_dot.data_ptr<float>(),
        BH, Sq, Sk, causal, scale
    );
}

// ---------------------------------------------------------------------------
// Multimem DP All-Reduce (Hopper)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[flat_tid] + block_id * world_size + rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank] + block_id * world_size + flat_tid);
    
    uint32_t tmp;
    do { asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(send_addr) : "memory"); } while (tmp != 0u);
    do { asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(wait_addr) : "memory"); } while (tmp != 1u);
}

__device__ __forceinline__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[flat_tid] + block_id * world_size + rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank] + block_id * world_size + flat_tid);
    
    uint32_t tmp;
    do { asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(send_addr) : "memory"); } while (tmp != 0u);
    do { asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(wait_addr) : "memory"); } while (tmp != 1u);
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base, const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128, int world_size, int rank, int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = block_id * block_stride; block_start < numel_per_rank; block_start += num_programs * block_stride) {
        const int64_t offsets = block_start + tid;
        if (offsets >= numel_per_rank) continue;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + (rank * numel_per_rank + offsets) * 2;
        uint32_t r0, r1, r2, r3;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(ptrs) : "memory");
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" : : "l"(ptrs), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

__global__ void allreduce_bf16_kernel(const long long* __restrict__ ptrs, __nv_bfloat16* __restrict__ out, int world_size, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            sum += __bfloat162float(((const __nv_bfloat16*)ptrs[r])[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr, torch::Tensor signal_pad_ptrs_tensor, int64_t numel_128,
    int world_size, int rank, int num_blocks, int block_size, int block_stride, int64_t stream_ptr
) {
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride);
}

void launch_allreduce(torch::Tensor ptrs_tensor, torch::Tensor out, int64_t n, int64_t stream_ptr) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = min((int)((n + threads - 1) / threads), 65535);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(d_ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), world_size, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("p2p_signal", &p2p_signal);
    m.def("p2p_wait", &p2p_wait);
    m.def("p2p_memcpy_async", &p2p_memcpy_async);
    m.def("fused_elementwise_bf16", &fused_elementwise_bf16);
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce", &launch_allreduce);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attention_bwd_ext", CUDA_SRC)
    return _ext

_cp_resource_cache = {}
def get_cp_resources(B, S, H, D, dtype, device, cp_group):
    key = (B, S, H, D, dtype, device)
    if key in _cp_resource_cache:
        return _cp_resource_cache[key]
    
    big_buf = symm_mem.empty((4, 2, B, S, H, D), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(big_buf, group=cp_group)
    ready_buf = symm_mem.empty((2,), dtype=torch.int32, device=device).zero_()
    ready_hdl = symm_mem.rendezvous(ready_buf, group=cp_group)
    done_buf = symm_mem.empty((2,), dtype=torch.int32, device=device).fill_(1)
    done_hdl = symm_mem.rendezvous(done_buf, group=cp_group)
    comm_stream = torch.cuda.Stream()
    
    res = (big_buf, hdl, ready_buf, ready_hdl, done_buf, done_hdl, comm_stream)
    _cp_resource_cache[key] = res
    return res

_dp_resource_cache = {}
def allreduce_dp(tensor, dp_group):
    n = tensor.numel()
    key = (n, tensor.dtype, tensor.device)
    if key not in _dp_resource_cache:
        buf = symm_mem.empty(n, dtype=tensor.dtype, device=tensor.device)
        hdl = symm_mem.rendezvous(buf, group=dp_group)
        out = torch.empty(n, dtype=tensor.dtype, device=tensor.device)
        ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=tensor.device, dtype=torch.int64)
        _dp_resource_cache[key] = (buf, hdl, out, ptrs_tensor)
        
    buf, hdl, out, ptrs_tensor = _dp_resource_cache[key]
    buf.copy_(tensor)
    
    numel_per_thread = 8
    if n % numel_per_thread != 0:
        hdl.barrier(channel=0)
        _get_ext().launch_allreduce(ptrs_tensor, out, n, torch.cuda.current_stream().cuda_stream)
        return out
        
    numel_128 = n // numel_per_thread
    num_threads = (numel_128 + hdl.world_size - 1) // hdl.world_size
    if num_threads < 1024:
        block_size = 1
        while block_size < num_threads: block_size *= 2
        num_blocks = 1
    else:
        block_size = 1024
        num_blocks = min((num_threads + 1023) // 1024, 4)
        
    dist.barrier(group=dp_group)
    _get_ext().launch_multimem_allreduce_bf16(
        int(hdl.multicast_ptr), hdl.signal_pad_ptrs_dev, numel_128, hdl.world_size, hdl.rank,
        num_blocks, block_size, block_size, torch.cuda.current_stream().cuda_stream
    )
    return buf.clone()

def compute_local_attn(q, k, v, dout, out, lse, scale, causal, row_dot):
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    doh = dout.transpose(1, 2)
    
    scores = torch.matmul(qh, kh.transpose(-2, -1))
    dP = torch.matmul(doh, vh.transpose(-2, -1))
    
    B, H, Sq, D = qh.shape
    Sk = kh.shape[2]
    
    _get_ext().fused_elementwise_bf16(
        scores, dP, lse, row_dot, B*H, Sq, Sk, causal, scale, torch.cuda.current_stream().cuda_stream
    )
    
    dQ = torch.matmul(dP, kh)
    dK = torch.matmul(dP.transpose(-2, -1), qh)
    dV = torch.matmul(scores.transpose(-2, -1), doh)
    return dQ.transpose(1, 2).contiguous(), dK.transpose(1, 2).contiguous(), dV.transpose(1, 2).contiguous()

@torch.no_grad()
def solution(
    dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, softmax_lse: torch.Tensor, softmax_scale: Optional[float] = None,
    causal: bool = False, cp_group: Optional[dist.ProcessGroup] = None, dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    cp_group = cp_group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
        
    C = dist.get_world_size(cp_group)
    row_dot = (dout.float() * out.float()).sum(dim=-1, keepdim=True).transpose(1, 2).contiguous()
    lse = softmax_lse.contiguous()
    
    if C == 1:
        dq, dk, dv = compute_local_attn(q, k, v, dout, out, lse, softmax_scale, causal, row_dot)
    else:
        rank = dist.get_rank(cp_group)
        B, S, H, D = q.shape
        big_buf, hdl, ready_buf, ready_hdl, done_buf, done_hdl, comm_stream = get_cp_resources(B, S, H, D, q.dtype, q.device, cp_group)
        
        global_next = dist.get_global_rank(cp_group, (rank + 1) % C)
        global_prev = dist.get_global_rank(cp_group, (rank - 1) % C)
        
        peer_next_base_ptr = int(hdl.buffer_ptrs[global_next])
        peer_next_ready_ptr = int(ready_hdl.buffer_ptrs[global_next])
        peer_prev_done_ptr = int(done_hdl.buffer_ptrs[global_prev])
        size_per_buf = B * S * H * D * q.element_size()
        compute_stream = torch.cuda.current_stream()
        
        dq, dk_curr, dv_curr = None, None, None
        
        for i in range(C):
            buf_idx = i % 2
            next_buf_idx = (i + 1) % 2
            
            if i > 0:
                _get_ext().p2p_wait(ready_buf.data_ptr() + buf_idx * 4, compute_stream.cuda_stream)
                
            k_curr = q if i == 0 else big_buf[0, buf_idx]
            v_curr = v if i == 0 else big_buf[1, buf_idx]
            
            if i <= rank or not causal:
                block_dq, block_dk, block_dv = compute_local_attn(q, k_curr, v_curr, dout, out, lse, softmax_scale, causal and (i == 0), row_dot)
                if i == 0:
                    dq, dk_curr, dv_curr = block_dq, block_dk, block_dv
                else:
                    dq.add_(block_dq)
                    dk_curr = block_dk.add_(big_buf[2, buf_idx])
                    dv_curr = block_dv.add_(big_buf[3, buf_idx])
            else:
                if i > 0:
                    dk_curr = big_buf[2, buf_idx]
                    dv_curr = big_buf[3, buf_idx]
                    
            if i > 0:
                _get_ext().p2p_signal(peer_prev_done_ptr + buf_idx * 4, compute_stream.cuda_stream)
                
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(compute_stream)
                _get_ext().p2p_wait(done_buf.data_ptr() + next_buf_idx * 4, comm_stream.cuda_stream)
                
                if i + 1 < C:
                    _get_ext().p2p_memcpy_async(peer_next_base_ptr + (next_buf_idx) * size_per_buf, k_curr.data_ptr(), size_per_buf, comm_stream.cuda_stream)
                    _get_ext().p2p_memcpy_async(peer_next_base_ptr + (2 + next_buf_idx) * size_per_buf, v_curr.data_ptr(), size_per_buf, comm_stream.cuda_stream)
                
                _get_ext().p2p_memcpy_async(peer_next_base_ptr + (4 + next_buf_idx) * size_per_buf, dk_curr.data_ptr(), size_per_buf, comm_stream.cuda_stream)
                _get_ext().p2p_memcpy_async(peer_next_base_ptr + (6 + next_buf_idx) * size_per_buf, dv_curr.data_ptr(), size_per_buf, comm_stream.cuda_stream)
                
                _get_ext().p2p_signal(peer_next_ready_ptr + next_buf_idx * 4, comm_stream.cuda_stream)
                
        compute_stream.wait_stream(comm_stream)
        final_buf_idx = C % 2
        _get_ext().p2p_wait(ready_buf.data_ptr() + final_buf_idx * 4, compute_stream.cuda_stream)
        dk = big_buf[2, final_buf_idx].clone()
        dv = big_buf[3, final_buf_idx].clone()
        _get_ext().p2p_signal(peer_prev_done_ptr + final_buf_idx * 4, compute_stream.cuda_stream)

    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        dp_size = dist.get_world_size(dp_group)
        packed = torch.cat([dq.flatten(), dk.flatten(), dv.flatten()])
        packed = allreduce_dp(packed, dp_group)
        packed.div_(dp_size)
        
        split_size = dq.numel()
        dq = packed[:split_size].view(dq.shape)
        dk = packed[split_size:2*split_size].view(dk.shape)
        dv = packed[2*split_size:].view(dv.shape)

    return dq, dk, dv