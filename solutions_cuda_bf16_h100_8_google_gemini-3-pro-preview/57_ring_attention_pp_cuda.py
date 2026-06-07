from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

# ---------------------------------------------------------------------------
# Custom CUDA Extension for P2P via Symmetric Memory & Fused Math
# ---------------------------------------------------------------------------

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Atomic release flag setting
__global__ void set_flag_kernel(uint32_t* addr, uint32_t val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        asm volatile("atom.global.release.sys.exch.b32 %0, [%1], %2;"
                     : "=r"(val) : "l"(addr), "r"(val) : "memory");
    }
}

// Single-thread spin wait for stream synchronization
__global__ void wait_kernel(uint32_t* flag_addr, uint32_t wait_val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t val;
        do {
            asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                         : "=r"(val) : "l"(flag_addr) : "memory");
        } while (val < wait_val);
    }
}

// Push K and V to peer and signal
void push_kv_and_signal(
    torch::Tensor local_k, torch::Tensor local_v,
    int64_t remote_k_ptr, int64_t remote_v_ptr,
    int64_t remote_flag_ptr, uint32_t flag_val,
    int64_t stream_ptr
) {
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : at::cuda::getCurrentCUDAStream().stream();
    int64_t bytes = local_k.numel() * sizeof(at::BFloat16);
    
    cudaMemcpyAsync(reinterpret_cast<void*>(remote_k_ptr), local_k.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<void*>(remote_v_ptr), local_v.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
    
    set_flag_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(remote_flag_ptr), flag_val);
}

// Push PP buffer to peer and signal
void push_pp_and_signal(
    torch::Tensor local_data,
    int64_t remote_data_ptr,
    int64_t remote_flag_ptr, uint32_t flag_val,
    int64_t stream_ptr
) {
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : at::cuda::getCurrentCUDAStream().stream();
    int64_t bytes = local_data.numel() * sizeof(at::BFloat16);
    
    cudaMemcpyAsync(reinterpret_cast<void*>(remote_data_ptr), local_data.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
    
    set_flag_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(remote_flag_ptr), flag_val);
}

void wait_signal(int64_t flag_ptr, uint32_t wait_val, int64_t stream_ptr) {
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : at::cuda::getCurrentCUDAStream().stream();
    wait_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag_ptr), wait_val);
}

void set_flag_python(int64_t flag_ptr, uint32_t flag_val, int64_t stream_ptr) {
    cudaStream_t stream = stream_ptr ? reinterpret_cast<cudaStream_t>(stream_ptr) : at::cuda::getCurrentCUDAStream().stream();
    set_flag_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag_ptr), flag_val);
}

// Fused init kernel
__global__ void init_out_lse_kernel(
    float* out, float* lse,
    const at::BFloat16* block_out, const float* block_lse,
    int B, int S, int H, int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * S * H * D;
    if (idx >= total) return;
    
    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    tmp = tmp / H;
    int s = tmp % S;
    int b = tmp / S;
    
    out[idx] = __bfloat162float(block_out[idx]);
    if (d == 0) {
        lse[(int64_t)b * (S * H) + s * H + h] = block_lse[(int64_t)b * (H * S) + h * S + s];
    }
}

// Fused in-place sigmoid LSE and output block merge
__global__ void merge_out_lse_kernel(
    float* out, float* lse,
    const at::BFloat16* block_out, const float* block_lse,
    int B, int S, int H, int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * S * H * D;
    if (idx >= total) return;
    
    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    tmp = tmp / H;
    int s = tmp % S;
    int b = tmp / S;
    
    int64_t lse_idx_block = (int64_t)b * (H * S) + h * S + s;
    int64_t lse_idx_out = (int64_t)b * (S * H) + s * H + h;
    
    float current_lse = lse[lse_idx_out];
    float b_lse = block_lse[lse_idx_block];
    
    float diff = b_lse - current_lse;
    float sig = 1.0f / (1.0f + expf(-diff));
    
    float current_out = out[idx];
    float b_out = __bfloat162float(block_out[idx]);
    
    out[idx] = current_out - sig * (current_out - b_out);
    
    if (d == 0) {
        float x = current_lse - b_lse;
        float log_sig = (x >= 0) ? -log1pf(expf(-x)) : (x - log1pf(expf(x)));
        lse[lse_idx_out] = current_lse - log_sig;
    }
}

void init_out_lse(torch::Tensor out, torch::Tensor lse, torch::Tensor block_out, torch::Tensor block_lse) {
    int B = out.size(0); int S = out.size(1); int H = out.size(2); int D = out.size(3);
    int threads = 256; int blocks = ((int64_t)B * S * H * D + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    init_out_lse_kernel<<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(), lse.data_ptr<float>(),
        reinterpret_cast<const at::BFloat16*>(block_out.data_ptr<at::BFloat16>()),
        block_lse.data_ptr<float>(), B, S, H, D
    );
}

void merge_out_lse(torch::Tensor out, torch::Tensor lse, torch::Tensor block_out, torch::Tensor block_lse) {
    int B = out.size(0); int S = out.size(1); int H = out.size(2); int D = out.size(3);
    int threads = 256; int blocks = ((int64_t)B * S * H * D + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    merge_out_lse_kernel<<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(), lse.data_ptr<float>(),
        reinterpret_cast<const at::BFloat16*>(block_out.data_ptr<at::BFloat16>()),
        block_lse.data_ptr<float>(), B, S, H, D
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("push_kv_and_signal", &push_kv_and_signal);
    m.def("push_pp_and_signal", &push_pp_and_signal);
    m.def("wait_signal", &wait_signal);
    m.def("set_flag_python", &set_flag_python);
    m.def("init_out_lse", &init_out_lse);
    m.def("merge_out_lse", &merge_out_lse);
}
'''

_ext_module = None

def _get_ext():
    global _ext_module
    if _ext_module is None:
        _ext_module = compile_cuda_extension("ring_attn_pp_ext", CUDA_SRC)
    return _ext_module


_cache = {}

def _get_resources(B, S, hidden_size, num_heads, head_dim, dtype, device):
    key = (B, S, hidden_size, num_heads, head_dim, dtype, device)
    if key in _cache:
        return _cache[key]

    # Global buffers since symmetric memory acts on WORLD
    kv_shape = (2, 2, B, S, num_heads, head_dim)
    kv_buf = symm_mem.empty(kv_shape, dtype=dtype, device=device)
    kv_hdl = symm_mem.rendezvous(kv_buf, group=dist.group.WORLD)
    
    pp_shape = (B, S, hidden_size)
    pp_buf = symm_mem.empty(pp_shape, dtype=dtype, device=device)
    pp_hdl = symm_mem.rendezvous(pp_buf, group=dist.group.WORLD)
    
    # 4 x uint32 [cp_flag, pp_flag, reserved, pp_ack]
    flags_buf = symm_mem.empty((4,), dtype=torch.int32, device=device)
    flags_buf.zero_()
    flags_hdl = symm_mem.rendezvous(flags_buf, group=dist.group.WORLD)
    
    out_buf = torch.empty((B, S, num_heads, head_dim), dtype=torch.float32, device=device)
    lse_buf = torch.empty((B, S, num_heads), dtype=torch.float32, device=device)
    
    comm_stream = torch.cuda.Stream()
    
    state = {
        "kv_buf": kv_buf, "kv_hdl": kv_hdl,
        "pp_buf": pp_buf, "pp_hdl": pp_hdl,
        "flags_buf": flags_buf, "flags_hdl": flags_hdl,
        "out_buf": out_buf, "lse_buf": lse_buf,
        "comm_stream": comm_stream,
        "cp_push_count": 0, "cp_wait_count": 0,
        "pp_push_count": 0, "pp_wait_count": 0,
        "pp_ack_push_count": 0, "pp_ack_wait_count": 0,
    }
    _cache[key] = state
    return state


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    cp_group = cp_group or dist.group.WORLD
    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)
    
    device = hidden_states.device
    dtype = hidden_states.dtype
    B, S_local, hidden_size = hidden_states.shape
    
    is_first, is_last = True, True
    pp_rank, pp_size = 0, 1
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        is_first = (pp_rank == 0)
        is_last = (pp_rank == pp_size - 1)
        
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    
    _ext = _get_ext()
    
    # Fast path for standalone
    if cp_size == 1 and pp_size == 1:
        qkv = F.linear(hidden_states, w_qkv).view(B, S_local, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        qh, kh, vh = q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(q.size(1), k.size(1), device=device, dtype=torch.bool), 1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
        return F.linear(block_out.to(dtype).reshape(B, S_local, -1), w_o)

    state = _get_resources(B, S_local, hidden_size, num_heads, head_dim, dtype, device)
    
    flags_ptrs = state["flags_hdl"].buffer_ptrs
    my_flags_base = flags_ptrs[dist.get_rank()]
    my_cp_flag_ptr = my_flags_base + 0
    my_pp_flag_ptr = my_flags_base + 4
    my_pp_ack_ptr = my_flags_base + 12
    pp_ptrs = state["pp_hdl"].buffer_ptrs

    pp_push_count, pp_wait_count = state["pp_push_count"], state["pp_wait_count"]
    pp_ack_push_count, pp_ack_wait_count = state["pp_ack_push_count"], state["pp_ack_wait_count"]

    # 1. Pipeline-Parallel Recv
    if not is_first:
        pp_wait_count += 1
        _ext.wait_signal(my_pp_flag_ptr, pp_wait_count, 0)
        stage_input = state["pp_buf"].clone()
        
        pp_ack_push_count += 1
        prev_rank = dist.get_global_rank(pp_group, (pp_rank - 1) % pp_size)
        _ext.set_flag_python(flags_ptrs[prev_rank] + 12, pp_ack_push_count, 0)
    else:
        stage_input = hidden_states

    # 2. Local Context Parallel / Attention QKV Split
    qkv = F.linear(stage_input, w_qkv).view(B, S_local, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    if cp_size == 1:
        qh, kh, vh = q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(q.size(1), k.size(1), device=device, dtype=torch.bool), 1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
        ctx = block_out.to(dtype)
    else:
        # Pipelined CP Ring (Overlapped Communication + Computation via double buffering)
        dist.barrier(group=cp_group) 
        
        kv_buf = state["kv_buf"]
        kv_buf[0, 0].copy_(k)
        kv_buf[0, 1].copy_(v)
        
        dist.barrier(group=cp_group) 
        
        out_buf, lse_buf, comm_stream = state["out_buf"], state["lse_buf"], state["comm_stream"]
        cp_push_count, cp_wait_count = state["cp_push_count"], state["cp_wait_count"]
        
        global_next_cp = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
        peer_kv_base = state["kv_hdl"].buffer_ptrs[global_next_cp]
        peer_cp_flag_ptr = flags_ptrs[global_next_cp] + 0
        
        buf_elements = 2 * B * S_local * num_heads * head_dim
        chunk_elements = B * S_local * num_heads * head_dim
        element_size = dtype.itemsize
        
        for step in range(cp_size):
            curr_buf_idx, next_buf_idx = step % 2, (step + 1) % 2
            
            if step > 0:
                cp_wait_count += 1
                _ext.wait_signal(my_cp_flag_ptr, cp_wait_count, 0)
                
            curr_k, curr_v = kv_buf[curr_buf_idx, 0], kv_buf[curr_buf_idx, 1]
            
            if step + 1 != cp_size:
                cp_push_count += 1
                peer_k_ptr = peer_kv_base + next_buf_idx * buf_elements * element_size
                peer_v_ptr = peer_kv_base + (next_buf_idx * buf_elements + chunk_elements) * element_size
                
                # Copy async into peer's symmetric memory and signal
                comm_stream.wait_stream(torch.cuda.current_stream())
                _ext.push_kv_and_signal(
                    curr_k, curr_v, peer_k_ptr, peer_v_ptr, peer_cp_flag_ptr, cp_push_count, comm_stream.cuda_stream
                )
                
            # Perform fused matmuls
            if not (causal and step > cp_rank):
                qh, kh, vh = q.transpose(1, 2).float(), curr_k.transpose(1, 2).float(), curr_v.transpose(1, 2).float()
                scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
                
                if causal and (step == 0):
                    mask = torch.triu(torch.ones(q.size(1), curr_k.size(1), device=device, dtype=torch.bool), 1)
                    scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    
                block_lse = torch.logsumexp(scores, dim=-1)
                block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous().to(dtype)
                
                if step == 0:
                    _ext.init_out_lse(out_buf, lse_buf, block_out, block_lse)
                else:
                    _ext.merge_out_lse(out_buf, lse_buf, block_out, block_lse)
                    
        state["cp_push_count"], state["cp_wait_count"] = cp_push_count, cp_wait_count
        ctx = out_buf.to(dtype)

    stage_output = F.linear(ctx.reshape(B, S_local, -1), w_o)

    # 3. Pipeline-Parallel Send
    if not is_last and pp_group is not None:
        if pp_push_count > 0:
            pp_ack_wait_count += 1
            _ext.wait_signal(my_pp_ack_ptr, pp_ack_wait_count, 0)
            
        pp_push_count += 1
        peer_rank = dist.get_global_rank(pp_group, (pp_rank + 1) % pp_size)
        _ext.push_pp_and_signal(
            stage_output, pp_ptrs[peer_rank], flags_ptrs[peer_rank] + 4, pp_push_count, 0
        )

    state["pp_push_count"], state["pp_wait_count"] = pp_push_count, pp_wait_count
    state["pp_ack_push_count"], state["pp_ack_wait_count"] = pp_ack_push_count, pp_ack_wait_count

    return stage_output