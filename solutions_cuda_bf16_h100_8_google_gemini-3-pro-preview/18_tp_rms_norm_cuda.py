"""
Strategy:
- **Kernel Fusion**: Combines local sum-of-squares computation, all-reduce communication, and scaling/normalization into a single optimized CUDA kernel. This prevents repeated round-trips to HBM for the massive hidden-states tensor.
- **Device-Side Communication via UVA**: Bypasses heavy NCCL collectives by using `torch.distributed._symmetric_memory` to allocate P2P-accessible buffers over NVLink. The kernel directly writes local sums and sequence flags into peer memory.
- **Compute-Communication Overlap**: Employs a persistent thread-block grid where each block handles independent rows. Thread 0 busy-waits on peer flags to fetch global sums, overlapping the fast intra-row communication directly with adjacent row computations and avoiding global barrier deadlocks.
- **Hardware Barrier Safety**: Utilizes `hdl.barrier(channel=0)`—a fast device-side hardware stream sync—before kernel launch to safely reuse persistent sync buffers across calls without risking race conditions or Python-side serialization delays.
- **Vectorized Memory Access**: Checks alignment dynamically to employ `uint4` (128-bit) vectorized loads and stores when processing `bfloat16` data, effectively doubling memory throughput.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void fused_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    const uint64_t* __restrict__ peer_sums_ptrs,
    const uint64_t* __restrict__ peer_flags_ptrs,
    int M,
    int local_hidden_size,
    float epsilon,
    int world_size,
    int rank,
    uint64_t seq,
    int global_hidden_size,
    bool aligned
) {
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const __nv_bfloat16* row_in = input + row * local_hidden_size;
        __nv_bfloat16* row_out = output + row * local_hidden_size;

        float local_sum = 0.0f;
        int tail_start = 0;

        if (aligned) {
            int col = threadIdx.x * 8;
            int limit = local_hidden_size; // guaranteed multiple of 8 if aligned
            
            for (; col < limit; col += blockDim.x * 8) {
                uint4 vals = *(reinterpret_cast<const uint4*>(&row_in[col]));
                __nv_bfloat162* halfs = reinterpret_cast<__nv_bfloat162*>(&vals);
                
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 f2 = __bfloat1622float2(halfs[i]);
                    local_sum += f2.x * f2.x;
                    local_sum += f2.y * f2.y;
                }
            }
            tail_start = limit; 
        }

        for (int c = tail_start + threadIdx.x; c < local_hidden_size; c += blockDim.x) {
            float val = __bfloat162float(row_in[c]);
            local_sum += val * val;
        }

        // Warp block reduce local_sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        __shared__ float s_data[32];
        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        
        if (lane == 0) {
            s_data[warp] = local_sum;
        }
        __syncthreads();

        if (warp == 0) {
            float val = (lane < (blockDim.x / 32)) ? s_data[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane == 0) {
                float* my_sums = reinterpret_cast<float*>(peer_sums_ptrs[rank]);
                uint64_t* my_flags = reinterpret_cast<uint64_t*>(peer_flags_ptrs[rank]);
                
                my_sums[row] = val;
                asm volatile("fence.acq_rel.sys;" ::: "memory");
                asm volatile("st.global.release.sys.b64 [%0], %1;" :: "l"(&my_flags[row]), "l"(seq) : "memory");
            }
        }
        __syncthreads();

        // Cross-GPU sync and summation
        __shared__ float s_global_sum;
        if (threadIdx.x == 0) {
            float g_sum = 0.0f;
            for (int p = 0; p < world_size; ++p) {
                float* peer_sums = reinterpret_cast<float*>(peer_sums_ptrs[p]);
                uint64_t* peer_flags = reinterpret_cast<uint64_t*>(peer_flags_ptrs[p]);
                
                uint64_t flag_val = 0;
                do {
                    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(flag_val) : "l"(&peer_flags[row]) : "memory");
                } while (flag_val != seq);
                
                g_sum += peer_sums[row];
            }
            s_global_sum = g_sum;
        }
        __syncthreads();

        float global_sum = s_global_sum;
        float variance = global_sum / static_cast<float>(global_hidden_size);
        float rsqrt_var = rsqrtf(variance + epsilon);

        // Scale and Output
        if (aligned) {
            int col = threadIdx.x * 8;
            int limit = local_hidden_size;
            
            for (; col < limit; col += blockDim.x * 8) {
                uint4 in_vals = *(reinterpret_cast<const uint4*>(&row_in[col]));
                uint4 w_vals = *(reinterpret_cast<const uint4*>(&weight[col]));
                
                __nv_bfloat162* in_halfs = reinterpret_cast<__nv_bfloat162*>(&in_vals);
                __nv_bfloat162* w_halfs = reinterpret_cast<__nv_bfloat162*>(&w_vals);
                
                uint4 out_vals;
                __nv_bfloat162* out_halfs = reinterpret_cast<__nv_bfloat162*>(&out_vals);
                
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 f_in = __bfloat1622float2(in_halfs[i]);
                    float2 f_w = __bfloat1622float2(w_halfs[i]);
                    
                    float2 f_out;
                    f_out.x = f_in.x * rsqrt_var * f_w.x;
                    f_out.y = f_in.y * rsqrt_var * f_w.y;
                    out_halfs[i] = __float22bfloat162_rn(f_out);
                }
                *(reinterpret_cast<uint4*>(&row_out[col])) = out_vals;
            }
        } else {
            for (int c = threadIdx.x; c < local_hidden_size; c += blockDim.x) {
                float val = __bfloat162float(row_in[c]);
                float w = __bfloat162float(weight[c]);
                row_out[c] = __float2bfloat16(val * rsqrt_var * w);
            }
        }
    }
}

void launch_fused_rmsnorm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor peer_sums_ptrs,
    torch::Tensor peer_flags_ptrs,
    float epsilon,
    int world_size,
    int rank,
    int64_t seq
) {
    int M = input.numel() / input.size(-1);
    int local_hidden_size = input.size(-1);
    int global_hidden_size = local_hidden_size * world_size;

    bool aligned = (local_hidden_size % 8 == 0) &&
                   ((uintptr_t)input.data_ptr() % 16 == 0) &&
                   ((uintptr_t)weight.data_ptr() % 16 == 0) &&
                   ((uintptr_t)output.data_ptr() % 16 == 0);

    int threads = 256;
    // Cap blocks to guarantee thread co-residency, thus preventing deadlocks on persistent flags
    int blocks = M < 128 ? M : 128; 
    if (blocks == 0) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    fused_rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint64_t*>(peer_sums_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(peer_flags_ptrs.data_ptr<int64_t>()),
        M,
        local_hidden_size,
        epsilon,
        world_size,
        rank,
        static_cast<uint64_t>(seq),
        global_hidden_size,
        aligned
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_rmsnorm", &launch_fused_rmsnorm, "Fused RMSNorm with P2P allreduce");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_rmsnorm_ext", CUDA_SRC)
    return _ext


class SymmMemManager:
    def __init__(self):
        self.cache = {}

    def get_buffers(self, M, device):
        if M in self.cache:
            return self.cache[M]
        
        # Calculate offsets considering 8-byte alignment bounds for uint64 flags
        sums_bytes = (M * 4 + 7) & ~7
        flags_bytes = M * 8
        total_bytes = sums_bytes + flags_bytes
        
        buf = symm_mem.empty(total_bytes, dtype=torch.int8, device=device)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        
        sums_ptrs = []
        flags_ptrs = []
        for p in hdl.buffer_ptrs:
            sums_ptrs.append(p)
            flags_ptrs.append(p + sums_bytes)
            
        sums_ptrs_t = torch.tensor(sums_ptrs, dtype=torch.int64, device=device)
        flags_ptrs_t = torch.tensor(flags_ptrs, dtype=torch.int64, device=device)
        
        self.cache[M] = (buf, hdl, sums_ptrs_t, flags_ptrs_t)
        return self.cache[M]

_symm_manager = SymmMemManager()
_seq_counter = 1


@torch.no_grad()
def solution(local_hidden_states: torch.Tensor, local_weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = local_hidden_states.dtype
    
    # Pure PyTorch fallback if disconnected or handling alternate dtypes
    if input_dtype != torch.bfloat16 or not dist.is_initialized() or dist.get_world_size() == 1:
        fp32_states = local_hidden_states.to(torch.float32)
        local_sum_squares = fp32_states.pow(2).sum(dim=-1, keepdim=True)
        
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(local_sum_squares, op=dist.ReduceOp.SUM)
                
        global_hidden_size = local_hidden_states.shape[-1] * world_size
        variance = local_sum_squares / global_hidden_size
        out = local_hidden_states * torch.rsqrt(variance + variance_epsilon)
        return local_weight * out.to(input_dtype)

    global _seq_counter
    if _ext is None:
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        _get_ext()
        
    input_tensor = local_hidden_states.contiguous()
    weight_tensor = local_weight.to(torch.bfloat16).contiguous()
    output_tensor = torch.empty_like(input_tensor)
    
    M = input_tensor.numel() // input_tensor.size(-1)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    seq = _seq_counter
    _seq_counter += 1
    
    _, hdl, sums_ptrs_t, flags_ptrs_t = _symm_manager.get_buffers(M, input_tensor.device)
    
    # Device-side stream barrier enforcing order against previously resident uses of this persistent buffer
    hdl.barrier(channel=0)
    
    _get_ext().launch_fused_rmsnorm(
        input_tensor,
        weight_tensor,
        output_tensor,
        sums_ptrs_t,
        flags_ptrs_t,
        float(variance_epsilon),
        world_size,
        rank,
        seq
    )
    
    return output_tensor