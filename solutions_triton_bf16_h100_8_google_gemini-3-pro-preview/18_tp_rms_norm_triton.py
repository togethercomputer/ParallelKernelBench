import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

#define MAX_RANKS 32

struct PeerPtrs {
    const float* ptrs[MAX_RANKS];
};

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (wid == 0) {
        val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void rmsnorm_sq_sum_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ local_sq_sum,
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    int tid = threadIdx.x;
    const __nv_bfloat16* row_input = input + row * D;
    
    float sum = 0.0f;
    
    // Fast path: Vectorized float4 read for multiple of 8
    if (D % 8 == 0) {
        int D8 = D / 8;
        const float4* row_input_f4 = reinterpret_cast<const float4*>(row_input);
        for (int i = tid; i < D8; i += blockDim.x) {
            float4 vecs = row_input_f4[i];
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&vecs);
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __bfloat1622float2(h2[j]);
                sum += f2.x * f2.x + f2.y * f2.y;
            }
        }
    } else {
        // Fallback scalar path
        for (int i = tid; i < D; i += blockDim.x) {
            float val = __bfloat162float(row_input[i]);
            sum += val * val;
        }
    }
    
    sum = blockReduceSum(sum);
    
    if (tid == 0) {
        local_sq_sum[row] = sum;
    }
}

__global__ void rmsnorm_norm_scale_kernel(
    const __nv_bfloat16* __restrict__ input,
    PeerPtrs peer_sq_sums,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float epsilon,
    int N, int D, int world_size, int global_D
) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    int tid = threadIdx.x;
    __shared__ float s_scale;
    
    // Thread 0 calculates total variance from symmetrically visible remote metrics
    if (tid == 0) {
        float total_sq_sum = 0.0f;
        for (int i = 0; i < world_size; ++i) {
            total_sq_sum += peer_sq_sums.ptrs[i][row];
        }
        float variance = total_sq_sum / global_D;
        s_scale = rsqrtf(variance + epsilon);
    }
    __syncthreads();
    
    float scale = s_scale;
    const __nv_bfloat16* row_input = input + row * D;
    __nv_bfloat16* row_output = output + row * D;
    
    if (D % 8 == 0) {
        int D8 = D / 8;
        const float4* row_input_f4 = reinterpret_cast<const float4*>(row_input);
        const float4* weight_f4 = reinterpret_cast<const float4*>(weight);
        float4* row_output_f4 = reinterpret_cast<float4*>(row_output);
        
        for (int i = tid; i < D8; i += blockDim.x) {
            float4 in_vec = row_input_f4[i];
            float4 w_vec = weight_f4[i];
            float4 out_vec;
            
            const __nv_bfloat162* in_h2 = reinterpret_cast<const __nv_bfloat162*>(&in_vec);
            const __nv_bfloat162* w_h2 = reinterpret_cast<const __nv_bfloat162*>(&w_vec);
            __nv_bfloat162* out_h2 = reinterpret_cast<__nv_bfloat162*>(&out_vec);
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 in_f2 = __bfloat1622float2(in_h2[j]);
                float2 w_f2 = __bfloat1622float2(w_h2[j]);
                
                float2 out_f2;
                out_f2.x = in_f2.x * scale * w_f2.x;
                out_f2.y = in_f2.y * scale * w_f2.y;
                out_h2[j] = __float22bfloat162_rn(out_f2);
            }
            row_output_f4[i] = out_vec;
        }
    } else {
        for (int i = tid; i < D; i += blockDim.x) {
            float val = __bfloat162float(row_input[i]);
            float w = __bfloat162float(weight[i]);
            row_output[i] = __float2bfloat16(val * scale * w);
        }
    }
}

void rmsnorm_sq_sum(
    torch::Tensor local_hidden_states,
    torch::Tensor local_sq_sum
) {
    int N = local_hidden_states.numel() / local_hidden_states.size(-1);
    int D = local_hidden_states.size(-1);
    
    int threads = 256;
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (N > 0) {
        rmsnorm_sq_sum_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(local_hidden_states.data_ptr<at::BFloat16>()),
            local_sq_sum.data_ptr<float>(),
            N, D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void rmsnorm_norm_scale(
    torch::Tensor local_hidden_states,
    std::vector<int64_t> remote_sq_sum_ptrs,
    torch::Tensor local_weight,
    torch::Tensor output,
    float epsilon,
    int global_D
) {
    int N = local_hidden_states.numel() / local_hidden_states.size(-1);
    int D = local_hidden_states.size(-1);
    int world_size = remote_sq_sum_ptrs.size();
    
    PeerPtrs peer_ptrs;
    for (int i = 0; i < world_size && i < MAX_RANKS; ++i) {
        peer_ptrs.ptrs[i] = reinterpret_cast<const float*>(remote_sq_sum_ptrs[i]);
    }
    
    int threads = 256;
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (N > 0) {
        rmsnorm_norm_scale_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(local_hidden_states.data_ptr<at::BFloat16>()),
            peer_ptrs,
            reinterpret_cast<const __nv_bfloat16*>(local_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            epsilon,
            N, D, world_size, global_D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_sq_sum", &rmsnorm_sq_sum, "Compute local sum of squares row-wise");
    m.def("rmsnorm_norm_scale", &rmsnorm_norm_scale, "Compute global variance, scale & normalize via UVA");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("tp_rmsnorm_bf16_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = None

def _get_symm_state(n: int, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] >= n and c["device"] == device:
            return c["buf"], c["hdl"]

    # Allocate enough powers of 2 size to gracefully handle dynamic seq len caching without stalls
    alloc_n = max(1024, 1 << (max(n, 1) - 1).bit_length())
    buf = symm_mem.empty(alloc_n, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"n": alloc_n, "device": device, "buf": buf, "hdl": hdl}
    
    return buf, hdl


@torch.no_grad()
def solution(local_hidden_states: torch.Tensor, local_weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    # Ensure initialized context to sidestep races
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    
    input_dtype = local_hidden_states.dtype
    
    # Constrain to bfloat16 hot path as natively supported hardware requirement bounds
    lhs_bf16 = local_hidden_states.to(torch.bfloat16).contiguous()
    lw_bf16 = local_weight.to(torch.bfloat16).contiguous()
    
    N = lhs_bf16.numel() // lhs_bf16.size(-1)
    D = lhs_bf16.size(-1)
    
    world_size = dist.get_world_size()
    global_D = D * world_size
    
    ext = _get_ext()
    buf, hdl = _get_symm_state(N, lhs_bf16.device)
    
    # Step 1: Push local metric logic 
    # Computed on bfloat16, accumulated up on float32 natively within the SM
    ext.rmsnorm_sq_sum(lhs_bf16, buf)
    
    # Barrier streams out to sync peers efficiently
    hdl.barrier(channel=0)
    
    remote_ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]
    
    # Step 2: Extract globally, map variance out
    out_bf16 = torch.empty_like(lhs_bf16)
    ext.rmsnorm_norm_scale(
        lhs_bf16, 
        remote_ptrs, 
        lw_bf16, 
        out_bf16, 
        variance_epsilon, 
        global_D
    )
    
    return out_bf16.to(input_dtype)