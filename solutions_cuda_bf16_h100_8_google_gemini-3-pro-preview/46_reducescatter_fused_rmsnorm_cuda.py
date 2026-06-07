import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16x8 = float4; // 16 bytes for 8 bfloat16s

__global__ void fused_rs_rmsnorm_kernel(
    const long long* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gamma,
    int world_size,
    int rank,
    int chunk,
    int hidden,
    int rows,
    float eps
) {
    int i = blockIdx.x; // process one row per block
    if (i >= rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    long long rank_offset = (long long)rank * chunk;
    long long row_offset = (long long)i * hidden;
    long long base_idx = rank_offset + row_offset;

    bool aligned = (hidden % 8 == 0);
    int vec_hidden = aligned ? hidden / 8 : 0;
    int tail_start = aligned ? hidden : 0;
    
    float sq_sum = 0.0f;

    // Pass 1: Reduce-scatter (sum and div), write to `out` intermediate, accumulate sq_sum
    for (int k_vec = tid; k_vec < vec_hidden; k_vec += stride) {
        float sums[8] = {0.0f};
        
        for (int p = 0; p < world_size; ++p) {
            const bf16x8* peer = reinterpret_cast<const bf16x8*>(peer_ptrs[p] + base_idx);
            bf16x8 vals = peer[k_vec];
            
            const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vals);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 pair = v2[j];
                const __nv_bfloat16* p_ptr = reinterpret_cast<const __nv_bfloat16*>(&pair);
                sums[j*2 + 0] += __bfloat162float(p_ptr[0]);
                sums[j*2 + 1] += __bfloat162float(p_ptr[1]);
            }
        }
        
        __nv_bfloat162 out_v2[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v0 = sums[j*2 + 0] / world_size;
            float v1 = sums[j*2 + 1] / world_size;
            
            __nv_bfloat16 b0 = __float2bfloat16(v0);
            __nv_bfloat16 b1 = __float2bfloat16(v1);
            
            __nv_bfloat162 b_pair;
            __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(&b_pair);
            b_ptr[0] = b0;
            b_ptr[1] = b1;
            out_v2[j] = b_pair;
            
            float f0 = __bfloat162float(b0);
            float f1 = __bfloat162float(b1);
            sq_sum += f0 * f0 + f1 * f1;
        }
        
        bf16x8* out_vec = reinterpret_cast<bf16x8*>(out + row_offset);
        out_vec[k_vec] = *reinterpret_cast<bf16x8*>(out_v2);
    }
    
    // Tail / scalar pass
    for (int k = tail_start + tid; k < hidden; k += stride) {
        float sum = 0.0f;
        for (int p = 0; p < world_size; ++p) {
            const __nv_bfloat16* peer = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[p]);
            sum += __bfloat162float(peer[base_idx + k]);
        }
        sum /= world_size;
        __nv_bfloat16 bval = __float2bfloat16(sum);
        out[row_offset + k] = bval;
        
        float fval = __bfloat162float(bval);
        sq_sum += fval * fval;
    }

    // Warp and Block reduction for sq_sum
    static __shared__ float shared_sq_sum[32]; // Accommodates up to 1024 threads
    unsigned int mask = 0xffffffff;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sq_sum += __shfl_down_sync(mask, sq_sum, offset);
    }
    
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        shared_sq_sum[wid] = sq_sum;
    }
    __syncthreads();
    
    float total_sq_sum = 0.0f;
    int num_warps = (blockDim.x + 31) / 32;
    
    if (wid == 0) {
        if (lane < num_warps) {
            total_sq_sum = shared_sq_sum[lane];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            total_sq_sum += __shfl_down_sync(mask, total_sq_sum, offset);
        }
        if (lane == 0) {
            shared_sq_sum[0] = total_sq_sum;
        }
    }
    __syncthreads();
    
    total_sq_sum = shared_sq_sum[0];
    float mean_sq = total_sq_sum / hidden;
    float rms = rsqrtf(mean_sq + eps);

    // Pass 2: Apply RMSNorm using `out` intermediate
    for (int k_vec = tid; k_vec < vec_hidden; k_vec += stride) {
        bf16x8* out_vec = reinterpret_cast<bf16x8*>(out + row_offset);
        bf16x8 vals = out_vec[k_vec];
        
        const bf16x8* gamma_vec_ptr = reinterpret_cast<const bf16x8*>(gamma);
        bf16x8 g_vals = gamma_vec_ptr[k_vec];
        
        const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vals);
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g_vals);
        
        __nv_bfloat162 out_v2[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 pair = v2[j];
            const __nv_bfloat16* p_ptr = reinterpret_cast<const __nv_bfloat16*>(&pair);
            float f_x = __bfloat162float(p_ptr[0]);
            float f_y = __bfloat162float(p_ptr[1]);
            
            __nv_bfloat162 g_pair = g2[j];
            const __nv_bfloat16* g_ptr = reinterpret_cast<const __nv_bfloat16*>(&g_pair);
            float g_x = __bfloat162float(g_ptr[0]);
            float g_y = __bfloat162float(g_ptr[1]);
            
            float v0 = f_x * rms * g_x;
            float v1 = f_y * rms * g_y;
            
            __nv_bfloat16 b0 = __float2bfloat16(v0);
            __nv_bfloat16 b1 = __float2bfloat16(v1);
            
            __nv_bfloat162 b_pair;
            __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(&b_pair);
            b_ptr[0] = b0;
            b_ptr[1] = b1;
            out_v2[j] = b_pair;
        }
        out_vec[k_vec] = *reinterpret_cast<bf16x8*>(out_v2);
    }
    
    // Tail / scalar pass
    for (int k = tail_start + tid; k < hidden; k += stride) {
        float val = __bfloat162float(out[row_offset + k]);
        float g = __bfloat162float(gamma[k]);
        out[row_offset + k] = __float2bfloat16(val * rms * g);
    }
}

void launch_fused_rs_rmsnorm(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    torch::Tensor gamma,
    int world_size,
    int rank,
    int chunk,
    int hidden,
    int rows,
    float eps
) {
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    __nv_bfloat16* d_out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    const __nv_bfloat16* d_gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());

    int threads = 256;
    int blocks = rows;
    if (blocks == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    fused_rs_rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, d_out, d_gamma, world_size, rank, chunk, hidden, rows, eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_rs_rmsnorm", &launch_fused_rs_rmsnorm, "Fused Reduce-Scatter and RMSNorm");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_rs_rmsnorm_ext", CUDA_SRC)
    return _ext


_resource_cache = {}

def _get_resources(n: int, dtype: torch.dtype, device: torch.device):
    """
    Returns handles, pointer tensors, and buffers.
    Uses double-buffering mapped to symm_mem to avoid blocking CPU syncs 
    while preventing buffer overwrites in tight recurrent loops.
    """
    key = (n, dtype, device)
    if key not in _resource_cache:
        bufs = []
        hdls = []
        ptrs = []
        for _ in range(2):
            buf = symm_mem.empty(n, device=device, dtype=dtype)
            hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
            ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
            bufs.append(buf)
            hdls.append(hdl)
            ptrs.append(ptrs_tensor)
        _resource_cache[key] = {'bufs': bufs, 'hdls': hdls, 'ptrs': ptrs, 'idx': 0}
        
    cache = _resource_cache[key]
    idx = cache['idx']
    cache['idx'] = (idx + 1) % 2
    return cache['bufs'][idx], cache['hdls'][idx], cache['ptrs'][idx]


@torch.no_grad()
def solution(
    rs_input_1d: Tensor,
    gamma: Tensor,
    eps: float,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    n = rs_input_1d.numel()
    assert n % world_size == 0
    chunk = n // world_size

    hidden = gamma.numel()
    assert chunk % hidden == 0, f"chunk ({chunk}) must divide hidden ({hidden})"
    rows = chunk // hidden

    input_bf16 = rs_input_1d.contiguous()
    if input_bf16.dtype != torch.bfloat16:
        input_bf16 = input_bf16.to(torch.bfloat16)

    gamma_bf16 = gamma.contiguous()
    if gamma_bf16.dtype != torch.bfloat16:
        gamma_bf16 = gamma_bf16.to(torch.bfloat16)

    if rank == 0:
        _get_ext()
    dist.barrier()  # Synchronize to ensure cleanly initialized CUDA compilation step limits out-of-order execution

    buf, hdl, ptrs_tensor = _get_resources(n, torch.bfloat16, input_bf16.device)
    
    # Ensure current writes to symm_mem complete correctly
    buf.copy_(input_bf16)
    hdl.barrier(channel=0)

    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=input_bf16.device)
    
    _get_ext().launch_fused_rs_rmsnorm(
        ptrs_tensor,
        out,
        gamma_bf16,
        world_size,
        rank,
        chunk,
        hidden,
        rows,
        eps
    )
    
    if out.dtype != rs_input_1d.dtype:
        out = out.to(rs_input_1d.dtype)

    return out

__all__ = ["solution"]