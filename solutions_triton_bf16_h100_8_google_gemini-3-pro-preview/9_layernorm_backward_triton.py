import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

union U32_BF162 {
    uint32_t u;
    __nv_bfloat162 bf2;
};

template <int MAX_N=16>
struct PtrArray {
    const float* ptrs[MAX_N];
};

__global__ void local_reduce_kernel(
    const __nv_bfloat16* __restrict__ X_hat,
    const __nv_bfloat16* __restrict__ dY,
    float* __restrict__ d_gamma_local,
    float* __restrict__ d_beta_local,
    int B, int H)
{
    int h4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int rows_per_block = (B + gridDim.y - 1) / gridDim.y;
    int b_start = blockIdx.y * rows_per_block;
    int b_end = b_start + rows_per_block;
    if (b_end > B) b_end = B;

    // Fast path: 8-byte vectorized loads when H is aligned
    if (H % 4 == 0 && h4 + 3 < H) {
        float acc_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int b = b_start; b < b_end; ++b) {
            size_t idx = (size_t)b * H + h4;
            uint2 x_u2 = *reinterpret_cast<const uint2*>(&X_hat[idx]);
            uint2 dy_u2 = *reinterpret_cast<const uint2*>(&dY[idx]);

            U32_BF162 cvt_x0, cvt_x1, cvt_dy0, cvt_dy1;
            cvt_x0.u = x_u2.x;
            cvt_x1.u = x_u2.y;
            cvt_dy0.u = dy_u2.x;
            cvt_dy1.u = dy_u2.y;

            // Direct conversion intrinsically supported on Hopper (SM90)
            float2 x_f0 = __bfloat1622float2(cvt_x0.bf2);
            float2 x_f1 = __bfloat1622float2(cvt_x1.bf2);
            float2 dy_f0 = __bfloat1622float2(cvt_dy0.bf2);
            float2 dy_f1 = __bfloat1622float2(cvt_dy1.bf2);

            acc_g[0] += dy_f0.x * x_f0.x;
            acc_g[1] += dy_f0.y * x_f0.y;
            acc_g[2] += dy_f1.x * x_f1.x;
            acc_g[3] += dy_f1.y * x_f1.y;

            acc_b[0] += dy_f0.x;
            acc_b[1] += dy_f0.y;
            acc_b[2] += dy_f1.x;
            acc_b[3] += dy_f1.y;
        }

        if (acc_g[0] != 0.0f) atomicAdd(&d_gamma_local[h4+0], acc_g[0]);
        if (acc_g[1] != 0.0f) atomicAdd(&d_gamma_local[h4+1], acc_g[1]);
        if (acc_g[2] != 0.0f) atomicAdd(&d_gamma_local[h4+2], acc_g[2]);
        if (acc_g[3] != 0.0f) atomicAdd(&d_gamma_local[h4+3], acc_g[3]);

        if (acc_b[0] != 0.0f) atomicAdd(&d_beta_local[h4+0], acc_b[0]);
        if (acc_b[1] != 0.0f) atomicAdd(&d_beta_local[h4+1], acc_b[1]);
        if (acc_b[2] != 0.0f) atomicAdd(&d_beta_local[h4+2], acc_b[2]);
        if (acc_b[3] != 0.0f) atomicAdd(&d_beta_local[h4+3], acc_b[3]);

    } else {
        // Scalar fallback for non-aligned tails
        int h_end = h4 + 4;
        if (h_end > H) h_end = H;
        for (int h = h4; h < h_end; ++h) {
            float acc_g = 0.0f;
            float acc_b = 0.0f;
            for (int b = b_start; b < b_end; ++b) {
                size_t idx = (size_t)b * H + h;
                float x = __bfloat162float(X_hat[idx]);
                float dy = __bfloat162float(dY[idx]);
                acc_g += dy * x;
                acc_b += dy;
            }
            if (acc_g != 0.0f) atomicAdd(&d_gamma_local[h], acc_g);
            if (acc_b != 0.0f) atomicAdd(&d_beta_local[h], acc_b);
        }
    }
}

__global__ void all_reduce_kernel_vec(
    PtrArray<16> gamma_ptrs,
    PtrArray<16> beta_ptrs,
    __nv_bfloat16* __restrict__ d_gamma_out,
    __nv_bfloat16* __restrict__ d_beta_out,
    int H, int N)
{
    int h4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (H % 4 == 0 && h4 + 3 < H) {
        float g_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float b_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 8
        for (int i = 0; i < N; ++i) {
            float4 g_val = *reinterpret_cast<const float4*>(&gamma_ptrs.ptrs[i][h4]);
            float4 b_val = *reinterpret_cast<const float4*>(&beta_ptrs.ptrs[i][h4]);
            
            g_sum[0] += g_val.x; g_sum[1] += g_val.y; g_sum[2] += g_val.z; g_sum[3] += g_val.w;
            b_sum[0] += b_val.x; b_sum[1] += b_val.y; b_sum[2] += b_val.z; b_sum[3] += b_val.w;
        }

        __nv_bfloat162 g01 = __floats2bfloat162_rn(g_sum[0], g_sum[1]);
        __nv_bfloat162 g23 = __floats2bfloat162_rn(g_sum[2], g_sum[3]);
        __nv_bfloat162 b01 = __floats2bfloat162_rn(b_sum[0], b_sum[1]);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(b_sum[2], b_sum[3]);

        U32_BF162 cvt_g01, cvt_g23, cvt_b01, cvt_b23;
        cvt_g01.bf2 = g01;
        cvt_g23.bf2 = g23;
        cvt_b01.bf2 = b01;
        cvt_b23.bf2 = b23;

        uint2 g_out, b_out;
        g_out.x = cvt_g01.u;
        g_out.y = cvt_g23.u;
        b_out.x = cvt_b01.u;
        b_out.y = cvt_b23.u;

        *reinterpret_cast<uint2*>(&d_gamma_out[h4]) = g_out;
        *reinterpret_cast<uint2*>(&d_beta_out[h4])  = b_out;

    } else {
        int h_end = h4 + 4;
        if (h_end > H) h_end = H;
        for (int h = h4; h < h_end; ++h) {
            float g_s = 0.0f;
            float b_s = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < N; ++i) {
                g_s += gamma_ptrs.ptrs[i][h];
                b_s += beta_ptrs.ptrs[i][h];
            }
            d_gamma_out[h] = __float2bfloat16(g_s);
            d_beta_out[h]  = __float2bfloat16(b_s);
        }
    }
}

void run_local_reduce(
    torch::Tensor X_hat,
    torch::Tensor dY,
    torch::Tensor d_gamma_local,
    torch::Tensor d_beta_local,
    int B, int H)
{
    int threads = 256;
    int blocks_x = (H + 4 * threads - 1) / (4 * threads);
    // 128 waves naturally saturates Hopper SMs for atomic ops
    int blocks_y = 128;
    if (B < 128) blocks_y = B; 
    
    dim3 blocks(blocks_x, blocks_y);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    local_reduce_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X_hat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        d_gamma_local.data_ptr<float>(),
        d_beta_local.data_ptr<float>(),
        B, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_all_reduce(
    std::vector<int64_t> gamma_ptr_ints,
    std::vector<int64_t> beta_ptr_ints,
    torch::Tensor d_gamma_out,
    torch::Tensor d_beta_out,
    int H, int N)
{
    PtrArray<16> g_ptrs, b_ptrs;
    for (int i = 0; i < N; ++i) {
        g_ptrs.ptrs[i] = reinterpret_cast<const float*>(gamma_ptr_ints[i]);
        b_ptrs.ptrs[i] = reinterpret_cast<const float*>(beta_ptr_ints[i]);
    }

    int threads = 256;
    int blocks = (H + 4 * threads - 1) / (4 * threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    all_reduce_kernel_vec<<<blocks, threads, 0, stream>>>(
        g_ptrs,
        b_ptrs,
        reinterpret_cast<__nv_bfloat16*>(d_gamma_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(d_beta_out.data_ptr()),
        H, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_local_reduce", &run_local_reduce, "Local reduction for LayerNorm backward");
    m.def("run_all_reduce", &run_all_reduce, "AllReduce over UVA pointers directly to BF16");
}
'''

_ext = None
_compiled = False

def _get_ext():
    global _ext, _compiled
    if not _compiled:
        # Avoid concurrent multi-rank compilation races 
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("layernorm_bw_symm_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("layernorm_bw_symm_ext", CUDA_SRC)
        _compiled = True
    return _ext

_symm_cache = {}

def _get_symm_state(H: int, device: torch.device):
    global _symm_cache
    if H in _symm_cache:
        return _symm_cache[H]

    # Combine gamma and beta into contiguous symmetric float32 memory chunks per rank
    buf = symm_mem.empty(2 * H, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    _symm_cache[H] = (buf, hdl)
    return buf, hdl

@torch.no_grad()
def solution(
    X_hat: torch.Tensor,
    dY: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert X_hat.is_cuda and dY.is_cuda, "Inputs must be CUDA tensors"
    assert X_hat.is_contiguous() and dY.is_contiguous(), "Inputs must be contiguous"
    assert X_hat.shape == dY.shape, "X_hat and dY must have the same shape [B, H]"

    world_size = dist.get_world_size()
    B, H = X_hat.shape

    # Extension compilation & load logic (ensured to sync cleanly)
    ext = _get_ext()

    # The kernels natively handle optimal bfloat16 routing.
    orig_dtype = X_hat.dtype
    if orig_dtype != torch.bfloat16:
        X_hat = X_hat.to(torch.bfloat16)
    if dY.dtype != torch.bfloat16:
        dY = dY.to(torch.bfloat16)

    # Acquire symmetric memory handles (cached tightly based on size H)
    buf, hdl = _get_symm_state(H, X_hat.device)
    
    # Ready accumulator buffer before kernels operate via atomics
    buf.zero_()
    d_gamma_local = buf[:H]
    d_beta_local = buf[H:]

    # Rapid partial row sum & write locally 
    ext.run_local_reduce(X_hat, dY, d_gamma_local, d_beta_local, B, H)

    # Barrier: Wait for all peers' partials to arrive in their local symmetrically-backed buffers
    hdl.barrier(channel=0)

    gamma_ptrs = [int(p) for p in hdl.buffer_ptrs]
    beta_ptrs = [int(p) + H * 4 for p in hdl.buffer_ptrs]

    d_gamma_out = torch.empty(H, device=X_hat.device, dtype=torch.bfloat16)
    d_beta_out = torch.empty(H, device=X_hat.device, dtype=torch.bfloat16)

    # P2P AllReduce kernel (directly reads remote values across nodes, returns global BF16 output)
    ext.run_all_reduce(gamma_ptrs, beta_ptrs, d_gamma_out, d_beta_out, H, world_size)

    # Barrier 2: Enforce completion strictly to avoid any subsequent LayerNorm BW 
    # resetting this specific `buf` zero state while peers are still finishing UVA pulls.
    hdl.barrier(channel=0)

    if orig_dtype != torch.bfloat16:
        d_gamma_out = d_gamma_out.to(orig_dtype)
        d_beta_out = d_beta_out.to(orig_dtype)

    return d_gamma_out, d_beta_out