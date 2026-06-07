"""
Strategy:
- Direct Peer-to-Peer Push/Pull: Replaced NCCL all-to-all and reduce-scatter with custom CUDA kernels executing direct device-to-device memory accesses over NVLink via symmetric_memory UVA pointers, bypassing PyTorch overhead and intermediate buffers.
- Fused Reshaping: Fused the complex tensor reshaping and transpositions inherent in the longitude/channel communication directly into the P2P read/write index math, eliminating separate `torch.split` and `torch.cat` overheads.
- Device-Side Reduce: Evaluated the polar group sum concurrently across peers by pulling remote symmetric buffers directly into local FP32 registers for accumulation and casting back to BF16, fully skipping allocation-heavy `all_reduce` + slicing.
- Fused Final Contraction: Grouped channel mixing and bias addition are performed with a single custom Triton kernel that dynamically decodes the strided memory layout, producing the final output in-place and hiding the bias addition.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from typing import List, Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

struct Offsets {
    int data[32];
};

__global__ void azimuth_a2a_fwd_push_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uintptr_t* __restrict__ Y_ptrs,
    Offsets c_offsets,
    Offsets lon_offsets,
    int B, int C, int nlat, int lon_local_size, int nlon_in,
    int rank)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * nlat * lon_local_size;
    if (idx >= total) return;

    int lon_idx = idx % lon_local_size;
    int lat_idx = (idx / lon_local_size) % nlat;
    int c_idx = (idx / (lon_local_size * nlat)) % C;
    int b_idx = idx / (lon_local_size * nlat * C);

    int j = 0;
    while (c_idx >= c_offsets.data[j+1]) j++;
    
    int dst_c = c_idx - c_offsets.data[j];
    int dst_lon = lon_idx + lon_offsets.data[rank];

    int C_split_j = c_offsets.data[j+1] - c_offsets.data[j];
    __nv_bfloat16* Y_j = reinterpret_cast<__nv_bfloat16*>(Y_ptrs[j]);

    int dst_idx = b_idx * (C_split_j * nlat * nlon_in) +
                  dst_c * (nlat * nlon_in) +
                  lat_idx * nlon_in +
                  dst_lon;

    Y_j[dst_idx] = X[idx];
}

__global__ void polar_reduce_scatter_pull_kernel(
    const uintptr_t* __restrict__ X_ptrs,
    __nv_bfloat16* __restrict__ Y,
    Offsets lat_offsets,
    int B, int C, int K, int nlat_out, int nlon_out,
    int rank, int P)
{
    int lat_local_size = lat_offsets.data[rank+1] - lat_offsets.data[rank];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * K * lat_local_size * nlon_out;
    if (idx >= total) return;

    int lon_idx = idx % nlon_out;
    int lat_idx = (idx / nlon_out) % lat_local_size;
    int k_idx = (idx / (nlon_out * lat_local_size)) % K;
    int c_idx = (idx / (nlon_out * lat_local_size * K)) % C;
    int b_idx = idx / (nlon_out * lat_local_size * K * C);

    int global_lat_idx = lat_idx + lat_offsets.data[rank];

    int src_idx = b_idx * (C * K * nlat_out * nlon_out) +
                  c_idx * (K * nlat_out * nlon_out) +
                  k_idx * (nlat_out * nlon_out) +
                  global_lat_idx * nlon_out +
                  lon_idx;

    float sum = 0.0f;
    for (int q = 0; q < P; q++) {
        const __nv_bfloat16* X_q = reinterpret_cast<const __nv_bfloat16*>(X_ptrs[q]);
        sum += __bfloat162float(X_q[src_idx]);
    }
    Y[idx] = __float2bfloat16(sum);
}

__global__ void azimuth_a2a_bwd_push_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uintptr_t* __restrict__ Y_ptrs,
    Offsets c_offsets,
    Offsets lon_offsets,
    int B, int C_local, int K, int nlat_local, int nlon_out,
    int num_chans, int rank)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_local * K * nlat_local * nlon_out;
    if (idx >= total) return;

    int lon = idx % nlon_out;
    int lat = (idx / nlon_out) % nlat_local;
    int k = (idx / (nlon_out * nlat_local)) % K;
    int c = (idx / (nlon_out * nlat_local * K)) % C_local;
    int b = idx / (nlon_out * nlat_local * K * C_local);

    int j = 0;
    while (lon >= lon_offsets.data[j+1]) j++;

    int dst_lon = lon - lon_offsets.data[j];
    int dst_c = c + c_offsets.data[rank];

    int lon_out_local_j = lon_offsets.data[j+1] - lon_offsets.data[j];

    int dst_idx = b * (num_chans * K * nlat_local * lon_out_local_j) +
                  dst_c * (K * nlat_local * lon_out_local_j) +
                  k * (nlat_local * lon_out_local_j) +
                  lat * lon_out_local_j +
                  dst_lon;

    __nv_bfloat16* Y_j = reinterpret_cast<__nv_bfloat16*>(Y_ptrs[j]);
    Y_j[dst_idx] = X[idx];
}

void azimuth_a2a_fwd(
    torch::Tensor x,
    torch::Tensor y_ptrs_tensor,
    std::vector<int> c_offs,
    std::vector<int> lon_offs,
    int B, int C, int nlat, int lon_local_size, int nlon_in,
    int rank)
{
    Offsets c, lon;
    for(size_t i=0; i<c_offs.size(); i++) c.data[i] = c_offs[i];
    for(size_t i=0; i<lon_offs.size(); i++) lon.data[i] = lon_offs[i];

    int total = B * C * nlat * lon_local_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    azimuth_a2a_fwd_push_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uintptr_t*>(y_ptrs_tensor.data_ptr<int64_t>()),
        c, lon, B, C, nlat, lon_local_size, nlon_in, rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void polar_reduce_scatter(
    torch::Tensor x_ptrs_tensor,
    torch::Tensor y,
    std::vector<int> lat_offs,
    int B, int C, int K, int nlat_out, int nlon_out,
    int rank, int P)
{
    Offsets lat;
    for(size_t i=0; i<lat_offs.size(); i++) lat.data[i] = lat_offs[i];

    int lat_local = lat_offs[rank+1] - lat_offs[rank];
    int total = B * C * K * lat_local * nlon_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    polar_reduce_scatter_pull_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uintptr_t*>(x_ptrs_tensor.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        lat, B, C, K, nlat_out, nlon_out, rank, P
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void azimuth_a2a_bwd(
    torch::Tensor x,
    torch::Tensor y_ptrs_tensor,
    std::vector<int> c_offs,
    std::vector<int> lon_offs,
    int B, int C_local, int K, int nlat_local, int nlon_out,
    int num_chans, int rank)
{
    Offsets c, lon;
    for(size_t i=0; i<c_offs.size(); i++) c.data[i] = c_offs[i];
    for(size_t i=0; i<lon_offs.size(); i++) lon.data[i] = lon_offs[i];

    int total = B * C_local * K * nlat_local * nlon_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    azimuth_a2a_bwd_push_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uintptr_t*>(y_ptrs_tensor.data_ptr<int64_t>()),
        c, lon, B, C_local, K, nlat_local, nlon_out, num_chans, rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("azimuth_a2a_fwd", &azimuth_a2a_fwd);
    m.def("polar_reduce_scatter", &polar_reduce_scatter);
    m.def("azimuth_a2a_bwd", &azimuth_a2a_bwd);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("disco_spherical_conv_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def get_symm_state(step_name, elements, dtype, pg, device):
    global _symm_cache
    key = (step_name, pg)
    size_bytes = elements * dtype.itemsize
    if key in _symm_cache:
        buf, hdl, ptrs = _symm_cache[key]
        if buf.numel() >= size_bytes:
            return buf.view(dtype)[:elements], ptrs, hdl
            
    buf = symm_mem.empty(size_bytes, dtype=torch.uint8, device=device)
    hdl = symm_mem.rendezvous(buf, pg)
    ptrs = torch.tensor([hdl.buffer_ptrs[i] for i in range(dist.get_world_size(pg))], dtype=torch.int64, device=device)
    
    _symm_cache[key] = (buf, hdl, ptrs)
    return buf.view(dtype)[:elements], ptrs, hdl

def _compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    return [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

@triton.jit
def grouped_mix_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, H, W, G, Cg, K_dim, C_out_g,
    stride_xb, stride_xg, stride_xcg, stride_xk, stride_xh, stride_xw,
    stride_wg, stride_wo, stride_wcg, stride_wk_w,
    stride_yb, stride_yg, stride_yo, stride_yh, stride_yw,
    M, K_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b = offs_m // (H * W)
    hw = offs_m % (H * W)
    h = hw // W
    w = hw % W

    x_base = x_ptr + b[:, None] * stride_xb + g * stride_xg + h[:, None] * stride_xh + w[:, None] * stride_xw
    w_base = w_ptr + g * stride_wg + offs_n[None, :] * stride_wo

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(0, K_in, BLOCK_K):
        offs_k = k_iter + tl.arange(0, BLOCK_K)
        cg = offs_k // K_dim
        k = offs_k % K_dim

        x_ptrs = x_base + cg[None, :] * stride_xcg + k[None, :] * stride_xk
        w_ptrs = w_base + cg[:, None] * stride_wcg + k[:, None] * stride_wk_w

        mask_m = offs_m[:, None] < M
        mask_n = offs_n[None, :] < C_out_g
        mask_k = offs_k < K_in

        x = tl.load(x_ptrs, mask=mask_m & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n, other=0.0)

        acc += tl.dot(x, w)

    if bias_ptr is not None:
        bias_ptrs = bias_ptr + g * C_out_g + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < C_out_g, other=0.0)
        acc += bias[None, :]

    y_ptrs = y_ptr + b[:, None] * stride_yb + g * stride_yg + offs_n[None, :] * stride_yo + h[:, None] * stride_yh + w[:, None] * stride_yw
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < C_out_g))

def grouped_channel_mixing(x, weight, bias, groups):
    B, C, K, H, W = x.shape
    C_out = weight.shape[0]
    Cg = weight.shape[1]
    C_out_g = C_out // groups
    
    out = torch.empty(B, C_out, H, W, device=x.device, dtype=x.dtype)
    M = B * H * W
    K_in = Cg * K
    
    if M == 0 or C_out_g == 0:
        return out
        
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out_g, BLOCK_N), groups)
    
    grouped_mix_kernel[grid](
        x, weight, bias, out,
        B, H, W, groups, Cg, K, C_out_g,
        groups * Cg * K * H * W, Cg * K * H * W, K * H * W, H * W, 1,
        C_out_g * Cg * K, Cg * K, K, 1,
        groups * C_out_g * H * W, C_out_g * H * W, H * W, W, 1,
        M, K_in,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out

@torch.no_grad()
def solution(
    x: torch.Tensor,
    psi: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    nlon_out: int,
    nlon_in: int,
    azimuth_group: Optional[dist.ProcessGroup] = None,
    polar_group: Optional[dist.ProcessGroup] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    azimuth_group = azimuth_group or dist.group.WORLD
    polar_group = polar_group or dist.group.WORLD
    azimuth_size = dist.get_world_size(group=azimuth_group)
    polar_size = dist.get_world_size(group=polar_group)
    azimuth_rank = dist.get_rank(group=azimuth_group)
    polar_rank = dist.get_rank(group=polar_group)

    B = x.shape[0]
    C = x.shape[1]
    nlat_in = x.shape[2]
    
    lon_in_shapes = _compute_split_shapes(nlon_in, azimuth_size)
    lon_offsets = [0] + torch.tensor(lon_in_shapes).cumsum(0).tolist()
    C_splits = _compute_split_shapes(C, azimuth_size)
    c_offsets = [0] + torch.tensor(C_splits).cumsum(0).tolist()
    my_C = C_splits[azimuth_rank]
    
    ext = _get_ext()

    # 1. Device-side Azimuth P2P Fwd Push
    if azimuth_size > 1:
        elements_1 = B * my_C * nlat_in * nlon_in
        y_step1, ptrs_1, hdl_1 = get_symm_state('step1', elements_1, x.dtype, azimuth_group, x.device)
        
        hdl_1.barrier(channel=0)
        ext.azimuth_a2a_fwd(
            x, ptrs_1, c_offsets, lon_offsets,
            B, C, nlat_in, lon_in_shapes[azimuth_rank], nlon_in,
            azimuth_rank
        )
        hdl_1.barrier(channel=0)
        x = y_step1.reshape(B, my_C, nlat_in, nlon_in)

    # 2. Sparse DISCO S2 contraction
    kernel_size, nlat_out, _ = psi.shape
    pscale = nlon_in // nlon_out
    
    x = x.reshape(1, B * my_C, nlat_in, nlon_in).permute(0, 2, 3, 1)
    x = x.expand(kernel_size, -1, -1, -1)
    
    y_disco = torch.empty(
        nlon_out, kernel_size, nlat_out, B * my_C,
        device=x.device, dtype=x.dtype
    )
    for pout in range(nlon_out):
        y_disco[pout] = torch.bmm(psi, x.reshape(kernel_size, nlat_in * nlon_in, -1))
        x = torch.roll(x, -pscale, dims=2)
        
    x = y_disco.permute(3, 1, 2, 0).reshape(B, my_C, kernel_size, nlat_out, nlon_out)

    # 3. Device-side Polar Reduce-Scatter Pull
    lat_out_shapes = _compute_split_shapes(nlat_out, polar_size)
    lat_offsets = [0] + torch.tensor(lat_out_shapes).cumsum(0).tolist()
    my_lat = lat_out_shapes[polar_rank]
    
    if polar_size > 1:
        elements_3 = B * my_C * kernel_size * nlat_out * nlon_out
        x_symm, ptrs_3, hdl_3 = get_symm_state('step3', elements_3, x.dtype, polar_group, x.device)
        
        hdl_3.barrier(channel=0)
        x_symm.copy_(x.reshape(-1))
        hdl_3.barrier(channel=0)
        
        y_step3 = torch.empty(B, my_C, kernel_size, my_lat, nlon_out, dtype=x.dtype, device=x.device)
        ext.polar_reduce_scatter(
            ptrs_3, y_step3, lat_offsets,
            B, my_C, kernel_size, nlat_out, nlon_out,
            polar_rank, polar_size
        )
        x = y_step3

    # 4 & 5. Device-side Azimuth P2P Bwd Push
    lon_out_shapes = _compute_split_shapes(nlon_out, azimuth_size)
    lon_out_offsets = [0] + torch.tensor(lon_out_shapes).cumsum(0).tolist()
    my_lon_out = lon_out_shapes[azimuth_rank]
    
    if azimuth_size > 1:
        elements_5 = B * C * kernel_size * my_lat * my_lon_out
        y_step5, ptrs_5, hdl_5 = get_symm_state('step5', elements_5, x.dtype, azimuth_group, x.device)
        
        hdl_5.barrier(channel=0)
        ext.azimuth_a2a_bwd(
            x, ptrs_5, c_offsets, lon_out_offsets,
            B, my_C, kernel_size, my_lat, nlon_out, C,
            azimuth_rank
        )
        hdl_5.barrier(channel=0)
        x = y_step5.reshape(B, C, kernel_size, my_lat, my_lon_out)

    # 6 & 7. Grouped channel mixing + Bias
    out = grouped_channel_mixing(x, weight, bias, groups)
    
    return out