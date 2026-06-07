"""
Optimized DISCO spherical convolution replacing PyTorch collectives and loops with custom CUDA.

Strategy:
1. **Fused All-to-All Push**: The azimuth all-to-all is implemented as a direct peer-to-peer 
   UVA push kernel, eliminating `torch.split`, `contiguous`, and `torch.cat` overheads.
2. **Fused Shifted SpMM**: The expensive `torch.roll` and loop of `torch.bmm` are folded 
   into a single CSR sparse matrix-vector multiplication kernel.
3. **Lock-Free Reduce-Scatter**: The polar all-reduce + split is replaced by a lock-free 
   UVA push kernel into peer buffers, followed by a local reduction kernel.
4. **BMM Layout Fusion**: The final azimuth scatter pushes data directly into the permuted 
   layout required for the grouped channel-mixing BMM.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void a2a_azimuth_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const long long* __restrict__ dest_ptrs,
    const int* __restrict__ az_global_ranks,
    const int* __restrict__ C_offsets,
    const int* __restrict__ C_splits,
    int B, int C, int nlat_in, int my_lon_in,
    int nlon_in, int my_lon_offset, int az_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * nlat_in * my_lon_in;
    if (idx >= total) return;

    int lon = idx % my_lon_in;
    int tmp = idx / my_lon_in;
    int lat = tmp % nlat_in;
    tmp = tmp / nlat_in;
    int c = tmp % C;
    int b = tmp / C;

    int dest_az_rank = 0;
    for (int i = 0; i < az_size; ++i) {
        if (c >= C_offsets[i] && c < C_offsets[i] + C_splits[i]) {
            dest_az_rank = i;
            break;
        }
    }

    int c_local = c - C_offsets[dest_az_rank];
    int dest_global_rank = az_global_ranks[dest_az_rank];
    __nv_bfloat16* dest_ptr = (__nv_bfloat16*)dest_ptrs[dest_global_rank];

    int dest_C = C_splits[dest_az_rank];
    int dest_idx = ((b * dest_C + c_local) * nlat_in + lat) * nlon_in + (my_lon_offset + lon);
    dest_ptr[dest_idx] = x[idx];
}

__global__ void spmm_shift_csr_kernel(
    const int* __restrict__ crow_indices,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y,
    int R, int N, int nlat_in, int nlon_in, int nlon_out, int pscale, int kernel_size, int nlat_out
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int pout = blockIdx.z;

    if (r >= R || n_idx >= N) return;

    int row_start = crow_indices[r];
    int row_end = crow_indices[r+1];

    float sum = 0.0f;
    for (int nz = row_start; nz < row_end; ++nz) {
        int in_idx = col_indices[nz];
        float val = values[nz];

        int lat_in = in_idx / nlon_in;
        int lon_in = in_idx % nlon_in;
        int lon_shifted = (lon_in + pout * pscale) % nlon_in;

        float x_val = __bfloat162float(x[(n_idx * nlat_in + lat_in) * nlon_in + lon_shifted]);
        sum += val * x_val;
    }

    int k = r / nlat_out;
    int lat_out = r % nlat_out;
    int y_idx = ((n_idx * kernel_size + k) * nlat_out + lat_out) * nlon_out + pout;
    y[y_idx] = sum;
}

__global__ void push_rs_chunk_kernel(
    const float* __restrict__ local_y,
    const long long* __restrict__ dest_ptrs,
    const int* __restrict__ polar_global_ranks,
    const int* __restrict__ nlat_out_offsets,
    const int* __restrict__ nlat_out_splits,
    int N, int kernel_size, int nlat_out, int nlon_out,
    int my_polar_rank, int polar_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * kernel_size * nlat_out * nlon_out;
    if (idx >= total) return;

    int lon = idx % nlon_out;
    int tmp = idx / nlon_out;
    int lat = tmp % nlat_out;
    tmp = tmp / nlat_out;
    int k = tmp % kernel_size;
    int n = tmp / kernel_size;

    int dest_polar_rank = 0;
    for (int i = 0; i < polar_size; ++i) {
        if (lat >= nlat_out_offsets[i] && lat < nlat_out_offsets[i] + nlat_out_splits[i]) {
            dest_polar_rank = i;
            break;
        }
    }

    int dest_global_rank = polar_global_ranks[dest_polar_rank];
    float* dest_ptr = (float*)dest_ptrs[dest_global_rank];

    int lat_local = lat - nlat_out_offsets[dest_polar_rank];
    int dest_nlat = nlat_out_splits[dest_polar_rank];

    int remote_idx = (((my_polar_rank * N + n) * kernel_size + k) * dest_nlat + lat_local) * nlon_out + lon;
    dest_ptr[remote_idx] = local_y[idx];
}

__global__ void reduce_rs_chunk_kernel(
    const float* __restrict__ local_buf,
    __nv_bfloat16* __restrict__ out,
    int polar_size, int chunk_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;

    float sum = 0.0f;
    for (int p = 0; p < polar_size; ++p) {
        sum += local_buf[p * chunk_size + idx];
    }
    out[idx] = __float2bfloat16(sum);
}

__global__ void a2a_azimuth_bwd_kernel(
    const __nv_bfloat16* __restrict__ local_x,
    const long long* __restrict__ dest_ptrs,
    const int* __restrict__ az_global_ranks,
    const int* __restrict__ lon_out_offsets,
    const int* __restrict__ lon_out_splits,
    int B, int C_local, int kernel_size, int nlat_out_local, int nlon_out,
    int my_C_offset, int az_size, int full_C, int groups, int groupsize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_local * kernel_size * nlat_out_local * nlon_out;
    if (idx >= total) return;

    int lon = idx % nlon_out;
    int tmp = idx / nlon_out;
    int lat = tmp % nlat_out_local;
    tmp = tmp / nlat_out_local;
    int k = tmp % kernel_size;
    tmp = tmp / kernel_size;
    int c = tmp % C_local;
    int b = tmp / C_local;

    int dest_az_rank = 0;
    for (int i = 0; i < az_size; ++i) {
        if (lon >= lon_out_offsets[i] && lon < lon_out_offsets[i] + lon_out_splits[i]) {
            dest_az_rank = i;
            break;
        }
    }

    int lon_local = lon - lon_out_offsets[dest_az_rank];
    int dest_global_rank = az_global_ranks[dest_az_rank];
    __nv_bfloat16* dest_ptr = (__nv_bfloat16*)dest_ptrs[dest_global_rank];

    int dest_nlon = lon_out_splits[dest_az_rank];
    int c_global = my_C_offset + c;
    
    int g = c_global / groupsize;
    int c_in_g = c_global % groupsize;
    int dest_HW = nlat_out_local * dest_nlon;
    int hw = lat * dest_nlon + lon_local;

    int dest_idx = (g * (groupsize * kernel_size) + (c_in_g * kernel_size + k)) * (B * dest_HW) + (b * dest_HW + hw);
    dest_ptr[dest_idx] = local_x[idx];
}

void launch_a2a_fwd(
    torch::Tensor x, torch::Tensor dest_ptrs, torch::Tensor az_global_ranks,
    torch::Tensor C_offsets, torch::Tensor C_splits,
    int B, int C, int nlat_in, int my_lon_in, int nlon_in, int my_lon_offset, int az_size
) {
    int total = B * C * nlat_in * my_lon_in;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    a2a_azimuth_fwd_kernel<<<blocks, threads>>>(
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const long long*)dest_ptrs.data_ptr<int64_t>(),
        az_global_ranks.data_ptr<int>(),
        C_offsets.data_ptr<int>(),
        C_splits.data_ptr<int>(),
        B, C, nlat_in, my_lon_in, nlon_in, my_lon_offset, az_size
    );
}

void launch_spmm(
    torch::Tensor crow, torch::Tensor col, torch::Tensor values,
    torch::Tensor x, torch::Tensor y,
    int R, int N, int nlat_in, int nlon_in, int nlon_out, int pscale, int kernel_size, int nlat_out
) {
    if (R == 0 || N == 0) return;
    dim3 threads(128, 1, 1);
    dim3 blocks((R + threads.x - 1) / threads.x, N, nlon_out);
    spmm_shift_csr_kernel<<<blocks, threads>>>(
        crow.data_ptr<int>(),
        col.data_ptr<int>(),
        values.data_ptr<float>(),
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        y.data_ptr<float>(),
        R, N, nlat_in, nlon_in, nlon_out, pscale, kernel_size, nlat_out
    );
}

void launch_push_rs(
    torch::Tensor local_y, torch::Tensor dest_ptrs, torch::Tensor polar_global_ranks,
    torch::Tensor nlat_out_offsets, torch::Tensor nlat_out_splits,
    int N, int kernel_size, int nlat_out, int nlon_out,
    int my_polar_rank, int polar_size
) {
    int total = N * kernel_size * nlat_out * nlon_out;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    push_rs_chunk_kernel<<<blocks, threads>>>(
        local_y.data_ptr<float>(),
        (const long long*)dest_ptrs.data_ptr<int64_t>(),
        polar_global_ranks.data_ptr<int>(),
        nlat_out_offsets.data_ptr<int>(),
        nlat_out_splits.data_ptr<int>(),
        N, kernel_size, nlat_out, nlon_out,
        my_polar_rank, polar_size
    );
}

void launch_reduce_rs(
    torch::Tensor local_buf, torch::Tensor out,
    int polar_size, int chunk_size
) {
    if (chunk_size == 0) return;
    int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;
    reduce_rs_chunk_kernel<<<blocks, threads>>>(
        local_buf.data_ptr<float>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        polar_size, chunk_size
    );
}

void launch_a2a_bwd(
    torch::Tensor local_x, torch::Tensor dest_ptrs, torch::Tensor az_global_ranks,
    torch::Tensor lon_out_offsets, torch::Tensor lon_out_splits,
    int B, int C_local, int kernel_size, int nlat_out_local, int nlon_out,
    int my_C_offset, int az_size, int full_C, int groups, int groupsize
) {
    int total = B * C_local * kernel_size * nlat_out_local * nlon_out;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    a2a_azimuth_bwd_kernel<<<blocks, threads>>>(
        (__nv_bfloat16*)local_x.data_ptr<at::BFloat16>(),
        (const long long*)dest_ptrs.data_ptr<int64_t>(),
        az_global_ranks.data_ptr<int>(),
        lon_out_offsets.data_ptr<int>(),
        lon_out_splits.data_ptr<int>(),
        B, C_local, kernel_size, nlat_out_local, nlon_out,
        my_C_offset, az_size, full_C, groups, groupsize
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_a2a_fwd", &launch_a2a_fwd);
    m.def("launch_spmm", &launch_spmm);
    m.def("launch_push_rs", &launch_push_rs);
    m.def("launch_reduce_rs", &launch_reduce_rs);
    m.def("launch_a2a_bwd", &launch_a2a_bwd);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("disco_s2_fused_ext", CUDA_SRC)
    return _ext

def _compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    return [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]


_symm_cache = {}
def _get_symm_mem(shape, dtype):
    key = (shape, dtype)
    if key not in _symm_cache:
        buf = symm_mem.empty(shape, dtype=dtype, device='cuda')
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device='cuda')
        _symm_cache[key] = (buf, ptrs)
    return _symm_cache[key]

_psi_cache = {}


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
    
    az_size = dist.get_world_size(group=azimuth_group)
    az_rank = dist.get_rank(group=azimuth_group)
    polar_size = dist.get_world_size(group=polar_group)
    polar_rank = dist.get_rank(group=polar_group)

    B, full_C, nlat_in, my_lon_in = x.shape
    kernel_size, nlat_out, _ = psi.shape
    pscale = nlon_in // nlon_out

    # Meta constants & shapes
    C_splits = _compute_split_shapes(full_C, az_size)
    C_offsets = [sum(C_splits[:i]) for i in range(az_size)]
    my_C = C_splits[az_rank]

    lon_in_splits = _compute_split_shapes(nlon_in, az_size)
    lon_in_offsets = [sum(lon_in_splits[:i]) for i in range(az_size)]

    nlat_out_splits = _compute_split_shapes(nlat_out, polar_size)
    nlat_out_offsets = [sum(nlat_out_splits[:i]) for i in range(polar_size)]
    my_nlat_out = nlat_out_splits[polar_rank]

    lon_out_splits = _compute_split_shapes(nlon_out, az_size)
    lon_out_offsets = [sum(lon_out_splits[:i]) for i in range(az_size)]
    my_lon_out = lon_out_splits[az_rank]

    def _to_t(lst): return torch.tensor(lst, dtype=torch.int32, device='cuda')
    
    az_global_ranks = _to_t([dist.get_global_rank(azimuth_group, i) for i in range(az_size)])
    polar_global_ranks = _to_t([dist.get_global_rank(polar_group, i) for i in range(polar_size)])
    
    C_splits_t = _to_t(C_splits)
    C_offsets_t = _to_t(C_offsets)
    lon_out_splits_t = _to_t(lon_out_splits)
    lon_out_offsets_t = _to_t(lon_out_offsets)
    nlat_out_splits_t = _to_t(nlat_out_splits)
    nlat_out_offsets_t = _to_t(nlat_out_offsets)

    # Convert psi to CSR once
    psi_ptr = psi.data_ptr()
    if psi_ptr not in _psi_cache:
        if psi.is_sparse:
            psi_coo = psi.coalesce()
            idx, vals = psi_coo.indices(), psi_coo.values()
        else:
            idx = psi.nonzero(as_tuple=False).t().contiguous()
            vals = psi[idx[0], idx[1], idx[2]]
            
        psi_csr = torch.sparse_coo_tensor(
            torch.stack([idx[0] * nlat_out + idx[1], idx[2]]), vals.float(),
            size=(kernel_size * nlat_out, nlat_in * nlon_in)
        ).coalesce().to_sparse_csr()
        _psi_cache[psi_ptr] = (psi_csr.crow_indices().int(), psi_csr.col_indices().int(), psi_csr.values())
        
    crow, col, vals = _psi_cache[psi_ptr]

    # --- Step 1: Azimuth FWD A2A Push ---
    symm_x_az, ptrs_x_az = _get_symm_mem((B, my_C, nlat_in, nlon_in), torch.bfloat16)
    dist.barrier()
    if az_size > 1:
        _get_ext().launch_a2a_fwd(
            x, ptrs_x_az, az_global_ranks, C_offsets_t, C_splits_t,
            B, full_C, nlat_in, my_lon_in, nlon_in, lon_in_offsets[az_rank], az_size
        )
        dist.barrier()
        curr_x = symm_x_az
    else:
        curr_x = x

    # --- Step 2: Fused SpMM Shift Contraction ---
    y_partial = torch.empty((B * my_C, kernel_size, nlat_out, nlon_out), dtype=torch.float32, device='cuda')
    _get_ext().launch_spmm(
        crow, col, vals, curr_x, y_partial,
        kernel_size * nlat_out, B * my_C, nlat_in, nlon_in, nlon_out, pscale, kernel_size, nlat_out
    )

    # --- Step 3 & 4: Lock-Free Polar Reduce-Scatter ---
    symm_y_polar, ptrs_y_polar = _get_symm_mem((polar_size, B * my_C, kernel_size, my_nlat_out, nlon_out), torch.float32)
    dist.barrier()
    if polar_size > 1:
        _get_ext().launch_push_rs(
            y_partial, ptrs_y_polar, polar_global_ranks, nlat_out_offsets_t, nlat_out_splits_t,
            B * my_C, kernel_size, nlat_out, nlon_out, polar_rank, polar_size
        )
        dist.barrier()
        y_local = torch.empty((B * my_C, kernel_size, my_nlat_out, nlon_out), dtype=torch.bfloat16, device='cuda')
        _get_ext().launch_reduce_rs(symm_y_polar, y_local, polar_size, y_local.numel())
    else:
        y_local = y_partial.to(torch.bfloat16)

    # --- Step 5: Azimuth BWD A2A Push (Directly to BMM layout) ---
    groupsize = full_C // groups
    symm_out_az, ptrs_out_az = _get_symm_mem((groups, groupsize * kernel_size, B * my_nlat_out * my_lon_out), torch.bfloat16)
    dist.barrier()
    
    if az_size > 1:
        _get_ext().launch_a2a_bwd(
            y_local, ptrs_out_az, az_global_ranks, lon_out_offsets_t, lon_out_splits_t,
            B, my_C, kernel_size, my_nlat_out, nlon_out, C_offsets[az_rank], az_size, full_C, groups, groupsize
        )
        dist.barrier()
        bmm_in = symm_out_az
    else:
        # If no azimuth distribution, emulate BMM layout preparation locally
        bmm_in = y_local.view(B, groups, groupsize, kernel_size, my_nlat_out, my_lon_out)
        bmm_in = bmm_in.permute(1, 2, 3, 0, 4, 5).reshape(groups, groupsize * kernel_size, B * my_nlat_out * my_lon_out)

    # --- Step 6 & 7: Grouped Channel Mixing & Bias ---
    weight_reshaped = weight.reshape(groups, -1, weight.shape[1] * weight.shape[2]).to(torch.bfloat16)
    out = torch.bmm(weight_reshaped, bmm_in)
    
    out = out.view(groups, -1, B, my_nlat_out, my_lon_out)
    out = out.permute(2, 0, 1, 3, 4).reshape(B, -1, my_nlat_out, my_lon_out)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1).to(out.dtype)
        
    return out.contiguous()