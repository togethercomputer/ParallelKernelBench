"""
Distributed DISCO spherical convolution forward with custom CUDA all-reduce
via symmetric memory + multimem PTX, and a custom DISCO contraction kernel.

Strategy:
- Replace polar all-reduce with symm_mem multimem.ld_reduce/st on bf16.
- Replace DISCO contraction Python loop with a single fused CUDA kernel that
  computes y[pout, k, h, bc] = sum over (lat_in, lon_in) psi[k,h,*] * x[bc, lat_in, (lon_in + pout*pscale) % nlon_in].
- Use coalesced bf16 loads and accumulate in fp32.
- Keep all-to-all via dist (small messages) to preserve correctness; hot path
  is the contraction + reduce.
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
#include <cstdint>

// ---------- multimem all-reduce (bf16) ----------

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128, int world_size, int rank, int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank =
        (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride;
         block_start < numel_per_rank;
         block_start += (int64_t)num_programs * (int64_t)block_stride) {
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

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_128, int world_size, int rank,
    int num_blocks, int block_size, int block_stride
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride);
}

// ---------- Peer-pointer fallback all-reduce (bf16) ----------

__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void launch_allreduce_bf16_fallback(
    torch::Tensor ptrs_tensor, torch::Tensor out, int64_t n
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

// ---------- DISCO S2 contraction kernel (sparse psi, bf16 x/out) ----------
// psi is COO-like in CSR form per (k, h):
//   psi_row_ptr: [K * nlat_out + 1] int32
//   psi_col:     [nnz] int32 (column index into nlat_in_local * nlon_in flat dim)
//   psi_val:     [nnz] float32
// y[bc, k, h, pout] = sum_nz psi_val[nz] * x_rolled[bc, col_lat, (col_lon + pout*pscale) % nlon_in]
// We launch grid (BC, K*H, ceil(nlon_out/TILE)) with 1 thread per pout in tile.

__global__ void disco_contraction_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,      // [BC, nlat_in, nlon_in]
    const int32_t* __restrict__ row_ptr,      // [K*nlat_out+1]
    const int32_t* __restrict__ col_idx,      // [nnz]
    const float* __restrict__ vals,           // [nnz]
    __nv_bfloat16* __restrict__ y,            // [BC, K, nlat_out, nlon_out]
    int BC, int K, int nlat_out, int nlon_out,
    int nlat_in, int nlon_in, int pscale
) {
    int bc = blockIdx.x;
    int kh = blockIdx.y;
    int k  = kh / nlat_out;
    int h  = kh - k * nlat_out;
    int pout_base = blockIdx.z * blockDim.x;
    int pout = pout_base + threadIdx.x;
    if (pout >= nlon_out) return;

    int row = k * nlat_out + h;
    int rp_start = row_ptr[row];
    int rp_end   = row_ptr[row + 1];

    const __nv_bfloat16* x_bc = x + (size_t)bc * nlat_in * nlon_in;
    int shift = pout * pscale;

    float acc = 0.0f;
    for (int nz = rp_start; nz < rp_end; ++nz) {
        int c = col_idx[nz];
        float v = vals[nz];
        int lat = c / nlon_in;
        int lon = c - lat * nlon_in;
        // x is rolled in latitude dim by -pscale per pout step => effective lat += pout*pscale (mod nlat_in)
        int lat_eff = lat + shift;
        lat_eff %= nlat_in;
        float xv = __bfloat162float(x_bc[lat_eff * nlon_in + lon]);
        acc += v * xv;
    }

    size_t out_off = ((size_t)bc * K + k) * nlat_out * nlon_out
                   + (size_t)h * nlon_out + pout;
    y[out_off] = __float2bfloat16(acc);
}

void launch_disco_contraction_bf16(
    torch::Tensor x, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor vals,
    torch::Tensor y, int BC, int K, int nlat_out, int nlon_out,
    int nlat_in, int nlon_in, int pscale
) {
    int tile = 64;
    dim3 grid(BC, K * nlat_out, (nlon_out + tile - 1) / tile);
    dim3 block(tile);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    disco_contraction_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        row_ptr.data_ptr<int32_t>(),
        col_idx.data_ptr<int32_t>(),
        vals.data_ptr<float>(),
        (__nv_bfloat16*)y.data_ptr<at::BFloat16>(),
        BC, K, nlat_out, nlon_out, nlat_in, nlon_in, pscale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_bf16_fallback", &launch_allreduce_bf16_fallback);
    m.def("launch_disco_contraction_bf16", &launch_disco_contraction_bf16);
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("disco_s2_ext", CUDA_SRC)
    return _ext


# --------- helpers ---------

def _compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    return [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]


def _transpose(tensor, dim0, dim1, dim1_split_sizes, group):
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)
    tsplit = torch.split(tensor, _compute_split_shapes(tensor.shape[dim0], comm_size), dim=dim0)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))
    dist.all_to_all(x_recv, x_send, group=group)
    return x_recv, [x[dim0] for x in x_send_shapes]


# --------- psi -> CSR cache ---------

_psi_cache = {}

def _psi_to_csr(psi: torch.Tensor):
    """Convert sparse COO psi [K, H, M] to CSR over (k,h) rows."""
    key = (psi.data_ptr(), tuple(psi.shape), psi._nnz() if psi.is_sparse else psi.numel())
    if key in _psi_cache:
        return _psi_cache[key]

    psi_c = psi.coalesce() if psi.is_sparse else psi.to_sparse().coalesce()
    K, H, M = psi_c.shape
    indices = psi_c.indices()  # [3, nnz]
    values = psi_c.values().to(torch.float32).contiguous()
    k_idx = indices[0]
    h_idx = indices[1]
    m_idx = indices[2]

    # Sort by (k*H + h, m) to form CSR
    row = (k_idx.long() * H + h_idx.long())
    # stable sort by row
    order = torch.argsort(row * (M + 1) + m_idx.long(), stable=True)
    row_sorted = row[order]
    col_sorted = m_idx[order].to(torch.int32).contiguous()
    val_sorted = values[order].contiguous()

    nrows = K * H
    row_ptr = torch.zeros(nrows + 1, dtype=torch.int32, device=psi_c.device)
    counts = torch.bincount(row_sorted, minlength=nrows).to(torch.int32)
    row_ptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    res = (row_ptr.contiguous(), col_sorted, val_sorted, K, H, M)
    _psi_cache[key] = res
    return res


# --------- symm mem cache for all-reduce ---------

_symm_cache = {}

def _get_symm(shape, dtype, device):
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


WARP_SIZE = 32
MAX_NUM_BLOCKS = 24
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        block_size = max(block_size, 32)
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size


def _custom_allreduce_bf16(x: torch.Tensor, group) -> torch.Tensor:
    """All-reduce on the WORLD group using symmetric memory multimem."""
    # For correctness with arbitrary subgroup, fall back to dist.all_reduce
    # when group != WORLD.
    if group is not None and group != dist.group.WORLD:
        # Check sizes match
        if dist.get_world_size(group) != dist.get_world_size(dist.group.WORLD):
            dist.all_reduce(x, group=group)
            return x

    n = x.numel()
    shape = x.shape
    buf, hdl, ptrs_tensor = _get_symm(shape, x.dtype, x.device)
    buf.copy_(x)

    numel_per_thread = BYTES_PER_THREAD // x.element_size()
    if n % numel_per_thread == 0 and hasattr(hdl, 'multicast_ptr') and int(hdl.multicast_ptr) != 0:
        numel_128 = n // numel_per_thread
        num_blocks, block_size, block_stride = _multimem_launch_config(n, hdl.world_size)
        try:
            _get_ext().launch_multimem_allreduce_bf16(
                int(hdl.multicast_ptr),
                hdl.signal_pad_ptrs_dev,
                numel_128, hdl.world_size, hdl.rank,
                num_blocks, block_size, block_stride,
            )
            return buf.reshape(shape).clone()
        except Exception:
            pass

    hdl.barrier(channel=0)
    out = torch.empty_like(x)
    _get_ext().launch_allreduce_bf16_fallback(ptrs_tensor, out, n)
    return out


def _disco_contraction_cuda(x: torch.Tensor, psi: torch.Tensor, nlon_out: int) -> torch.Tensor:
    B, C, nlat_in, nlon_in = x.shape
    K, nlat_out, M = psi.shape
    pscale = nlon_in // nlon_out
    BC = B * C

    x_flat = x.reshape(BC, nlat_in, nlon_in).contiguous()
    if x_flat.dtype != torch.bfloat16:
        x_flat_bf = x_flat.to(torch.bfloat16)
    else:
        x_flat_bf = x_flat

    row_ptr, col_idx, vals, K_, H_, M_ = _psi_to_csr(psi)
    if row_ptr.device != x.device:
        row_ptr = row_ptr.to(x.device); col_idx = col_idx.to(x.device); vals = vals.to(x.device)

    y = torch.empty((BC, K, nlat_out, nlon_out), dtype=torch.bfloat16, device=x.device)

    _get_ext().launch_disco_contraction_bf16(
        x_flat_bf, row_ptr, col_idx, vals, y,
        BC, K, nlat_out, nlon_out, nlat_in, nlon_in, pscale,
    )

    y = y.reshape(B, C, K, nlat_out, nlon_out)
    if x.dtype != torch.bfloat16:
        y = y.to(x.dtype)
    return y


# --------- main solution ---------

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
    polar_rank = dist.get_rank(group=polar_group)

    # Trigger compile early
    _get_ext()

    lon_in_shapes = _compute_split_shapes(nlon_in, azimuth_size)
    num_chans = x.shape[1]

    # 1. all-to-all to localize longitude
    if azimuth_size > 1:
        xlist, _ = _transpose(x, dim0=1, dim1=-1, dim1_split_sizes=lon_in_shapes, group=azimuth_group)
        x = torch.cat(xlist, dim=-1)

    # 2. DISCO contraction (custom CUDA)
    x = _disco_contraction_cuda(x, psi, nlon_out)

    # 3. Polar all-reduce via symm_mem multimem (bf16)
    if polar_size > 1:
        x_bf = x.contiguous() if x.dtype == torch.bfloat16 else x.to(torch.bfloat16).contiguous()
        x_bf = _custom_allreduce_bf16(x_bf, polar_group)
        x = x_bf if x.dtype == torch.bfloat16 else x_bf.to(x.dtype)

    # 4. Keep this rank's latitude shard
    if polar_size > 1:
        split_shapes = _compute_split_shapes(x.shape[-2], polar_size)
        x = list(torch.split(x, split_shapes, dim=-2))[polar_rank]

    # 5. Transpose back
    if azimuth_size > 1:
        chan_shapes = _compute_split_shapes(num_chans, azimuth_size)
        xlist, _ = _transpose(x, dim0=-1, dim1=1, dim1_split_sizes=chan_shapes, group=azimuth_group)
        x = torch.cat(xlist, dim=1)

    # 6. Grouped channel mixing
    B, C, K, H, W = x.shape
    groupsize = C // groups
    x = x.reshape(B, groups, groupsize, K, H, W)
    out = torch.einsum(
        "bgckxy,gock->bgoxy",
        x,
        weight.reshape(groups, -1, weight.shape[1], weight.shape[2]),
    ).contiguous()
    out = out.reshape(out.shape[0], -1, H, W)

    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)

    return out