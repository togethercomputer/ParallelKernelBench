"""
Hyena CP forward with symmetric-memory all-to-all replacing NCCL.
"""

from typing import Optional

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

// Pack local input [B, D_global, L_local] into peer buffers.
// For destination rank r, write the channel-slab x[:, r*Dl:(r+1)*Dl, :]
// into peer r's symmetric buffer at offset for our rank's slot.
// Optionally apply inverse-zigzag along sequence axis when assembling at full layout.
//
// Layout in peer buf (per peer): [world_size, B, Dl, L_local]
// where the first dim indexes the source rank.
//
// Args:
//   x:          [B, D_global, L_local]  (this rank's input, contiguous)
//   peer_ptrs:  int64 array length world_size  (BF16* device pointers)
//   B, Dg, Ll, world_size, my_rank
//
// Each thread handles one BF16 element. Block tiles over (b, dl_chunk, ll_chunk).

extern "C" {

__global__ void pack_split_to_full_kernel(
    const __nv_bfloat16* __restrict__ x,   // [B, Dg, Ll]
    const long long* __restrict__ peer_ptrs,
    int B, int Dg, int Ll, int world_size, int my_rank
) {
    int Dl = Dg / world_size;
    long long total_per_dest = (long long)B * Dl * Ll;
    long long total = total_per_dest * world_size;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long idx = tid; idx < total; idx += stride) {
        long long dest = idx / total_per_dest;
        long long rem = idx % total_per_dest;
        long long b = rem / ((long long)Dl * Ll);
        long long r2 = rem % ((long long)Dl * Ll);
        long long dl = r2 / Ll;
        long long ll = r2 % Ll;

        // Source: x[b, dest*Dl + dl, ll]
        long long src_off = b * (long long)Dg * Ll + (dest * Dl + dl) * Ll + ll;
        __nv_bfloat16 val = x[src_off];

        // Destination: peer_ptrs[dest] + slot for (my_rank, b, dl, ll)
        // peer buffer layout: [world_size_src, B, Dl, Ll]
        __nv_bfloat16* dst_base = reinterpret_cast<__nv_bfloat16*>(peer_ptrs[dest]);
        long long dst_off = ((long long)my_rank * B + b) * Dl * Ll + dl * Ll + ll;
        dst_base[dst_off] = val;
    }
}

// After barrier, gather from local symm buf [world_size, B, Dl, Ll]
// into full tensor [B, Dl, L_full] with optional inverse-zigzag.
// L_full = world_size * Ll
//
// Without zigzag: out[b, dl, src*Ll + ll] = buf[src, b, dl, ll]
// With zigzag (num_chunks = 2*world_size, chunk_size = L_full / num_chunks = Ll/2):
//   inverse zigzag indices map output chunk c -> source chunk perm_inv[c]
//   We compute: for each output position s in [0, L_full), find chunk c = s / chunk_size,
//   src_chunk = perm_inv[c], src_pos = src_chunk * chunk_size + (s % chunk_size)
//   then map src_pos -> (src_rank = src_pos / Ll, ll = src_pos % Ll)

__global__ void unpack_full_kernel(
    const __nv_bfloat16* __restrict__ buf,  // [world_size, B, Dl, Ll]
    __nv_bfloat16* __restrict__ out,        // [B, Dl, L_full]
    const int* __restrict__ inv_zigzag,     // [num_chunks] or nullptr
    int B, int Dl, int Ll, int world_size, int chunk_size, int num_chunks,
    int use_zigzag
) {
    long long L_full = (long long)world_size * Ll;
    long long total = (long long)B * Dl * L_full;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long idx = tid; idx < total; idx += stride) {
        long long b = idx / ((long long)Dl * L_full);
        long long rem = idx % ((long long)Dl * L_full);
        long long dl = rem / L_full;
        long long s = rem % L_full;

        long long src_pos;
        if (use_zigzag) {
            int c = (int)(s / chunk_size);
            int off = (int)(s % chunk_size);
            int sc = inv_zigzag[c];
            src_pos = (long long)sc * chunk_size + off;
        } else {
            src_pos = s;
        }
        long long src_rank = src_pos / Ll;
        long long ll = src_pos % Ll;

        long long src_off = ((src_rank * B) + b) * Dl * Ll + dl * Ll + ll;
        out[idx] = buf[src_off];
    }
}

// Pack full tensor [B, Dl, L_full] into peer buffers for full->split,
// with optional zigzag re-ordering along sequence dim.
// Destination peer r receives the slab corresponding to its sequence shard.
// peer buffer layout: [world_size_src, B, Dl, Ll]
//
// Without zigzag: peer r gets x[:, :, r*Ll:(r+1)*Ll]
// With zigzag: forward zigzag indices map dest chunk -> source chunk
//   For each output position in dest's local layout (dest_seq = r*Ll + ll, but
//   sequence is in zigzag order, so dest sequence index = r*Ll + ll, and
//   actual src position = chunk_table).
// Simpler: precompute a [L_full] mapping dst_seq_idx -> src_seq_idx.
//   For position s in full output (zigzagged), src_pos = zigzag[s/chunk]*chunk + s%chunk

__global__ void pack_full_to_split_kernel(
    const __nv_bfloat16* __restrict__ x,   // [B, Dl, L_full]
    const long long* __restrict__ peer_ptrs,
    const int* __restrict__ fwd_zigzag,    // [num_chunks] or nullptr
    int B, int Dl, int Ll, int world_size,
    int chunk_size, int num_chunks, int use_zigzag, int my_rank
) {
    long long L_full = (long long)world_size * Ll;
    long long total = (long long)B * Dl * L_full;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long idx = tid; idx < total; idx += stride) {
        long long b = idx / ((long long)Dl * L_full);
        long long rem = idx % ((long long)Dl * L_full);
        long long dl = rem / L_full;
        long long s = rem % L_full;  // index in zigzag-output space

        long long src_pos;
        if (use_zigzag) {
            int c = (int)(s / chunk_size);
            int off = (int)(s % chunk_size);
            int sc = fwd_zigzag[c];
            src_pos = (long long)sc * chunk_size + off;
        } else {
            src_pos = s;
        }

        // After (optional) zigzag, the value at logical position s belongs to
        // dest_rank = s / Ll, ll = s % Ll
        long long dest_rank = s / Ll;
        long long ll = s % Ll;

        // Source: x[b, dl, src_pos]
        long long src_off = b * (long long)Dl * L_full + dl * L_full + src_pos;
        __nv_bfloat16 val = x[src_off];

        __nv_bfloat16* dst_base = reinterpret_cast<__nv_bfloat16*>(peer_ptrs[dest_rank]);
        // peer buffer layout for full->split recv: [world_size_src, B, Dl, Ll]
        long long dst_off = ((long long)my_rank * B + b) * Dl * Ll + dl * Ll + ll;
        dst_base[dst_off] = val;
    }
}

// Final unpack: from local symm buf [world_size, B, Dl, Ll]
// produce out [B, world_size*Dl, Ll] (channels gathered)
// out[b, src*Dl + dl, ll] = buf[src, b, dl, ll]

__global__ void unpack_split_kernel(
    const __nv_bfloat16* __restrict__ buf,  // [world_size, B, Dl, Ll]
    __nv_bfloat16* __restrict__ out,        // [B, world_size*Dl, Ll]
    int B, int Dl, int Ll, int world_size
) {
    long long Dg = (long long)world_size * Dl;
    long long total = (long long)B * Dg * Ll;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long idx = tid; idx < total; idx += stride) {
        long long b = idx / (Dg * Ll);
        long long rem = idx % (Dg * Ll);
        long long d = rem / Ll;
        long long ll = rem % Ll;
        long long src = d / Dl;
        long long dl = d % Dl;
        long long src_off = ((src * B) + b) * Dl * Ll + dl * Ll + ll;
        out[idx] = buf[src_off];
    }
}

// Fused elementwise: z = x2 * v
__global__ void mul_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    long long n
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < n; i += stride) {
        float av = __bfloat162float(a[i]);
        float bv = __bfloat162float(b[i]);
        out[i] = __float2bfloat16(av * bv);
    }
}

// Fused: z = x1 * z (already containing fftconv result)
__global__ void mul_inplace_kernel(
    const __nv_bfloat16* __restrict__ x1,
    __nv_bfloat16* __restrict__ z,
    long long n
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < n; i += stride) {
        float x1v = __bfloat162float(x1[i]);
        float zv = __bfloat162float(z[i]);
        z[i] = __float2bfloat16(x1v * zv);
    }
}

// Transpose [B, D, L] -> [B, L, D] for final output (BF16)
__global__ void transpose_bld_kernel(
    const __nv_bfloat16* __restrict__ in,   // [B, D, L]
    __nv_bfloat16* __restrict__ out,        // [B, L, D]
    int B, int D, int L
) {
    long long total = (long long)B * D * L;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = tid; idx < total; idx += stride) {
        long long b = idx / ((long long)D * L);
        long long rem = idx % ((long long)D * L);
        long long d = rem / L;
        long long l = rem % L;
        long long out_off = b * (long long)L * D + l * D + d;
        out[out_off] = in[idx];
    }
}

}  // extern "C"

void launch_pack_split_to_full(
    torch::Tensor x, torch::Tensor peer_ptrs,
    int64_t B, int64_t Dg, int64_t Ll, int64_t world_size, int64_t my_rank
) {
    long long n = (long long)B * Dg * Ll;
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_split_to_full_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        (int)B, (int)Dg, (int)Ll, (int)world_size, (int)my_rank);
}

void launch_unpack_full(
    torch::Tensor buf, torch::Tensor out, torch::Tensor inv_zigzag,
    int64_t B, int64_t Dl, int64_t Ll, int64_t world_size,
    int64_t chunk_size, int64_t num_chunks, int64_t use_zigzag
) {
    long long n = (long long)B * Dl * world_size * Ll;
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int* idxp = use_zigzag ? inv_zigzag.data_ptr<int>() : nullptr;
    unpack_full_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)buf.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        idxp,
        (int)B, (int)Dl, (int)Ll, (int)world_size,
        (int)chunk_size, (int)num_chunks, (int)use_zigzag);
}

void launch_pack_full_to_split(
    torch::Tensor x, torch::Tensor peer_ptrs, torch::Tensor fwd_zigzag,
    int64_t B, int64_t Dl, int64_t Ll, int64_t world_size,
    int64_t chunk_size, int64_t num_chunks, int64_t use_zigzag, int64_t my_rank
) {
    long long n = (long long)B * Dl * world_size * Ll;
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int* idxp = use_zigzag ? fwd_zigzag.data_ptr<int>() : nullptr;
    pack_full_to_split_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        idxp,
        (int)B, (int)Dl, (int)Ll, (int)world_size,
        (int)chunk_size, (int)num_chunks, (int)use_zigzag, (int)my_rank);
}

void launch_unpack_split(
    torch::Tensor buf, torch::Tensor out,
    int64_t B, int64_t Dl, int64_t Ll, int64_t world_size
) {
    long long n = (long long)B * world_size * Dl * Ll;
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    unpack_split_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)buf.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (int)B, (int)Dl, (int)Ll, (int)world_size);
}

void launch_mul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    long long n = a.numel();
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    mul_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)a.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)b.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), n);
}

void launch_mul_inplace(torch::Tensor x1, torch::Tensor z) {
    long long n = z.numel();
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    mul_inplace_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x1.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)z.data_ptr<at::BFloat16>(), n);
}

void launch_transpose_bld(torch::Tensor in_, torch::Tensor out,
                          int64_t B, int64_t D, int64_t L) {
    long long n = (long long)B * D * L;
    int threads = 256;
    int blocks = std::min<long long>((n + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    transpose_bld_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)in_.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (int)B, (int)D, (int)L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_split_to_full", &launch_pack_split_to_full);
    m.def("unpack_full", &launch_unpack_full);
    m.def("pack_full_to_split", &launch_pack_full_to_split);
    m.def("unpack_split", &launch_unpack_split);
    m.def("mul", &launch_mul);
    m.def("mul_inplace", &launch_mul_inplace);
    m.def("transpose_bld", &launch_transpose_bld);
}
'''


_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("hyena_cp_ext", CUDA_SRC)
    return _ext


# ---------------- Symmetric memory caches ----------------

_symm_cache = {}


def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, group, tag: str):
    key = (tag, numel, dtype, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, peer_ptrs)
    return _symm_cache[key]


_index_cache = {}


def _get_zigzag_indices(num_chunks: int, device: torch.device):
    key = (num_chunks, device.index)
    if key in _index_cache:
        return _index_cache[key]
    half_f = (num_chunks + 1) // 2
    left = torch.arange(half_f, device=device)
    right = torch.arange(num_chunks - 1, half_f - 1, -1, device=device)
    fwd = torch.empty(num_chunks, dtype=torch.long, device=device)
    fwd[0::2] = left
    fwd[1::2] = right

    half_i = num_chunks // 2
    left_i = torch.arange(half_i, device=device)
    right_i = torch.arange(num_chunks - 1, half_i - 1, -1, device=device)
    inv_src = torch.empty(num_chunks, dtype=torch.long, device=device)
    inv_src[0::2] = left_i
    inv_src[1::2] = right_i
    inv = torch.argsort(inv_src)

    fwd_i = fwd.to(torch.int32).contiguous()
    inv_i = inv.to(torch.int32).contiguous()
    _index_cache[key] = (fwd_i, inv_i)
    return _index_cache[key]


# ---------------- FFT conv (kept on PyTorch, leveraging cuFFT) ----------------

def _fftconv(u: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    seq_len = u.shape[-1]
    fft_size = 2 * seq_len
    u_float = u.float()
    kernel_float = kernel.float()
    kernel_f = torch.fft.rfft(kernel_float, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u_float, n=fft_size)
    y = torch.fft.irfft(u_f * kernel_f.unsqueeze(0), n=fft_size, norm="forward")[..., :seq_len]
    y = y + u_float * bias.float().unsqueeze(-1)
    return y.to(dtype=u.dtype)


# ---------------- Symmetric-memory all-to-all primitives ----------------

def _a2a_split_to_full_symm(x: torch.Tensor, group, hdl, peer_ptrs, with_zigzag: bool):
    """
    x: [B, Dg, Ll] BF16 -> returns [B, Dl, L_full] BF16
    Uses symm_mem buffer of size [world, B, Dl, Ll] sitting on local rank.
    """
    ext = _get_ext()
    world = hdl.world_size
    rank = hdl.rank
    B, Dg, Ll = x.shape
    Dl = Dg // world
    L_full = world * Ll

    # The symm buf is allocated with shape numel = world*B*Dl*Ll, viewed as bf16.
    buf_flat, _, _ = _get_symm_buf(world * B * Dl * Ll, torch.bfloat16, x.device, group,
                                   tag=f"s2f_{B}_{Dg}_{Ll}")
    # peer pointers for THIS buffer
    _, _, peer_ptrs_local = _get_symm_buf(world * B * Dl * Ll, torch.bfloat16, x.device, group,
                                          tag=f"s2f_{B}_{Dg}_{Ll}")

    # Pre-barrier: ensure all ranks ready
    hdl_buf = _symm_cache[(f"s2f_{B}_{Dg}_{Ll}", world * B * Dl * Ll, torch.bfloat16, x.device.index)][1]
    hdl_buf.barrier(channel=0)

    # Direct peer writes
    ext.pack_split_to_full(x.contiguous(), peer_ptrs_local, B, Dg, Ll, world, rank)

    # Barrier after writes complete
    hdl_buf.barrier(channel=1)

    out = torch.empty((B, Dl, L_full), dtype=torch.bfloat16, device=x.device)
    if with_zigzag:
        num_chunks = 2 * world
        chunk_size = L_full // num_chunks
        _, inv_idx = _get_zigzag_indices(num_chunks, x.device)
        ext.unpack_full(buf_flat, out, inv_idx, B, Dl, Ll, world, chunk_size, num_chunks, 1)
    else:
        dummy = torch.empty(1, dtype=torch.int32, device=x.device)
        ext.unpack_full(buf_flat, out, dummy, B, Dl, Ll, world, 0, 0, 0)
    return out


def _a2a_full_to_split_symm(x: torch.Tensor, group, with_zigzag: bool):
    """
    x: [B, Dl, L_full] BF16 -> returns [B, world*Dl, Ll] BF16
    """
    ext = _get_ext()
    world = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    B, Dl, L_full = x.shape
    Ll = L_full // world

    buf_flat, hdl_buf, peer_ptrs_local = _get_symm_buf(
        world * B * Dl * Ll, torch.bfloat16, x.device, group,
        tag=f"f2s_{B}_{Dl}_{Ll}")

    hdl_buf.barrier(channel=2)

    if with_zigzag:
        num_chunks = 2 * world
        chunk_size = L_full // num_chunks
        fwd_idx, _ = _get_zigzag_indices(num_chunks, x.device)
        ext.pack_full_to_split(x.contiguous(), peer_ptrs_local, fwd_idx,
                               B, Dl, Ll, world, chunk_size, num_chunks, 1, rank)
    else:
        dummy = torch.empty(1, dtype=torch.int32, device=x.device)
        ext.pack_full_to_split(x.contiguous(), peer_ptrs_local, dummy,
                               B, Dl, Ll, world, 0, 0, 0, rank)

    hdl_buf.barrier(channel=3)

    out = torch.empty((B, world * Dl, Ll), dtype=torch.bfloat16, device=x.device)
    ext.unpack_split(buf_flat, out, B, Dl, Ll, world)
    return out


@torch.no_grad()
def solution(
    x1_seq: torch.Tensor,
    x2_seq: torch.Tensor,
    v_seq: torch.Tensor,
    h: torch.Tensor,
    conv_bias: torch.Tensor,
    num_groups: int,
    group_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    with_zigzag_splitting: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    # Compile on rank 0 first, then everyone
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    ext = _get_ext()

    # Ensure BF16 path
    assert x1_seq.dtype == torch.bfloat16

    x1 = _a2a_split_to_full_symm(x1_seq, group, None, None, with_zigzag_splitting)
    x2 = _a2a_split_to_full_symm(x2_seq, group, None, None, with_zigzag_splitting)
    v  = _a2a_split_to_full_symm(v_seq,  group, None, None, with_zigzag_splitting)

    local_channels = x1.shape[1]
    local_groups = num_groups // world_size
    h_local = h[rank * local_groups : (rank + 1) * local_groups]
    h_local = h_local.repeat_interleave(group_dim, dim=0)
    bias_local = conv_bias[rank * local_channels : (rank + 1) * local_channels]

    # Fused x2 * v
    z = torch.empty_like(x2)
    ext.mul(x2, v, z)

    # FFT conv (cuFFT-backed)
    z = _fftconv(z, h_local, bias_local)

    # Fused x1 * z (in-place on z)
    ext.mul_inplace(x1, z)

    # All-to-all back to seq-sharded
    z_full = _a2a_full_to_split_symm(z, group, with_zigzag_splitting)

    # Transpose [B, D, l] -> [B, l, D]
    B, D, L = z_full.shape
    out = torch.empty((B, L, D), dtype=z_full.dtype, device=z_full.device)
    ext.transpose_bld(z_full, out, B, D, L)
    return out