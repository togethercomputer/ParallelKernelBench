from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused all-to-all + transpose for QKV gather_seq_scatter_heads.
//
// Logical view per rank input (after view): [B, S_local, 3, H_total, D]
// where qkv_tensor is [..., 3*H_total*D] reinterpreted; here we flatten
// leading dims into B (product of all dims before seq_dim) and middle dims
// (between seq_dim and last) into M, so input is [B, S_local, M, 3, H_total*D]
// Actually we keep it simpler: flatten as [outer, S_local, inner_per_seq],
// where inner_per_seq = product(shape[seq_dim+1:-1]) * 3 * H_total * D.
//
// After all_to_all (scatter heads, gather seq):
// Each rank holds heads [H_total/W] but full seq S_total = S_local * W.
// Output logical: [outer, S_total, inner_mid, 3, H_local, D]
// where H_local = H_total / W.
//
// For correctness wrt original semantics: original code does
//   bef = view([..., 3, qkv_proj_dim/3])  -- last dim split into (3, H*D)
//   _SeqAllToAll(scatter_dim=ndim (the new "3" axis position? actually
//     scatter_dim = qkv_tensor.dim() before the view, which is original ndim,
//     so after view that is dim index = original_ndim, which is the "3" axis)
// Hmm — scatter_dim is set to qkv_tensor.dim() (original dim count), and after
// view the tensor has ndim+1 dims, so scatter_dim points at the "3" axis.
// gather_dim = seq_dim.
//
// All_to_all_single with scatter_dim != 0 and gather_dim != 0 takes path with
// scatter_dim<=1 only when both are <=1; else falls to _all_to_all (tensor_split).
// With scatter_dim = ndim (likely > 1), it goes through _all_to_all path:
//   split along scatter_dim into W chunks, all_to_all, cat along gather_dim.
// So scatter axis is the "3" axis, which has size 3 — that's wrong unless
// world_size divides 3. Let me re-read.
//
// Actually: orig last dim qkv_proj_dim. View reshapes last dim into (3, qkv_proj_dim/3).
// scatter_dim = qkv_tensor.dim() — that is the ORIGINAL ndim, BEFORE view.
// The view increases ndim by 1. So if original ndim = N, new tensor has ndim N+1,
// and scatter_dim = N points to the second-to-last axis = the "3" axis... wait,
// no: indices 0..N. New axes: 0..N-1 are original 0..N-2, then N-1 is "3", N is
// "qkv_proj_dim/3" = H*D. So scatter_dim=N points at the LAST axis (H*D).
// Wait, original ndim = N means dims 0..N-1. After view (split last), ndim = N+1,
// dims 0..N. Scatter_dim = N (== original ndim). So scatter_dim = N is the last
// axis of new tensor = H*D axis. Good — that's H*D dimension being scattered.
//
// So algorithm: split H*D into W parts, all_to_all (each peer gets H*D/W slice),
// concat along seq_dim. Result has S*W along seq, H*D/W along last.
//
// Restore_shape view: out_shape = orig_shape with [seq_dim]*=W and [-1]/=W.
// So final output: [..., S*W, ..., 3*H*D/W] (last dim still includes the 3).

// Kernel: input_local has shape [outer, S_local, mid, 3, HD] flattened.
// We treat layout as [outer * S_local * mid, 3 * HD] effectively, but the
// scatter axis is HD (last axis). After view orig_shape -> new with extra dim,
// we have [..., S_local, ..., 3, HD]. Then final restore concatenates along
// seq_dim with size S_local*W and last dim HD/W.
//
// Per rank source tensor layout (contiguous): outer × S_local × mid × 3 × HD
// where outer = product(shape[0..seq_dim-1]),
//       mid = product(shape[seq_dim+1..ndim-2])  (between seq and last),
//       HD  = qkv_proj_dim / 3,  (3 split out)
// Total elements = outer * S_local * mid * 3 * HD.
//
// Per rank dest tensor layout: outer × (S_local*W) × mid × 3 × (HD/W).
// Mapping: for output index (o, s_global, m, q, hd_local):
//   peer_rank = s_global / S_local  (which peer's data along seq)
//   s_local   = s_global % S_local
//   hd_global = rank * (HD/W) + hd_local   -- this rank holds slice [rank*HD/W..(rank+1)*HD/W)
//   src element on peer 'peer_rank' at index (o, s_local, m, q, hd_global).

extern "C" __global__ void fused_a2a_qkv_kernel_bf16(
    const long long* __restrict__ peer_ptrs,  // [W] device pointers (uintptr) of each rank's input
    __nv_bfloat16* __restrict__ output,
    int world_size,
    int rank,
    long long outer,
    long long S_local,
    long long mid,
    long long HD,        // total HD = H_total * D
    long long HD_local   // HD / W
) {
    long long S_total = S_local * world_size;
    // Output total elements
    long long total = outer * S_total * mid * 3 * HD_local;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    // Vectorize: process 8 bf16 (16 bytes) when HD_local is divisible by 8.
    // We'll do scalar fallback if not aligned.
    bool vec_ok = (HD_local % 8 == 0);

    if (vec_ok) {
        long long total_vec = total / 8;
        for (long long v = tid; v < total_vec; v += stride) {
            long long e = v * 8;
            // decode e -> (o, s_global, m, q, hd_local)
            long long hd_local = e % HD_local;
            long long t = e / HD_local;
            long long q = t % 3;
            t = t / 3;
            long long m = t % mid;
            t = t / mid;
            long long s_global = t % S_total;
            long long o = t / S_total;

            long long peer_rank = s_global / S_local;
            long long s_local = s_global % S_local;
            long long hd_global = rank * HD_local + hd_local;

            long long src_idx = ((((o * S_local + s_local) * mid + m) * 3 + q) * HD) + hd_global;
            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[peer_rank]);

            // 16-byte vector load
            const uint4* src_v = reinterpret_cast<const uint4*>(src + src_idx);
            uint4* dst_v = reinterpret_cast<uint4*>(output + e);
            *dst_v = __ldg(src_v);
        }
    } else {
        for (long long e = tid; e < total; e += stride) {
            long long hd_local = e % HD_local;
            long long t = e / HD_local;
            long long q = t % 3;
            t = t / 3;
            long long m = t % mid;
            t = t / mid;
            long long s_global = t % S_total;
            long long o = t / S_total;

            long long peer_rank = s_global / S_local;
            long long s_local = s_global % S_local;
            long long hd_global = rank * HD_local + hd_local;

            long long src_idx = ((((o * S_local + s_local) * mid + m) * 3 + q) * HD) + hd_global;
            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[peer_rank]);
            output[e] = src[src_idx];
        }
    }
}

void launch_fused_a2a_qkv_bf16(
    torch::Tensor peer_ptrs,
    torch::Tensor output,
    int world_size,
    int rank,
    int64_t outer,
    int64_t S_local,
    int64_t mid,
    int64_t HD,
    int64_t HD_local
) {
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(output.dtype() == torch::kBFloat16);

    int threads = 256;
    long long total = outer * S_local * world_size * mid * 3 * HD_local;
    long long total_units = (HD_local % 8 == 0) ? (total / 8) : total;
    int blocks = (int)std::min<long long>((total_units + threads - 1) / threads, 65535LL);
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_a2a_qkv_kernel_bf16<<<blocks, threads, 0, stream>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)output.data_ptr<at::BFloat16>(),
        world_size, rank,
        outer, S_local, mid, HD, HD_local
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_a2a_qkv_bf16", &launch_fused_a2a_qkv_bf16,
          "Fused QKV all-to-all + transpose using symm_mem peer pointers");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_qkv_a2a_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, group):
    key = (numel, dtype, device, id(group))
    e = _symm_cache.get(key)
    if e is not None:
        return e
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


@torch.no_grad()
def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        # Trivial: just possibly unpad
        out = qkv_tensor
        sp = 1
        if unpadded_dim_size and unpadded_dim_size % sp != 0:
            pass
        return out

    sp_world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    assert qkv_tensor.dtype == torch.bfloat16, "This optimized path expects bf16"
    qkv_tensor = qkv_tensor.contiguous()

    orig_shape = list(qkv_tensor.shape)
    ndim = qkv_tensor.dim()
    qkv_proj_dim = orig_shape[-1]
    HD = qkv_proj_dim // 3
    assert qkv_proj_dim % 3 == 0
    assert HD % sp_world == 0, "H*D must be divisible by world size"
    HD_local = HD // sp_world

    # Compute outer / mid wrt seq_dim
    # Normalize seq_dim
    if seq_dim < 0:
        seq_dim_norm = ndim + seq_dim
    else:
        seq_dim_norm = seq_dim
    outer = 1
    for i in range(seq_dim_norm):
        outer *= orig_shape[i]
    S_local = orig_shape[seq_dim_norm]
    mid = 1
    for i in range(seq_dim_norm + 1, ndim - 1):
        mid *= orig_shape[i]

    numel = qkv_tensor.numel()
    device = qkv_tensor.device

    # Lazy ext compile (only rank 0 first to populate cache, then barrier)
    if rank == 0:
        _get_ext()
    dist.barrier(group)
    ext = _get_ext()

    buf, hdl, ptrs_tensor = _get_symm_buf(numel, torch.bfloat16, device, group)

    # Copy local input into symm buffer
    buf.copy_(qkv_tensor.view(-1))

    # Cross-rank synchronization: ensure all peers have written their input
    hdl.barrier(channel=0)

    # Output shape
    out_shape = list(orig_shape)
    out_shape[seq_dim_norm] = S_local * sp_world
    out_shape[-1] = qkv_proj_dim // sp_world  # 3 * HD_local

    output = torch.empty(out_shape, dtype=torch.bfloat16, device=device)

    ext.launch_fused_a2a_qkv_bf16(
        ptrs_tensor,
        output,
        sp_world,
        rank,
        outer,
        S_local,
        mid,
        HD,
        HD_local,
    )

    # Post-kernel barrier so peers don't overwrite buf before our reads complete
    hdl.barrier(channel=1)

    if not restore_shape:
        # Reference returns the tensor still in "after all-to-all" view (with the
        # extra '3' dim split out). Build that view from output.
        view_shape = out_shape[:-1] + [3, HD_local]
        return output.view(view_shape)

    # Optional unpad along seq dim
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = output.size(seq_dim_norm) - unpadded_dim_size
        if padding_size > 0:
            slc = [slice(None)] * output.dim()
            slc[seq_dim_norm] = slice(0, -padding_size)
            output = output[tuple(slc)].contiguous()

    return output