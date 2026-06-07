from typing import Optional, Tuple

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

// Merge top-k by reading peer partial top-k buffers via UVA pointers.
// Each (query, peer) contributes K candidates. We pick top-K across peers.
// Simple kernel: one block per query, threads cooperate via shared memory.
//
// Layout per peer:
//   sims_peer:   [Q_total, K] bf16   (only rows [q_off..q_off+Q_owner) are valid for owner)
//   labels_peer: [Q_total, K] int64
// Each rank wants its own queries' rows merged across all peers.

template<int K_MAX>
__global__ void merge_topk_kernel(
    const uint64_t* __restrict__ sim_ptrs,     // world_size pointers to bf16 [Q, K]
    const uint64_t* __restrict__ label_ptrs,   // world_size pointers to int64 [Q, K]
    int world_size,
    int Q,
    int K,
    int q_offset,                              // this rank's query offset into the global query layout
    int Q_local,
    float* __restrict__ out_sims,              // [Q_local, K] float (we'll cast outside or keep float)
    int64_t* __restrict__ out_labels           // [Q_local, K]
) {
    int q = blockIdx.x;
    if (q >= Q_local) return;
    int global_q = q + q_offset;

    // Total candidates = world_size * K
    extern __shared__ unsigned char smem_raw[];
    float* s_sims = reinterpret_cast<float*>(smem_raw);
    int64_t* s_labels = reinterpret_cast<int64_t*>(s_sims + world_size * K_MAX);

    int total = world_size * K;
    int tid = threadIdx.x;

    // Load all candidates from peers
    for (int i = tid; i < total; i += blockDim.x) {
        int peer = i / K;
        int kk = i - peer * K;
        const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(sim_ptrs[peer]);
        const int64_t* lp = reinterpret_cast<const int64_t*>(label_ptrs[peer]);
        size_t row_off = (size_t)global_q * (size_t)K;
        s_sims[i] = __bfloat162float(sp[row_off + kk]);
        s_labels[i] = lp[row_off + kk];
    }
    __syncthreads();

    // Single-thread selection sort for top-K (K is small; up to 200 typical).
    if (tid == 0) {
        for (int sel = 0; sel < K; ++sel) {
            float best = -INFINITY;
            int best_idx = -1;
            for (int i = 0; i < total; ++i) {
                float v = s_sims[i];
                if (v > best) {
                    best = v;
                    best_idx = i;
                }
            }
            if (best_idx < 0) {
                out_sims[(size_t)q * K + sel] = -INFINITY;
                out_labels[(size_t)q * K + sel] = -1;
            } else {
                out_sims[(size_t)q * K + sel] = best;
                out_labels[(size_t)q * K + sel] = s_labels[best_idx];
                s_sims[best_idx] = -INFINITY;
            }
        }
    }
}

void launch_merge_topk(
    torch::Tensor sim_ptrs,      // int64 [world]
    torch::Tensor label_ptrs,    // int64 [world]
    int64_t world_size,
    int64_t Q_total,
    int64_t K,
    int64_t q_offset,
    int64_t Q_local,
    torch::Tensor out_sims,      // float [Q_local, K]
    torch::Tensor out_labels     // int64 [Q_local, K]
) {
    if (Q_local <= 0) return;
    int threads = 128;
    int blocks = (int)Q_local;

    int K_MAX = 256;  // upper bound padding for shared memory layout
    size_t smem = (size_t)world_size * K_MAX * (sizeof(float) + sizeof(int64_t));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // dispatch single instantiation
    merge_topk_kernel<256><<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const uint64_t*>(sim_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(label_ptrs.data_ptr<int64_t>()),
        (int)world_size,
        (int)Q_total,
        (int)K,
        (int)q_offset,
        (int)Q_local,
        out_sims.data_ptr<float>(),
        out_labels.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_merge_topk", &launch_merge_topk, "Top-K merge across peers via UVA");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_knn_merge_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_query_symm(D, max_Q, dtype, device, group):
    key = ("queries", D, max_Q, dtype, device)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty((max_Q, D), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _cache[key] = (buf, hdl, ptrs)
    return _cache[key]


def _get_partial_symm(Q_total, K, device, group):
    key = ("partial", Q_total, K, device)
    if key in _cache:
        return _cache[key]
    sims_buf = symm_mem.empty((Q_total, K), device=device, dtype=torch.bfloat16)
    sims_hdl = symm_mem.rendezvous(sims_buf, group)
    labels_buf = symm_mem.empty((Q_total, K), device=device, dtype=torch.int64)
    labels_hdl = symm_mem.rendezvous(labels_buf, group)
    sim_ptrs = torch.tensor(sims_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    label_ptrs = torch.tensor(labels_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _cache[key] = (sims_buf, sims_hdl, labels_buf, labels_hdl, sim_ptrs, label_ptrs)
    return _cache[key]


@torch.no_grad()
def solution(
    test_features_rank: torch.Tensor,
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    max_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    if max_k > train_features_rank_T.shape[1]:
        raise ValueError("max_k must not exceed the local train shard size")

    device = test_features_rank.device
    dtype = test_features_rank.dtype
    Q_local, D = test_features_rank.shape

    # Compile extension once (rank 0 first to avoid races on shared cache).
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    # ---- Step 1: Exchange query shapes so every rank knows each peer's Q ----
    qsizes = torch.zeros(world_size, dtype=torch.int64, device=device)
    qsizes[rank] = Q_local
    dist.all_reduce(qsizes, op=dist.ReduceOp.SUM, group=group)
    qsizes_cpu = qsizes.cpu().tolist()
    Q_total = int(sum(qsizes_cpu))
    q_offsets = [0]
    for s in qsizes_cpu[:-1]:
        q_offsets.append(q_offsets[-1] + int(s))
    max_Q = max(qsizes_cpu)

    # ---- Step 2: Place own queries into symmetric buffer ----
    qbuf, qhdl, qptrs = _get_query_symm(D, max_Q, dtype, device, group)
    qbuf[:Q_local].copy_(test_features_rank)

    # ---- Step 3: Allocate symmetric partial top-k buffers (sized to Q_total) ----
    sims_buf, sims_hdl, labels_buf, labels_hdl, sim_ptrs, label_ptrs = \
        _get_partial_symm(Q_total, max_k, device, group)

    # Synchronize so all peers' query buffers are filled before we read.
    qhdl.barrier(channel=0)

    # ---- Step 4: For each owner, read peer queries (UVA), compute local top-k,
    #              write into our symmetric partial buffer at the owner's row range.
    # Use multiple streams to pipeline matmuls.
    main_stream = torch.cuda.current_stream(device=device)
    streams = [torch.cuda.Stream(device=device) for _ in range(min(2, world_size))]

    train_labels_row = train_labels_rank.view(1, -1)

    for i, owner in enumerate(range(world_size)):
        Q_owner = int(qsizes_cpu[owner])
        if Q_owner == 0:
            continue
        q_off = q_offsets[owner]

        s = streams[i % len(streams)]
        s.wait_stream(main_stream)
        with torch.cuda.stream(s):
            if owner == rank:
                queries = qbuf[:Q_owner]
            else:
                # Build a tensor view over peer's symmetric query buffer via UVA.
                peer_ptr = int(qhdl.buffer_ptrs[owner])
                # Wrap into a tensor using from_blob via torch.cuda; use storage trick:
                # We allocate a CPU descriptor and use torch.utils.dlpack? Simpler:
                # Use a uint8 storage view through torch._C._CudaDeviceptr... 
                # Easiest portable approach: cudaMemcpyAsync into local staging, but
                # that defeats the purpose. Instead, use torch.as_strided on a fake
                # tensor produced via from_blob through cpp ext-free path:
                queries = _tensor_from_ptr(peer_ptr, (Q_owner, D), dtype, device)

            # GEMM: (Q_owner, D) @ (D, T_local) -> (Q_owner, T_local) bf16
            similarity = torch.matmul(queries, train_features_rank_T)
            topk_sims, idx = similarity.topk(max_k, dim=1, largest=True, sorted=True)
            topk_labels = torch.gather(
                train_labels_row.expand(Q_owner, -1), 1, idx
            )
            # Write into symmetric partial buffer at rows [q_off, q_off+Q_owner)
            sims_buf[q_off:q_off + Q_owner].copy_(topk_sims.to(torch.bfloat16))
            labels_buf[q_off:q_off + Q_owner].copy_(topk_labels)

        main_stream.wait_stream(s)

    # ---- Step 5: barrier on partial buffers so all peers' partials are visible ----
    sims_hdl.barrier(channel=1)

    # ---- Step 6: Merge: each rank merges across peers for its own queries ----
    out_sims_f = torch.empty((Q_local, max_k), device=device, dtype=torch.float32)
    out_labels = torch.empty((Q_local, max_k), device=device, dtype=torch.int64)

    if Q_local > 0:
        _get_ext().launch_merge_topk(
            sim_ptrs, label_ptrs,
            world_size, Q_total, max_k,
            q_offsets[rank], Q_local,
            out_sims_f, out_labels,
        )

    # Final barrier so we don't overwrite buffers before peers finish reading.
    sims_hdl.barrier(channel=2)

    out_sims = out_sims_f.to(dtype)
    return out_sims, out_labels


# ---- helper: build a tensor view from a raw device pointer (no copy) ----
# We use torch's CUDA caching allocator-free path via cudaIpcOpenMemHandle? No:
# symm_mem buffer_ptrs are already valid in our address space (UVA). Use
# torch.cuda.memory utilities through a tiny ctypes wrapper.

import ctypes
_libcudart = None

def _tensor_from_ptr(ptr: int, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Construct a torch.Tensor that aliases the memory at `ptr` (device-side, UVA).
    Uses torch.utils.cpp_extension-free approach via from_dlpack on a synthesized
    capsule is complex; instead, leverage torch.Tensor.set_ on a fresh tensor with
    a Storage wrapping the pointer.
    """
    nbytes = 1
    for s in shape:
        nbytes *= s
    elem_size = torch.tensor([], dtype=dtype).element_size()
    nbytes *= elem_size

    # Use torch's UntypedStorage._new_with_weak_ptr? Not available publicly.
    # Use the documented path: torch.cuda.caching_allocator-independent storage
    # via torch.UntypedStorage.from_buffer is CPU-only.
    # Workaround: use a small inline cpp call. We piggyback on the compiled ext.
    return _ptr_to_tensor_helper(ptr, shape, dtype, device, nbytes)


# Extend the CUDA extension with a from_blob helper. We'll lazily JIT a tiny
# second extension to avoid editing CUDA_SRC above.

_FROMBLOB_SRC = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

torch::Tensor tensor_from_ptr(
    int64_t ptr,
    std::vector<int64_t> shape,
    int64_t dtype_int,
    int64_t device_index
) {
    auto options = torch::TensorOptions()
        .dtype(static_cast<c10::ScalarType>(dtype_int))
        .device(torch::kCUDA, device_index);
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    return torch::from_blob(p, shape, [](void*){}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_from_ptr", &tensor_from_ptr, "from_blob device pointer");
}
'''

_fb_ext = None
def _get_fb_ext():
    global _fb_ext
    if _fb_ext is None:
        _fb_ext = compile_cuda_extension("dinov2_knn_fromblob_ext", _FROMBLOB_SRC)
    return _fb_ext


def _ptr_to_tensor_helper(ptr, shape, dtype, device, nbytes):
    ext = _get_fb_ext()
    dtype_int = int(dtype)
    return ext.tensor_from_ptr(int(ptr), list(shape), dtype_int, device.index)