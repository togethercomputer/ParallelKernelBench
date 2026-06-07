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

__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

// Single-block barrier across ranks using signal pad slot `slot`.
__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank, int world_size, int slot
) {
    int tid = threadIdx.x;
    if (tid >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + (uint64_t)slot * world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + (uint64_t)slot * world_size + (uint64_t)tid);
    send_signal(send_addr);
    wait_signal(wait_addr);
}

// For each peer p, read peer's send-buffer slice (indices destined for THIS rank),
// look them up in local_shard, write result into peer's output buffer at the
// position the peer expects (peer-side offset for source rank == this rank).
//
// Layout per rank in symmetric idx_buf: [world_size, max_per_pair] long indices
//   idx_buf[r][s][k] = the k-th index that rank r wants to send to rank s
// Layout per rank in symmetric out_buf: [world_size, max_per_pair, D] bf16
//   out_buf[r][s][k] = vector for the k-th index that rank r requested from rank s
// counts: [world_size, world_size] long; counts[r][s] = how many idx rank r sends to s
__global__ void p2p_lookup_scatter_kernel(
    const uint64_t* __restrict__ idx_buf_ptrs,   // [world_size]
    const uint64_t* __restrict__ out_buf_ptrs,   // [world_size]
    const long* __restrict__ counts,             // [world_size, world_size]
    const __nv_bfloat16* __restrict__ local_shard,
    int rank, int world_size,
    int64_t max_per_pair,
    int64_t shard_size,
    int64_t embed_dim
) {
    // grid.y = peer id (the rank we are serving), grid.x = chunk over its requests
    int peer = blockIdx.y;
    int64_t n_peer = counts[peer * world_size + rank]; // peer wants n_peer items from us
    if (n_peer == 0) return;

    int64_t chunk_start = (int64_t)blockIdx.x * blockDim.y;
    int64_t k = chunk_start + threadIdx.y;
    if (k >= n_peer) return;

    // Read the index from peer's idx_buf[peer][rank][k]
    const long* peer_idx_base = reinterpret_cast<const long*>(idx_buf_ptrs[peer]);
    int64_t global_idx = peer_idx_base[(int64_t)rank * max_per_pair + k];
    int64_t local_idx = global_idx - (int64_t)rank * shard_size;
    if (local_idx < 0) local_idx = 0;
    if (local_idx >= shard_size) local_idx = shard_size - 1;

    // Source row in our local_shard
    const __nv_bfloat16* src_row = local_shard + local_idx * embed_dim;
    // Dest: peer's out_buf[peer][rank][k]
    __nv_bfloat16* peer_out_base = reinterpret_cast<__nv_bfloat16*>(out_buf_ptrs[peer]);
    __nv_bfloat16* dst_row = peer_out_base
        + ((int64_t)rank * max_per_pair + k) * embed_dim;

    // Copy embed_dim elements (use vectorized 4x bf16 = 8 bytes when aligned)
    int tid = threadIdx.x;
    int blockx = blockDim.x;

    // Try 4-wide bf16 (uint64) copies
    if ((embed_dim % 4) == 0
        && ((uintptr_t)src_row % 8 == 0)
        && ((uintptr_t)dst_row % 8 == 0)) {
        const uint64_t* s4 = reinterpret_cast<const uint64_t*>(src_row);
        uint64_t* d4 = reinterpret_cast<uint64_t*>(dst_row);
        int64_t n4 = embed_dim / 4;
        for (int64_t i = tid; i < n4; i += blockx) {
            d4[i] = s4[i];
        }
    } else {
        for (int64_t i = tid; i < embed_dim; i += blockx) {
            dst_row[i] = src_row[i];
        }
    }
}

// Permute a [N, D] bf16 tensor according to permutation perm of length N:
// output[perm[i]] = input[i]
__global__ void permute_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ in_buf,    // [world_size, max_per_pair, D] flat
    __nv_bfloat16* __restrict__ out,             // [N, D]
    const long* __restrict__ src_pair_rank,      // [N] which rank produced
    const long* __restrict__ src_pair_offset,    // [N] which k within that rank
    int64_t N,
    int64_t max_per_pair,
    int64_t embed_dim
) {
    int64_t row = blockIdx.x;
    if (row >= N) return;
    long sr = src_pair_rank[row];
    long so = src_pair_offset[row];
    const __nv_bfloat16* src = in_buf + (sr * max_per_pair + so) * embed_dim;
    __nv_bfloat16* dst = out + row * embed_dim;
    int tid = threadIdx.x;
    int bx = blockDim.x;
    if ((embed_dim % 4) == 0
        && ((uintptr_t)src % 8 == 0)
        && ((uintptr_t)dst % 8 == 0)) {
        const uint64_t* s4 = reinterpret_cast<const uint64_t*>(src);
        uint64_t* d4 = reinterpret_cast<uint64_t*>(dst);
        int64_t n4 = embed_dim / 4;
        for (int64_t i = tid; i < n4; i += bx) d4[i] = s4[i];
    } else {
        for (int64_t i = tid; i < embed_dim; i += bx) dst[i] = src[i];
    }
}

void launch_barrier(
    torch::Tensor signal_pad_ptrs,
    int64_t rank, int64_t world_size, int64_t slot
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d = reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());
    barrier_kernel<<<1, world_size, 0, stream>>>(d, (int)rank, (int)world_size, (int)slot);
}

void launch_p2p_lookup_scatter(
    torch::Tensor idx_buf_ptrs,
    torch::Tensor out_buf_ptrs,
    torch::Tensor counts,
    torch::Tensor local_shard,
    int64_t rank, int64_t world_size,
    int64_t max_per_pair,
    int64_t shard_size,
    int64_t embed_dim,
    int64_t max_n_per_peer
) {
    if (max_n_per_peer == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int items_per_block = 4;
    int threads_x = 64;
    dim3 block(threads_x, items_per_block, 1);
    int gx = (int)((max_n_per_peer + items_per_block - 1) / items_per_block);
    dim3 grid(gx, (int)world_size, 1);
    const uint64_t* idx_p = reinterpret_cast<const uint64_t*>(idx_buf_ptrs.data_ptr<int64_t>());
    const uint64_t* out_p = reinterpret_cast<const uint64_t*>(out_buf_ptrs.data_ptr<int64_t>());
    p2p_lookup_scatter_kernel<<<grid, block, 0, stream>>>(
        idx_p, out_p,
        counts.data_ptr<long>(),
        reinterpret_cast<const __nv_bfloat16*>(local_shard.data_ptr<at::BFloat16>()),
        (int)rank, (int)world_size,
        max_per_pair, shard_size, embed_dim
    );
}

void launch_permute_rows(
    torch::Tensor in_buf, torch::Tensor out,
    torch::Tensor src_pair_rank, torch::Tensor src_pair_offset,
    int64_t N, int64_t max_per_pair, int64_t embed_dim
) {
    if (N == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 128;
    dim3 grid((unsigned)N, 1, 1);
    permute_rows_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in_buf.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        src_pair_rank.data_ptr<long>(),
        src_pair_offset.data_ptr<long>(),
        N, max_per_pair, embed_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_barrier", &launch_barrier, "device barrier via signal pad");
    m.def("launch_p2p_lookup_scatter", &launch_p2p_lookup_scatter, "p2p lookup + scatter");
    m.def("launch_permute_rows", &launch_permute_rows, "permute rows bf16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("embedding_p2p_ext", CUDA_SRC)
    return _ext


_state = {}

def _get_state(world_size, embed_dim, dtype, device):
    key = ("v1", world_size, embed_dim, dtype, device)
    return _state.get(key), key


def _alloc_state(key, world_size, embed_dim, dtype, device, max_per_pair):
    # Symmetric buffers
    idx_buf = symm_mem.empty((world_size, max_per_pair), device=device, dtype=torch.long)
    idx_hdl = symm_mem.rendezvous(idx_buf, dist.group.WORLD)
    out_buf = symm_mem.empty((world_size, max_per_pair, embed_dim), device=device, dtype=dtype)
    out_hdl = symm_mem.rendezvous(out_buf, dist.group.WORLD)
    # Counts buffer (each rank publishes its send_counts row; peers read it)
    counts_buf = symm_mem.empty((world_size,), device=device, dtype=torch.long)
    counts_hdl = symm_mem.rendezvous(counts_buf, dist.group.WORLD)

    idx_ptrs = torch.tensor(idx_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out_ptrs = torch.tensor(out_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    counts_ptrs = torch.tensor(counts_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    st = {
        "max_per_pair": max_per_pair,
        "idx_buf": idx_buf, "idx_hdl": idx_hdl, "idx_ptrs": idx_ptrs,
        "out_buf": out_buf, "out_hdl": out_hdl, "out_ptrs": out_ptrs,
        "counts_buf": counts_buf, "counts_hdl": counts_hdl, "counts_ptrs": counts_ptrs,
        "signal_pad_ptrs": idx_hdl.signal_pad_ptrs_dev,
    }
    _state[key] = st
    return st


def _ensure_capacity(st, key, world_size, embed_dim, dtype, device, needed):
    if st is None or st["max_per_pair"] < needed:
        # Reallocate with new capacity (round up)
        new_cap = max(needed, 1)
        # round up to multiple of 16 to keep alignment friendly
        new_cap = ((new_cap + 15) // 16) * 16
        if st is not None:
            new_cap = max(new_cap, st["max_per_pair"] * 2)
        st = _alloc_state(key, world_size, embed_dim, dtype, device, new_cap)
    return st


@torch.no_grad()
def solution(indices: torch.Tensor, local_shard: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = local_shard.shape[0]
    embed_dim = local_shard.shape[1]
    dtype = local_shard.dtype

    indices = indices.contiguous()
    if indices.device != device:
        indices = indices.to(device)
    N = indices.numel()

    # JIT-compile (first call): make rank 0 compile, others wait via dist.barrier.
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    # Step 1: bucket indices by target rank using sort (stable, on-device).
    if N > 0:
        target_ranks = torch.div(indices, shard_size, rounding_mode='floor').to(torch.long)
        target_ranks.clamp_(0, world_size - 1)
        # Sort by target_ranks; gather sorted indices and original positions
        sorted_tr, perm = torch.sort(target_ranks, stable=True)
        sorted_indices = indices[perm]
        # send_counts via bincount
        send_counts = torch.bincount(sorted_tr, minlength=world_size).to(torch.long)
    else:
        sorted_indices = torch.empty(0, dtype=torch.long, device=device)
        perm = torch.empty(0, dtype=torch.long, device=device)
        send_counts = torch.zeros(world_size, dtype=torch.long, device=device)

    # We need each peer to know how many we send to it -> we publish send_counts
    # in symmetric counts_buf, then peers read it.
    # Counts matrix: counts[r][s] = how many r sends to s. We publish row `rank`.
    # First, we need state, but capacity depends on max sends — we don't know peer
    # counts yet. Use a two-step: publish send_counts, barrier, peers read full matrix,
    # compute global max_per_pair, then size symmetric idx/out buffers.

    # Use a small persistent symmetric counts buffer of shape [world_size] per rank.
    # We need it allocated — bootstrap a minimal state if absent.
    st_existing, key = _get_state(world_size, embed_dim, dtype, device)
    if st_existing is None:
        st = _alloc_state(key, world_size, embed_dim, dtype, device, max_per_pair=16)
    else:
        st = st_existing

    # Publish our send_counts into our symmetric counts_buf
    st["counts_buf"].copy_(send_counts)
    ext.launch_barrier(st["signal_pad_ptrs"], rank, world_size, 0)

    # Read counts matrix: counts_matrix[r] = peer r's send_counts vector
    # We can read peers' counts_buf via P2P. Build matrix on device.
    counts_matrix = torch.empty((world_size, world_size), dtype=torch.long, device=device)
    # Each peer's counts_buf is a length-world_size tensor. Use buffer pointers.
    buf_ptrs = st["counts_hdl"].buffer_ptrs
    for r in range(world_size):
        ptr = int(buf_ptrs[r])
        peer_counts = torch.from_dlpack(
            _as_tensor_from_ptr(ptr, (world_size,), torch.long, device)
        ) if False else None
        # Use simpler approach: construct via cuda IPC isn't needed; use UnsafeTensor pattern.
        # We instead pack via a small kernel-free path: use torch.empty + cudaMemcpyPeer-like
        # via from_blob is not exposed in python. Fall back: reuse our own buf for our row,
        # and use a tiny custom kernel? Simpler: use dist.all_gather_into_tensor for counts only.
        pass

    # Simpler & robust: use a single all_gather for the small counts vector.
    counts_matrix_flat = torch.empty(world_size * world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(counts_matrix_flat, send_counts)
    counts_matrix = counts_matrix_flat.view(world_size, world_size)

    # Determine max_per_pair globally
    max_per_pair_needed = int(counts_matrix.max().item()) if world_size > 0 else 0
    st = _ensure_capacity(st_existing if st_existing is not None else st, key,
                          world_size, embed_dim, dtype, device, max_per_pair_needed)

    max_per_pair = st["max_per_pair"]

    # Step 2: write our sorted indices into our symmetric idx_buf at rows [s], offsets [0..send_counts[s])
    # Layout: idx_buf[s, k]  (rows == target rank s)
    if N > 0:
        # send_offsets per target s = cumulative sum of send_counts
        offsets = torch.zeros(world_size + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(send_counts, dim=0)
        # Build destination row indices per element: it's sorted_tr already
        # Build dest position within row: position - offsets[sorted_tr]
        pos_in_row = torch.arange(N, device=device, dtype=torch.long) - offsets[sorted_tr]
        # Scatter into idx_buf
        idx_buf = st["idx_buf"]  # shape [world_size, max_per_pair]
        # Clear (optional) — not needed since kernel only reads up to count
        idx_buf[sorted_tr, pos_in_row] = sorted_indices
    # Barrier so all peers' idx_buf are visible
    ext.launch_barrier(st["signal_pad_ptrs"], rank, world_size, 1)

    # Step 3: P2P lookup + scatter directly into peers' out_buf
    # counts.flatten passed as [world_size*world_size] long
    counts_flat = counts_matrix.contiguous().view(-1)
    # max_n_per_peer = max over peers p of counts_matrix[p, rank]
    if world_size > 0:
        col = counts_matrix[:, rank]
        max_n_per_peer = int(col.max().item()) if col.numel() > 0 else 0
    else:
        max_n_per_peer = 0

    # Ensure local_shard is bf16 contiguous (per spec it should already be)
    ls = local_shard.contiguous()
    if ls.dtype != torch.bfloat16:
        # Upcast path: do it in fp32 fallback by using a temp; but spec says bf16.
        ls = ls.to(torch.bfloat16)

    ext.launch_p2p_lookup_scatter(
        st["idx_ptrs"], st["out_ptrs"],
        counts_flat,
        ls,
        rank, world_size,
        max_per_pair,
        shard_size,
        embed_dim,
        max_n_per_peer,
    )

    # Barrier so all peers wrote into our out_buf
    ext.launch_barrier(st["signal_pad_ptrs"], rank, world_size, 2)

    # Step 4: Permute out_buf rows back to the original `indices` order.
    # out_buf layout (ours): out_buf[s, k, :] is the vector for our k-th query to rank s,
    # where order matches the sorted order. Original position in `indices` = perm[sorted_pos].
    # We want output[i] = vector for query i in original order.
    # sorted-position for i -> need inverse perm: inv_perm[perm[j]] = j  =>
    #   for each sorted_pos j (with target s=sorted_tr[j], k=pos_in_row[j]),
    #   output[ perm[j] ] = out_buf[s, k]
    # Equivalent: we set src_pair_rank[orig_i] = sorted_tr[j], src_pair_offset[orig_i] = pos_in_row[j]
    # where j is the sorted index whose perm[j] == orig_i.
    out_dtype = local_shard.dtype
    output = torch.empty((N, embed_dim), dtype=out_dtype, device=device)

    if N > 0 and embed_dim > 0:
        src_rank = torch.empty(N, dtype=torch.long, device=device)
        src_off = torch.empty(N, dtype=torch.long, device=device)
        # perm is sorted->original mapping; assign:
        src_rank[perm] = sorted_tr
        src_off[perm] = pos_in_row

        # If output dtype isn't bf16, do the permute into bf16 temp then cast.
        if out_dtype == torch.bfloat16:
            ext.launch_permute_rows(
                st["out_buf"].view(-1, embed_dim).view(world_size, max_per_pair, embed_dim),
                output, src_rank, src_off, N, max_per_pair, embed_dim
            )
        else:
            tmp = torch.empty((N, embed_dim), dtype=torch.bfloat16, device=device)
            ext.launch_permute_rows(
                st["out_buf"], tmp, src_rank, src_off, N, max_per_pair, embed_dim
            )
            output.copy_(tmp.to(out_dtype))

    return output


def _as_tensor_from_ptr(ptr, shape, dtype, device):
    # Unused helper placeholder; kept for clarity. We use all_gather for counts.
    raise NotImplementedError