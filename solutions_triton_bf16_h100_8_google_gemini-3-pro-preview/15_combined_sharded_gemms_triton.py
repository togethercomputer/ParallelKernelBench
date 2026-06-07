"""
Strategy:
- **Zero-Communication Reduce-Scatter:** The reference mathematically assigns each rank its own locally computed block padded with zeros, making the final `reduce_scatter` functionally return the local block. We completely eliminate the collective and return the local block directly.
- **8x Less Comm via UVA All-to-All:** Instead of an all-gather to form `x_full` ([M, H]), rank `r` only needs an `M_local` row slice of `x_full` to compute its part of `z`. We use symmetric memory and a custom Triton UVA kernel to pull exactly the needed `[M_local, H_local]` remote chunks from peers, dropping communication volume by $N\times$.
- **Perfect Compute-Comm Overlap:** The first GEMM is chunked. The main stream instantly computes its local contribution (`local_x @ local_W1`). Concurrently, a background stream fetches the remote blocks. Once fetched, the main stream accumulates the remote contributions, flawlessly hiding the NVLink transfer behind dense Tensor Core math.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

@triton.jit
def gather_remote_blocks_kernel(
    out_ptr,
    ptrs_ptr,
    M_local: int,
    H_local: int,
    rank: int,
    N: int,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    blocks_per_peer = (H_local + BLOCK_H - 1) // BLOCK_H
    remote_peer_idx = pid_h // blocks_per_peer
    
    # Map logical remote peer to actual peer ID
    actual_peer_idx = remote_peer_idx + (1 if remote_peer_idx >= rank else 0)
    
    # Load peer pointer from symmetric memory rendezvous 
    peer_ptr_int = tl.load(ptrs_ptr + actual_peer_idx)
    peer_ptr = peer_ptr_int.to(tl.pointer_type(tl.bfloat16))
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh_local = (pid_h % blocks_per_peer) * BLOCK_H + tl.arange(0, BLOCK_H)
    
    # Rank 'r' needs rows [r*M_local : (r+1)*M_local] from each peer's x_local
    row_offset = rank * M_local
    
    offs_m = rm[:, None]
    offs_h = rh_local[None, :]
    
    src_ptrs = peer_ptr + (row_offset + offs_m) * H_local + offs_h
    
    mask_m = rm < M_local
    mask_h = rh_local < H_local
    mask = mask_m[:, None] & mask_h[None, :]
    
    x = tl.load(src_ptrs, mask=mask)
    
    # Pack into local destination buffer
    rh_global = remote_peer_idx * H_local + rh_local
    stride_m = (N - 1) * H_local
    dst_ptrs = out_ptr + offs_m * stride_m + rh_global[None, :]
    
    tl.store(dst_ptrs, x, mask=mask)

_symm_cache = None

def _get_symm_state(shape, dtype, device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["shape"] == shape and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"], c["ptrs"]

    numel = 1
    for s in shape: 
        numel *= s

    buf = symm_mem.empty(numel, dtype=dtype, device=device).view(shape)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)

    _symm_cache = {
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "buf": buf,
        "hdl": hdl,
        "ptrs": ptrs
    }
    return buf, hdl, ptrs

_comm_stream = None

def _get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
    return _comm_stream

@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x_local.is_cuda and W1.is_cuda and W2.is_cuda, "Inputs must be CUDA tensors"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, H_local = x_local.shape
    H, ffn_dim = W1.shape
    ffn2, H_out = W2.shape
    
    assert ffn_dim == ffn2, f"W1 and W2 inner dims must match: {ffn_dim} vs {ffn2}"
    assert H_out == H, f"W2 out dim must match gathered hidden H: {H_out} vs {H}"
    assert H == H_local * world_size, (
        f"Hidden must split across ranks: H={H}, H_local={H_local}, world_size={world_size}"
    )
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"

    M_local = M // world_size
    x_local = x_local.contiguous()

    main_stream = torch.cuda.current_stream()
    comm_stream = _get_comm_stream()

    sym_x, sym_hdl, ptrs_tensor = _get_symm_state((M, H_local), x_local.dtype, x_local.device)

    # Ensure main stream produced x_local before communication starts
    comm_stream.wait_stream(main_stream)

    if world_size > 1:
        with torch.cuda.stream(comm_stream):
            sym_x.copy_(x_local)
            sym_hdl.barrier(channel=0) # Signifies local write complete

            x_remote_buf = torch.empty(
                (M_local, H - H_local), 
                dtype=x_local.dtype, 
                device=x_local.device
            )

            grid = lambda meta: (
                triton.cdiv(M_local, meta['BLOCK_M']),
                (world_size - 1) * triton.cdiv(H_local, meta['BLOCK_H'])
            )
            
            # Fetch remaining N-1 chunks via UVA directly into a packed contiguous buffer
            gather_remote_blocks_kernel[grid](
                x_remote_buf,
                ptrs_tensor,
                M_local, H_local, rank, world_size,
                BLOCK_M=64, BLOCK_H=64,
                num_warps=4
            )

            comm_event = torch.cuda.Event()
            comm_event.record(comm_stream)

    # Computations heavily overlapped with the background UVA fetch 
    local_x = x_local[rank * M_local : (rank + 1) * M_local]
    local_W1 = W1[rank * H_local : (rank + 1) * H_local]
    
    # Part 1 GEMM
    z_loc = torch.matmul(local_x, local_W1)

    if world_size > 1:
        # Sync main stream only when we strictly need the remote blocks
        main_stream.wait_event(comm_event)

        # Part 2 GEMMs (Remote Left & Right Contributions)
        x_remote_left = x_remote_buf[:, :rank * H_local]
        W1_left = W1[:rank * H_local, :]
        if rank > 0:
            z_loc.addmm_(x_remote_left, W1_left)

        x_remote_right = x_remote_buf[:, rank * H_local:]
        W1_right = W1[(rank + 1) * H_local:, :]
        if rank < world_size - 1:
            z_loc.addmm_(x_remote_right, W1_right)

    # Fused inplace activation & sequence-parallel block projection
    a_loc = F.silu(z_loc, inplace=True)
    y_local = torch.matmul(a_loc, W2)

    if world_size > 1:
        # Prevent the next loop iteration from stomping sym_x before peers finish reading it
        sym_hdl.barrier(channel=1)

    return y_local