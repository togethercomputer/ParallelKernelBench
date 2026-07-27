"""
Distributed element-wise vector add across two ranks using a Triton kernel with
symmetric memory (UVA). Each rank adds its local buffer to the peer's buffer
via a device-visible pointer from symm_mem rendezvous. World size must be 2.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def symmetric_add_kernel(
    local_ptr,
    remote_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    local = tl.load(local_ptr + offs, mask=mask)
    remote_base = tl.cast(remote_ptr, tl.pointer_type(tl.float32))
    remote = tl.load(remote_base + offs, mask=mask)
    tl.store(out_ptr + offs, local + remote, mask=mask)


_symm_cache = None


def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype:
            return c["buf"], c["hdl"], c["out"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty(n, device=device, dtype=dtype)
    _symm_cache = {"n": n, "dtype": dtype, "buf": buf, "hdl": hdl, "out": out}
    return buf, hdl, out


def _launch_symmetric_add(local: torch.Tensor, remote_ptr: int, out: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    symmetric_add_kernel[grid](
        local,
        remote_ptr,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    assert dist.is_initialized()
    assert dist.get_world_size() == 2

    rank = dist.get_rank()
    peer = 1 - rank
    n = tensor.numel()

    buf, hdl, out = _get_symm_state(n, tensor.dtype, tensor.device)
    buf.copy_(tensor.reshape(-1))
    hdl.barrier(channel=0)

    remote_ptr = int(hdl.buffer_ptrs[peer])
    _launch_symmetric_add(buf, remote_ptr, out, n)

    return out.reshape_as(tensor)
