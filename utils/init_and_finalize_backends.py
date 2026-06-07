"""
Backend init/finalize for reference, triton, cuda, and parallelkittens.

Used by scripts/worker.py to set up and tear down the distributed environment
per backend.
"""

import os
import torch
import torch.distributed as dist


def _init_nccl_process_group(rank: int, world_size: int) -> None:
    """NCCL init compatible with PyTorch 2.6+ (device_id must be torch.device, not int)."""
    kwargs = dict(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        dist.init_process_group(**kwargs, device_id=torch.device("cuda", rank))
    except TypeError:
        dist.init_process_group(**kwargs)


# ---------------------------------------------------------------------------
# Backend: reference (NCCL)
# ---------------------------------------------------------------------------

def init_reference(rank: int, world_size: int) -> None:
    """Initialize torch.distributed with NCCL for reference backend."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    _init_nccl_process_group(rank, world_size)


def finalize_reference() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Backend: cuda (raw CUDA via load_inline + symmetric memory UVA)
# ---------------------------------------------------------------------------

def init_cuda(rank: int, world_size: int) -> None:
    """Initialize torch.distributed with NCCL for the cuda backend.

    Symmetric memory (torch.distributed._symmetric_memory) requires NCCL and
    device_id so it can set up CUDA IPC mappings.  The actual symmetric
    allocations happen inside each solution file.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    _init_nccl_process_group(rank, world_size)


def finalize_cuda() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Backend: triton (identical to cuda)
# ---------------------------------------------------------------------------

def init_triton(rank: int, world_size: int) -> None:
    init_cuda(rank, world_size)


def finalize_triton() -> None:
    finalize_cuda()


# ---------------------------------------------------------------------------
# Backend: parallelkittens (ThunderKittens via load_inline + symmetric memory UVA)
# ---------------------------------------------------------------------------

def init_parallelkittens(rank: int, world_size: int) -> None:
    """Initialize torch.distributed with NCCL for the parallelkittens backend.

    ThunderKittens handles multi-GPU coordination internally; we just need
    NCCL + device_id for symmetric memory UVA mappings.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    _init_nccl_process_group(rank, world_size)


def finalize_parallelkittens() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
