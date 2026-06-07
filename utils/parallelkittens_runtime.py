"""
Runtime helpers for the parallelkittens backend.

Provides thin wrappers around TKParallelTensor (exposed by the compiled
ThunderKittens extension via BIND_TK_PARALLEL_TENSOR) so that solution files
don't repeat the rank / world-size / dtype boilerplate.
"""

import torch
import torch.distributed as dist

# Module-level cache so expensive TKParallelTensor objects (VMM + multicast
# IPC setup) are reused across calls with the same key.
_tensor_cache: dict[tuple, object] = {}
_barrier_cache: dict[int, object] = {}


def _tensor_cache_key(ext, shape: tuple | list, dtype: torch.dtype, multicast: bool) -> tuple:
    return (id(ext), tuple(int(x) for x in shape), dtype, bool(multicast))


def get_or_create_parallel_tensor(
    ext,
    shape: tuple | list,
    dtype: torch.dtype = torch.bfloat16,
    *,
    multicast: bool = True,
):
    """Return a cached TKParallelTensor for this (extension, shape, dtype, multicast).

    Reuses VMM + IPC broker setup across ``solution()`` calls so perf benchmarks
    time steady-state work (like NCCL), not allocation. All ranks must call with
    the same key on the same program step (same as ``create_parallel_tensor``).
    """
    key = _tensor_cache_key(ext, shape, dtype, multicast)
    if key not in _tensor_cache:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _tensor_cache[key] = ext.TKParallelTensor(
            list(shape), dtype, rank, world_size, multicast
        )
    return _tensor_cache[key]

NUM_DEVICES = 8
BARRIER_ELEMS = 64


def create_parallel_tensor(ext, shape: tuple | list, dtype: torch.dtype = torch.bfloat16,
                           *, multicast: bool = True):
    """Allocate a TKParallelTensor with VMM + multicast.

    All ranks must call this at the same point in execution because
    TKParallelTensor uses KittensBroker for cross-process IPC.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return ext.TKParallelTensor(list(shape), dtype, rank, world_size, multicast)


def get_or_create_barrier(ext, *, num_devices: int = NUM_DEVICES):
    """Return a cached barrier TKParallelTensor (int32, multicast-enabled).

    All ranks must call this at the same point on the first invocation
    (when the barrier is actually created).
    """
    if num_devices not in _barrier_cache:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _barrier_cache[num_devices] = ext.TKParallelTensor(
            [BARRIER_ELEMS], torch.int32, rank, world_size, True,
        )
    return _barrier_cache[num_devices]
