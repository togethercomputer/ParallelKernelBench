"""
Performance measurement utilities for ParallelKernelBench.

Benchmarking convention:

- Bitwise-identical inputs: the caller prepares tensors (e.g. via create_input_tensor); we clone into
  multiple *input groups* so each group is an independent buffer.

- If the timed tensor's footprint is smaller than 3x GPU L2, use several input groups and cycle
  ``i % num_groups`` so later groups naturally evict earlier ones from L2 (cold-cache-like).

- ``warmup_iters`` default 500 for power-steady state; ``measure_iters`` default 100.

- Timing: ``measure_iters + 1`` CUDA events — one before the first iter and one after each iter.
  Events are async stream markers, so kernels still launch back-to-back with no per-iteration host
  synchronization; a single sync at the end lets us read the intervals between consecutive events.

- ``wall_time_ms`` is the mean per-iteration time; ``wall_time_std_ms`` / ``min_time_ms`` /
  ``max_time_ms`` are the real dispersion over the ``measure_iters`` per-iteration samples.
"""

import torch
from typing import Any, Dict

_DEFAULT_L2_BYTES = 50 * 1024 * 1024
# Avoid pathological group counts when input_bytes is tiny (still cap memory / Python overhead).
_MAX_INPUT_GROUPS = 128


def _device_l2_cache_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return _DEFAULT_L2_BYTES
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        n = int(torch.cuda.get_device_properties(idx).l2_cache_size)
    except Exception:
        return _DEFAULT_L2_BYTES
    return n if n > 0 else _DEFAULT_L2_BYTES


def _num_input_groups(input_bytes: int, l2_bytes: int) -> int:
    """1 group if input >= 3×L2; else floor(3×L2 / input) + 1, capped."""
    if input_bytes <= 0:
        return 1
    threshold = 3 * l2_bytes
    if input_bytes >= threshold:
        return 1
    raw = threshold // input_bytes + 1
    return int(min(max(2, raw), _MAX_INPUT_GROUPS))


# This follows the benchmarking conventions in: https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
def measure_solution_performance(
    solution_fn,
    tensor: torch.Tensor,
    warmup_iters: int = 500,
    measure_iters: int = 100,
) -> Dict[str, Any]:
    """
    Time ``solution_fn`` using the module docstring convention (input groups + per-iteration
    CUDA events), returning the mean plus real std / min / max over the per-iteration samples.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"measure_solution_performance expected a torch.Tensor, got {type(tensor).__name__!r}"
        )

    l2_bytes = _device_l2_cache_bytes(tensor.device)
    input_bytes = int(tensor.element_size() * tensor.numel())
    num_input_groups = _num_input_groups(input_bytes, l2_bytes)
    input_groups = [tensor.clone() for _ in range(num_input_groups)]

    torch.cuda.synchronize()
    for i in range(warmup_iters):
        _ = solution_fn(input_groups[i % num_input_groups])
    torch.cuda.synchronize()

    # One event before the first iter plus one after every iter. Events are async stream
    # markers, so kernels still launch back-to-back with no host sync inside the loop; the
    # intervals between consecutive events give real per-iteration samples after one sync.
    events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters + 1)]
    events[0].record()
    for i in range(measure_iters):
        _ = solution_fn(input_groups[i % num_input_groups])
        events[i + 1].record()
    torch.cuda.synchronize()

    per_iter_ms = [
        float(events[i].elapsed_time(events[i + 1])) for i in range(measure_iters)
    ]
    total_ms = float(events[0].elapsed_time(events[measure_iters]))
    avg_time_ms = total_ms / float(measure_iters)

    n = len(per_iter_ms)
    if n > 1:
        variance = sum((t - avg_time_ms) ** 2 for t in per_iter_ms) / (n - 1)
        std_time_ms = variance ** 0.5
    else:
        std_time_ms = 0.0

    return {
        "wall_time_ms": avg_time_ms,
        "wall_time_total_ms": total_ms,
        "wall_time_std_ms": std_time_ms,
        "min_time_ms": min(per_iter_ms) if per_iter_ms else avg_time_ms,
        "max_time_ms": max(per_iter_ms) if per_iter_ms else avg_time_ms,
        "iterations": measure_iters,
        "warmup_iterations": warmup_iters,
        "num_input_groups": float(num_input_groups),
        "l2_cache_size_bytes": float(l2_bytes),
        "timed_tensor_bytes": float(input_bytes),
    }


def format_performance_report(metrics: Dict[str, Any], rank: int = 0) -> str:
    """Format performance metrics as a human-readable string."""
    lines = [
        f"[Rank {rank}] Performance Metrics:",
        f"  Wall-clock time: {metrics['wall_time_ms']:.3f} ± {metrics['wall_time_std_ms']:.3f} ms",
        f"  Time range: [{metrics['min_time_ms']:.3f}, {metrics['max_time_ms']:.3f}] ms",
        f"  Iterations: {metrics['iterations']} (warmup: {metrics['warmup_iterations']})",
    ]
    return "\n".join(lines)
