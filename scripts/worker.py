#!/usr/bin/env python3
"""
Distributed "worker" script for running multi-GPU programs.

Supports:
  - reference: NCCL via torch.distributed (mp.spawn or torchrun)
  - triton + Torch symmetric memory
  - cuda: raw CUDA kernels via load_inline + symmetric memory UVA
  - parallelkittens: ThunderKittens kernels via load_inline + symmetric memory UVA
"""

import os
import sys
import argparse
import traceback
import importlib.util
from typing import Any, Callable, Optional, Tuple

# PKB eval targets Hopper (H100).
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

from utils.input_output_tensors import save_tensor, create_input_tensor
from utils.init_and_finalize_backends import (
    init_reference,
    finalize_reference,
    init_triton,
    finalize_triton,
    init_cuda,
    finalize_cuda,
    init_parallelkittens,
    finalize_parallelkittens,
)

BACKENDS = ("reference", "triton", "cuda", "parallelkittens")

def _isolate_torch_extensions_cache(rank: int) -> None:
    """Avoid torchrun races on ~/.cache/torch_extensions (concurrent JIT → corrupt .so)."""
    default_base = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    base = os.environ.get("TORCH_EXTENSIONS_DIR", default_base)
    isolated = os.path.join(base, f"pkb_rank_{rank}")
    os.makedirs(isolated, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = isolated

def _perf_run_fn_and_first_tensor(
    solution_fn, x: tuple
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """
    build (input_tensor, run_fn) for measure_solution_performance.

    This is needed because some problems pass non-Tensors first (e.g. int flags); perf must time using the first
    torch.Tensor and substitute only that tensor when run_fn(t) is invoked.
    """
    input_tensor: Optional[torch.Tensor] = None
    first_tensor_idx: Optional[int] = None
    for i, a in enumerate(x):
        if isinstance(a, torch.Tensor):
            input_tensor = a
            first_tensor_idx = i
            break
    if input_tensor is None or first_tensor_idx is None:
        raise RuntimeError(
            "create_input_tensor returned no torch.Tensor; cannot run --measure_perf"
        )

    def run_fn(t: torch.Tensor):
        args = list(x)
        args[first_tensor_idx] = t
        return solution_fn(*args)

    return input_tensor, run_fn


# ---------------------------------------------------------------------------
# Shared: load solution, run, save
# ---------------------------------------------------------------------------

def load_solution(path: str, backend: str):
    """Load solution module; return (solution_fn, solution_mod)."""
    spec = importlib.util.spec_from_file_location("solution_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "solution"), f"'solution' not found in {path}"
    return mod.solution, mod


def run_worker(
    rank: int,
    world_size: int,
    backend: str,
    problem_py: str,
    logs_dir: str,
    shape: tuple,
    dtype: torch.dtype,
    problem_id: int,
    measure_perf: bool = False,
    profile: bool = False,
    trial: int = 0,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
) -> None:
    """Single-rank worker logic; backend-specific init/finalize and solution call."""
    _isolate_torch_extensions_cache(rank)
    solution_fn, solution_mod = load_solution(problem_py, backend)

    backend_initialized = False
    try:
        # Backend-specific init
        if backend == "reference":
            init_reference(rank, world_size)
        elif backend == "triton":
            init_triton(rank, world_size)
        elif backend == "cuda":
            init_cuda(rank, world_size)
        elif backend == "parallelkittens":
            init_parallelkittens(rank, world_size)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        backend_initialized = True
        print(f"[Rank {rank}/{world_size}] Backend={backend} GPU {torch.cuda.current_device()}")

        dev = torch.device("cuda", rank)
        # create_input_tensor always returns a tuple of tensors; we call solution_fn(*x)
        x = create_input_tensor(rank, world_size, problem_id, shape, dtype, trial=trial, device=dev)

        y = solution_fn(*x)
        torch.cuda.synchronize()
        dist.barrier()

        save_tensor(y, logs_dir, rank)

        if rank == 0:
            print(f"Wrote per-rank outputs to: {logs_dir}")

        dist.barrier()

        if measure_perf:
            from utils.performance import measure_solution_performance

            input_tensor, run_fn = _perf_run_fn_and_first_tensor(solution_fn, x)

            metrics = measure_solution_performance(
                run_fn,
                input_tensor,
                warmup_iters=measure_warmup_iters,
                measure_iters=measure_profiling_iters,
            )
            perf_path = os.path.join(logs_dir, f"rank_{rank}_perf.json")
            with open(perf_path, "w") as f:
                json.dump(metrics, f, indent=2)
            if rank == 0:
                print(f"Wrote per-rank perf to: {logs_dir}")
            dist.barrier()

        if profile:
            traces_dir = os.path.join(logs_dir, "traces")
            os.makedirs(traces_dir, exist_ok=True)

            for _ in range(3):
                solution_fn(*x)
                torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for _ in range(10):
                    solution_fn(*x)
                    torch.cuda.synchronize()

            trace_path = os.path.join(traces_dir, f"rank_{rank}.json")
            prof.export_chrome_trace(trace_path)

            dist.barrier()
            if rank == 0:
                print(f"Wrote profiler traces to: {traces_dir}")

    except Exception:
        print(
            f"[Rank {rank}/{world_size}] worker error — traceback:\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if backend_initialized:
            try:
                if backend == "reference":
                    finalize_reference()
                elif backend == "triton":
                    finalize_triton()
                elif backend == "cuda":
                    finalize_cuda()
                elif backend == "parallelkittens":
                    finalize_parallelkittens()
            except Exception as fin_exc:
                print(
                    f"[Rank {rank}/{world_size}] Warning during finalize: {fin_exc}",
                    file=sys.stderr,
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description="Unified distributed worker for kernel evaluation")
    parser.add_argument("--backend", type=str, required=True, choices=BACKENDS, help="Backend: reference or triton")
    parser.add_argument("--problem_py", type=str, required=True, help="Path to solution .py file")
    parser.add_argument("--logs_dir", type=str, required=True, help="Directory to save output tensors")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs (reference only, when not using torchrun)")
    parser.add_argument("--rows", type=int, default=1024, help="Tensor rows")
    parser.add_argument("--cols", type=int, default=1024, help="Tensor cols")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    parser.add_argument("--problem_id", type=int, default=1, help="Problem ID (affects input tensor shape)")
    parser.add_argument("--measure_perf", action="store_true", help="Run warmup+timed iterations and save perf JSON alongside .pt files")
    parser.add_argument(
        "--measure-warmup-iters",
        type=int,
        default=500,
        dest="measure_warmup_iters",
        help="Warmup iterations before timed perf (used with --measure_perf; default 500)",
    )
    parser.add_argument(
        "--measure-profiling-iters",
        type=int,
        default=100,
        dest="measure_profiling_iters",
        help="Timed profiling iterations inside one CUDA event pair (default 100)",
    )
    parser.add_argument("--profile", action="store_true", help="Run PyTorch profiler and save TensorBoard traces")
    parser.add_argument("--trial", type=int, default=0, help="RNG trial index for eval multi-trial runs (affects random inputs)")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]
    shape = (args.rows, args.cols)

    # Under torchrun: RANK, LOCAL_RANK, WORLD_SIZE are set
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is not None and world_size_env is not None:
        # Launched via torchrun (or similar): one process per GPU
        rank = int(rank_env)
        world_size = int(world_size_env)
        run_worker(
            rank=rank,
            world_size=world_size,
            backend=args.backend,
            problem_py=args.problem_py,
            logs_dir=args.logs_dir,
            shape=shape,
            dtype=dtype,
            problem_id=args.problem_id,
            measure_perf=args.measure_perf,
            profile=args.profile,
            trial=args.trial,
            measure_warmup_iters=args.measure_warmup_iters,
            measure_profiling_iters=args.measure_profiling_iters,
        )
    else:
        raise RuntimeError(
            f"Backend '{args.backend}' must be run under torchrun (RANK/WORLD_SIZE in env). "
            "Use: torchrun --nproc-per-node N worker.py --backend ..."
        )

if __name__ == "__main__":
    main()
