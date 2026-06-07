"""
Convenience wrapper around torch.utils.cpp_extension.load_inline for the
cuda and parallelkittens backends.

Provides compile_cuda_extension() so solution files don't repeat the
load_inline boilerplate (with_cuda, cflags, verbose, etc.).
"""

import os

from torch.utils.cpp_extension import load_inline


def compile_cuda_extension(
    name: str,
    cuda_src: str,
    extra_cuda_cflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
):
    """JIT-compile a CUDA extension from source and return the loaded module.

    Results are cached under ``TORCH_EXTENSIONS_DIR`` (default ``~/.cache/torch_extensions``).
    ``scripts/worker.py`` sets a per-rank subdirectory when using torchrun so ranks do not
    race writing the same ``.so``.

    In CUDA source strings, include PyTorch headers with their real paths (e.g.
    ``#include <ATen/cuda/CUDAContext.h>``). Linux builds are case-sensitive; ``at/`` is not
    the same as ``ATen/`` and will fail on Modal even if macOS appears to work.
    """
    # Match scripts/worker.py: Hopper-only JIT unless TORCH_CUDA_ARCH_LIST is set (e.g. A100 → 8.0).
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

    cflags = ["-O3"]
    if extra_cuda_cflags:
        cflags.extend(extra_cuda_cflags)
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        extra_cuda_cflags=cflags,
        extra_include_paths=extra_include_paths or [],
        extra_ldflags=extra_ldflags or [],
        with_cuda=True,
        verbose=False,
    )
