#!/usr/bin/env python3
"""
Compare output tensors between reference and solution implementations.

Artifacts from ``run_local.py --download`` live under::

    logs/problem_<stem>/{reference|cuda|triton|...}/rank_*.pt

Usage:
    # Compare reference vs solution (default solution: cuda)
    python utils/compare_outputs.py --problem 1
    python utils/compare_outputs.py --problem 1_allreduce --solution triton

    # Inspect tensors from one backend folder
    python utils/compare_outputs.py --problem 1 --inspect reference
    python utils/compare_outputs.py -p 1 -i cuda --atol 1e-2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from utils.problem_id import (
    list_problem_log_backends,
    resolve_logs_problem_dir,
)

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"


def print_header(text):
    width = 70
    print()
    print(colorize("═" * width, Colors.CYAN))
    print(colorize(f"  {text}", Colors.BOLD + Colors.CYAN))
    print(colorize("═" * width, Colors.CYAN))


def print_subheader(text):
    print()
    print(colorize(f"── {text} ", Colors.BLUE) + colorize("─" * (50 - len(text)), Colors.DIM))


def print_tensor_info(tensor, name, num_elements=8):
    """Print tensor information in a nice format."""
    print(f"  {colorize('Shape:', Colors.DIM)} {colorize(str(tuple(tensor.shape)), Colors.YELLOW)}")
    print(f"  {colorize('Dtype:', Colors.DIM)} {colorize(str(tensor.dtype), Colors.YELLOW)}")
    print(f"  {colorize('Min:', Colors.DIM)}   {tensor.min().item():.6f}")
    print(f"  {colorize('Max:', Colors.DIM)}   {tensor.max().item():.6f}")
    print(f"  {colorize('Mean:', Colors.DIM)}  {tensor.float().mean().item():.6f}")
    
    # Show first few elements
    flat = tensor.flatten()
    n = min(num_elements, len(flat))
    elements = [f"{flat[i].item():.4f}" for i in range(n)]
    suffix = ", ..." if len(flat) > n else ""
    print(f"  {colorize('First', Colors.DIM)} {colorize(str(n), Colors.YELLOW)}{colorize(':', Colors.DIM)} [{', '.join(elements)}{suffix}]")


def compare_tensors(ref_tensor, sol_tensor, atol=1e-5, rtol=1e-5):
    """Compare two tensors and return comparison results."""
    results = {
        "shape_match": ref_tensor.shape == sol_tensor.shape,
        "dtype_match": ref_tensor.dtype == sol_tensor.dtype,
        "exact_match": False,
        "close_match": False,
        "max_diff": float('inf'),
        "mean_diff": float('inf'),
    }
    
    if results["shape_match"]:
        results["exact_match"] = torch.equal(ref_tensor, sol_tensor)
        results["close_match"] = torch.allclose(ref_tensor.float(), sol_tensor.float(), atol=atol, rtol=rtol)
        diff = (ref_tensor.float() - sol_tensor.float()).abs()
        results["max_diff"] = diff.max().item()
        results["mean_diff"] = diff.mean().item()
    
    return results


def _rank_pt_files(directory: str) -> list[str]:
    return sorted(f for f in os.listdir(directory) if f.startswith("rank_") and f.endswith(".pt"))


def inspect_tensors(problem_dir: str, target_type: str) -> None:
    """Inspect and print tensors from a specific backend folder."""
    target_dir = os.path.join(problem_dir, target_type)
    
    print_header(f"Inspecting: {target_type}")
    
    if not os.path.isdir(target_dir):
        print(colorize(f"\n  ✗ Directory not found: {target_dir}", Colors.RED))
        sys.exit(1)
    
    print(f"\n  {colorize('Path:', Colors.DIM)} {target_dir}")
    
    pt_files = _rank_pt_files(target_dir)
    
    if not pt_files:
        print(colorize(f"\n  ✗ No .pt files found in directory", Colors.RED))
        sys.exit(1)
    
    print(f"  {colorize('Files:', Colors.DIM)} {len(pt_files)} tensor files")
    
    for fname in pt_files:
        rank = fname.replace("rank_", "").replace(".pt", "")
        print_subheader(f"Rank {rank}")
        
        fpath = os.path.join(target_dir, fname)
        output = torch.load(fpath, weights_only=True)
        
        # Handle different output types
        if isinstance(output, dict):
            print(f"  {colorize('Output type:', Colors.DIM)} dict with {len(output)} keys")
            for key, value in sorted(output.items()):
                if isinstance(value, torch.Tensor):
                    print(f"\n  {colorize(key + ':', Colors.CYAN)}")
                    print_tensor_info(value, key)
        elif isinstance(output, (tuple, list)):
            print(f"  {colorize('Output type:', Colors.DIM)} {type(output).__name__} with {len(output)} elements")
            for i, value in enumerate(output):
                if isinstance(value, torch.Tensor):
                    print(f"\n  {colorize(f'Output {i}:', Colors.CYAN)}")
                    print_tensor_info(value, f"output_{i}")
        else:
            # Single tensor
            print_tensor_info(output, fname)
    
    print()


def run_output_comparison(
    ref_dir: str,
    sol_dir: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    title: str | None = None,
) -> bool:
    """
    Compare reference and solution .pt outputs in two directories.
    Prints report and returns True if all ranks match within tolerance, False otherwise.
    """
    if title is None:
        title = "Comparing reference vs solution"
    print_header(title)
    
    # Check directories exist
    if not os.path.isdir(ref_dir):
        print(colorize(f"\n  ✗ Reference directory not found: {ref_dir}", Colors.RED))
        return False
    
    if not os.path.isdir(sol_dir):
        print(colorize(f"\n  ✗ Solution directory not found: {sol_dir}", Colors.RED))
        return False
    
    ref_files = _rank_pt_files(ref_dir)
    sol_files = _rank_pt_files(sol_dir)
    
    print(f"\n  {colorize('Reference:', Colors.DIM)} {ref_dir}")
    print(f"  {colorize('Solution:', Colors.DIM)}  {sol_dir}")
    print(f"  {colorize('Tolerance:', Colors.DIM)} atol={atol}, rtol={rtol}")
    
    if not ref_files:
        print(colorize(f"\n  ✗ No .pt files found in reference directory", Colors.RED))
        return False
    
    if not sol_files:
        print(colorize(f"\n  ✗ No .pt files found in solution directory", Colors.RED))
        return False
    
    if ref_files != sol_files:
        print(colorize(f"\n  ⚠ File mismatch!", Colors.YELLOW))
        print(f"    Reference: {ref_files}")
        print(f"    Solution:  {sol_files}")
    
    # Compare each rank
    all_passed = True
    total_ranks = len(ref_files)
    
    for fname in ref_files:
        rank = fname.replace("rank_", "").replace(".pt", "")
        print_subheader(f"Rank {rank}")
        
        ref_path = os.path.join(ref_dir, fname)
        sol_path = os.path.join(sol_dir, fname)
        
        if not os.path.exists(sol_path):
            print(colorize(f"  ✗ Solution file missing: {fname}", Colors.RED))
            all_passed = False
            continue
        
        # Load outputs (can be tensor, tuple, or dict)
        ref_output = torch.load(ref_path, weights_only=True)
        sol_output = torch.load(sol_path, weights_only=True)
        
        # Handle different output types
        if isinstance(ref_output, dict) and isinstance(sol_output, dict):
            # Dict output - compare each key
            print(colorize("  Reference:", Colors.CYAN))
            for key in sorted(ref_output.keys()):
                if key in sol_output:
                    ref_t = ref_output[key]
                    sol_t = sol_output[key]
                    if isinstance(ref_t, torch.Tensor) and isinstance(sol_t, torch.Tensor):
                        print(f"    {colorize(key + ':', Colors.DIM)}")
                        print_tensor_info(ref_t, "reference")
                        
                        print(colorize(f"\n    Solution {key}:", Colors.CYAN))
                        print_tensor_info(sol_t, "solution")
                        
                        # Compare
                        results = compare_tensors(ref_t, sol_t, atol=atol, rtol=rtol)
                        
                        print(colorize(f"\n    Comparison {key}:", Colors.CYAN))
                        if results["shape_match"]:
                            print(f"      {colorize('✓', Colors.GREEN)} Shape match")
                        else:
                            print(f"      {colorize('✗', Colors.RED)} Shape mismatch: {ref_t.shape} vs {sol_t.shape}")
                            all_passed = False
                            continue
                        
                        if results["close_match"]:
                            print(f"      {colorize('✓', Colors.GREEN)} Values match (atol={atol}, rtol={rtol})")
                            print(f"      Max diff: {results['max_diff']:.2e}, Mean diff: {results['mean_diff']:.2e}")
                        else:
                            print(f"      {colorize('✗', Colors.RED)} Values don't match")
                            print(f"      Max diff: {results['max_diff']:.2e}, Mean diff: {results['mean_diff']:.2e}")
                            all_passed = False
                else:
                    print(f"      {colorize('✗', Colors.RED)} Key '{key}' missing in solution")
                    all_passed = False
        elif isinstance(ref_output, (tuple, list)) and isinstance(sol_output, (tuple, list)):
            # Tuple/list output - compare each element
            if len(ref_output) != len(sol_output):
                print(f"  {colorize('✗', Colors.RED)} Output length mismatch: {len(ref_output)} vs {len(sol_output)}")
                all_passed = False
                continue
            
            for i, (ref_t, sol_t) in enumerate(zip(ref_output, sol_output)):
                if isinstance(ref_t, torch.Tensor) and isinstance(sol_t, torch.Tensor):
                    print(colorize(f"  Reference output {i}:", Colors.CYAN))
                    print_tensor_info(ref_t, "reference")
                    
                    print(colorize(f"\n  Solution output {i}:", Colors.CYAN))
                    print_tensor_info(sol_t, "solution")
                    
                    # Compare
                    results = compare_tensors(ref_t, sol_t, atol=atol, rtol=rtol)
                    
                    print(colorize(f"\n  Comparison output {i}:", Colors.CYAN))
                    if results["shape_match"]:
                        print(f"    {colorize('✓', Colors.GREEN)} Shape match")
                    else:
                        print(f"    {colorize('✗', Colors.RED)} Shape mismatch: {ref_t.shape} vs {sol_t.shape}")
                        all_passed = False
                        continue
                    
                    if results["close_match"]:
                        print(f"    {colorize('✓', Colors.GREEN)} Values match (atol={atol}, rtol={rtol})")
                        print(f"    Max diff: {results['max_diff']:.2e}, Mean diff: {results['mean_diff']:.2e}")
                    else:
                        print(f"    {colorize('✗', Colors.RED)} Values don't match")
                        print(f"    Max diff: {results['max_diff']:.2e}, Mean diff: {results['mean_diff']:.2e}")
                        all_passed = False
        else:
            # Single tensor output (original behavior)
            ref_tensor = ref_output if isinstance(ref_output, torch.Tensor) else ref_output
            sol_tensor = sol_output if isinstance(sol_output, torch.Tensor) else sol_output
            
            # Print info
            print(colorize("  Reference:", Colors.CYAN))
            print_tensor_info(ref_tensor, "reference")
            
            print(colorize("\n  Solution:", Colors.CYAN))
            print_tensor_info(sol_tensor, "solution")
            
            # Compare
            results = compare_tensors(ref_tensor, sol_tensor, atol=atol, rtol=rtol)
            
            print(colorize("\n  Comparison:", Colors.CYAN))
            
            # Shape match
            if results["shape_match"]:
                print(f"  {colorize('✓', Colors.GREEN)} Shape match")
            else:
                print(f"  {colorize('✗', Colors.RED)} Shape mismatch: {ref_tensor.shape} vs {sol_tensor.shape}")
                all_passed = False
                continue
            
            # Value comparison
            if results["exact_match"]:
                print(f"  {colorize('✓', Colors.GREEN)} Exact match")
            elif results["close_match"]:
                print(f"  {colorize('✓', Colors.GREEN)} Close match (within tolerance)")
                print(f"    {colorize('Max diff:', Colors.DIM)}  {results['max_diff']:.2e}")
                print(f"    {colorize('Mean diff:', Colors.DIM)} {results['mean_diff']:.2e}")
            else:
                print(f"  {colorize('✗', Colors.RED)} Values differ beyond tolerance")
                print(f"    {colorize('Max diff:', Colors.DIM)}  {results['max_diff']:.2e}")
                print(f"    {colorize('Mean diff:', Colors.DIM)} {results['mean_diff']:.2e}")
                all_passed = False
    
    # Summary
    print_header("Summary")
    if all_passed:
        print(f"\n  {colorize('✓ ALL RANKS PASSED', Colors.GREEN + Colors.BOLD)}")
        print(f"  {colorize(f'{total_ranks}/{total_ranks} ranks match within tolerance', Colors.DIM)}")
    else:
        print(f"\n  {colorize('✗ SOME RANKS FAILED', Colors.RED + Colors.BOLD)}")
        print(f"  {colorize('Check the comparison details above', Colors.DIM)}")
    
    print()
    return all_passed


def _default_paths() -> tuple[Path, Path]:
    return _PROJECT_ROOT, _PROJECT_ROOT / "logs"


def main():
    parser = argparse.ArgumentParser(description="Compare reference and solution tensor outputs")
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        required=True,
        help='Problem id or stem (e.g. 1 -> logs/problem_1_allreduce/, or 1_allreduce)',
    )
    parser.add_argument(
        "--solution",
        "-s",
        type=str,
        default="cuda",
        help="Solution backend subdir under logs/problem_<stem>/ (default: cuda)",
    )
    parser.add_argument(
        "--inspect",
        "-i",
        type=str,
        default=None,
        help="Inspect mode: print tensors from one backend folder (e.g. reference, cuda, triton)",
    )
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for comparison")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for comparison")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Base logs directory (default: <repo>/logs)",
    )
    args = parser.parse_args()

    project_root, default_logs = _default_paths()
    logs_base = Path(args.logs_dir).resolve() if args.logs_dir else default_logs

    try:
        problem_logs_dir, stem = resolve_logs_problem_dir(
            project_root,
            logs_base,
            args.problem,
            solution_type=args.solution if not args.inspect else "reference",
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    problem_dir = str(problem_logs_dir)
    if not problem_logs_dir.is_dir():
        available = sorted(
            p.name.removeprefix("problem_")
            for p in logs_base.glob("problem_*")
            if p.is_dir()
        )
        hint = f" Available under {logs_base}: {available}" if available else ""
        print(
            f"Error: logs directory not found: {problem_logs_dir}.{hint}",
            file=sys.stderr,
        )
        print(
            "  Run run_local.py with --download first, e.g.\n"
            "  uv run python run_local.py --nproc-per-node 4 --mode dryrun --problem 1 "
            "--solution reference --download",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.inspect:
        backends = list_problem_log_backends(problem_logs_dir)
        if args.inspect not in backends:
            print(
                f"Error: backend {args.inspect!r} not found under {problem_logs_dir}.",
                file=sys.stderr,
            )
            if backends:
                print(f"  Available backends: {', '.join(backends)}", file=sys.stderr)
            sys.exit(1)
        inspect_tensors(problem_dir, args.inspect)
        sys.exit(0)

    ref_dir = os.path.join(problem_dir, "reference")
    sol_dir = os.path.join(problem_dir, args.solution)
    title = f"Comparing {stem}: reference vs {args.solution}"
    passed = run_output_comparison(ref_dir, sol_dir, atol=args.atol, rtol=args.rtol, title=title)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

