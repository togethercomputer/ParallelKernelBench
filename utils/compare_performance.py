#!/usr/bin/env python3
"""
Compare performance metrics between reference and solution implementations.

Reads ``rank_*_perf.json`` from::

    logs/problem_<stem>/{reference|cuda|triton|...}/

Aggregate timing uses the slowest rank (max wall_time_ms), matching distributed eval summaries.

Usage:
    python utils/compare_performance.py --problem 1
    python utils/compare_performance.py --problem 1_allreduce --solution cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.problem_id import list_problem_log_backends, resolve_logs_problem_dir


def load_performance_metrics(logs_dir: str) -> Dict[int, Dict]:
    """Load performance metrics from all ranks (rank -> metrics dict)."""
    metrics = {}
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"Warning: Logs directory {logs_dir} does not exist")
        return metrics

    for rank_file in sorted(logs_path.glob("rank_*_perf.json")):
        rank = int(rank_file.stem.split("_")[1])
        with open(rank_file, "r") as f:
            metrics[rank] = json.load(f)

    return metrics


def aggregate_metrics(rank_metrics: Dict[int, Dict]) -> Dict:
    """Aggregate wall-clock metrics across all ranks."""
    if not rank_metrics:
        return {}

    import statistics

    times_ms = [m["wall_time_ms"] for m in rank_metrics.values()]

    return {
        "num_ranks": len(rank_metrics),
        "wall_time_ms": {
            "mean": statistics.mean(times_ms),
            "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            "min": min(times_ms),
            "max": max(times_ms),
            "per_rank": {rank: m["wall_time_ms"] for rank, m in rank_metrics.items()},
        },
    }


def compare_performance(ref_metrics: Dict, sol_metrics: Dict) -> Dict:
    """Compare reference and solution performance (time only)."""
    if not ref_metrics or not sol_metrics:
        return {}

    ref_time = ref_metrics["wall_time_ms"]["max"]
    sol_time = sol_metrics["wall_time_ms"]["max"]

    return {
        "speedup": ref_time / sol_time if sol_time > 0 else 0.0,
        "time_improvement_pct": ((ref_time - sol_time) / ref_time * 100) if ref_time > 0 else 0.0,
        "reference_time_ms": ref_time,
        "solution_time_ms": sol_time,
    }


def format_comparison_report(ref_agg: Dict, sol_agg: Dict, comparison: Dict) -> str:
    """Format a human-readable comparison report."""
    lines = [
        "=" * 70,
        "PERFORMANCE COMPARISON",
        "=" * 70,
        "",
        "Reference Implementation:",
        f"  Wall-clock time: {ref_agg['wall_time_ms']['mean']:.3f} ± {ref_agg['wall_time_ms']['std']:.3f} ms "
        f"(max: {ref_agg['wall_time_ms']['max']:.3f} ms)",
        "",
        "Solution Implementation:",
        f"  Wall-clock time: {sol_agg['wall_time_ms']['mean']:.3f} ± {sol_agg['wall_time_ms']['std']:.3f} ms "
        f"(max: {sol_agg['wall_time_ms']['max']:.3f} ms)",
        "",
        "Comparison:",
        f"  Speedup: {comparison['speedup']:.3f}x",
        f"  Time improvement: {comparison['time_improvement_pct']:+.2f}%",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


def run_performance_comparison(
    problem_logs_dir: Path,
    stem: str,
    solution: str,
    *,
    save_json: bool = True,
) -> bool:
    """
    Load reference and solution perf JSONs, aggregate, compare, and print report.
    Returns True if both ref and solution metrics were found and report was printed; False otherwise.
    """
    ref_logs_dir = problem_logs_dir / "reference"
    sol_logs_dir = problem_logs_dir / solution

    print(f"Problem: {stem}")
    print(f"Loading reference metrics from: {ref_logs_dir}")
    ref_rank_metrics = load_performance_metrics(str(ref_logs_dir))
    print(f"Loading solution metrics from: {sol_logs_dir}")
    sol_rank_metrics = load_performance_metrics(str(sol_logs_dir))

    if not ref_rank_metrics:
        print(f"Error: No reference metrics found in {ref_logs_dir}")
        print("  Run reference with --measure-perf to generate rank_*_perf.json")
        backends = list_problem_log_backends(problem_logs_dir)
        if backends:
            print(f"  Backends with artifacts under {problem_logs_dir}: {', '.join(backends)}")
        return False
    if not sol_rank_metrics:
        print(f"Error: No solution metrics found in {sol_logs_dir}")
        print(f"  Run {solution} with --measure-perf to generate rank_*_perf.json")
        backends = list_problem_log_backends(problem_logs_dir)
        if backends:
            print(f"  Backends with artifacts under {problem_logs_dir}: {', '.join(backends)}")
        return False

    ref_agg = aggregate_metrics(ref_rank_metrics)
    sol_agg = aggregate_metrics(sol_rank_metrics)
    comparison = compare_performance(ref_agg, sol_agg)

    print()
    print(format_comparison_report(ref_agg, sol_agg, comparison))

    if save_json:
        output_file = problem_logs_dir / f"comparison_{solution}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        comparison_data = {
            "problem_stem": stem,
            "solution_type": solution,
            "reference": ref_agg,
            "solution": sol_agg,
            "comparison": comparison,
        }
        with open(output_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nSaved comparison to: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Compare performance metrics")
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        required=True,
        help="Problem id or stem (e.g. 1 -> logs/problem_1_allreduce/)",
    )
    parser.add_argument(
        "--solution",
        "-s",
        type=str,
        default="cuda",
        help="Solution backend subdir (default: cuda)",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Base logs directory (default: <repo>/logs)",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not write comparison_<solution>.json under the problem logs dir",
    )
    args = parser.parse_args()

    logs_base = Path(args.logs_dir).resolve() if args.logs_dir else _PROJECT_ROOT / "logs"

    try:
        problem_logs_dir, stem = resolve_logs_problem_dir(
            _PROJECT_ROOT,
            logs_base,
            args.problem,
            solution_type=args.solution,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not problem_logs_dir.is_dir():
        print(f"Error: logs directory not found: {problem_logs_dir}", file=sys.stderr)
        sys.exit(1)

    ok = run_performance_comparison(
        problem_logs_dir,
        stem,
        args.solution,
        save_json=not args.no_save_json,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
