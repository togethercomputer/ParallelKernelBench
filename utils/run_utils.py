"""
Shared helpers for runner scripts.

This module owns runner-neutral path resolution, preflight checks, performance
artifact aggregation, and harness-failure classification. Runner entrypoints
should import these helpers instead of importing from one another.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils.problem_id import rank_outputs_compare_details


class JobCapture(Protocol):
    """Minimal torchrun job result surface used by eval trial reporting."""

    returncode: int
    stdout: str
    stderr: str
    artifacts: list[tuple[str, bytes]]
    elastic_error_files: str | None
    worker_logs_dir: str | None


# Header: return the single expected .py path for a problem stem and backend.
# Reference uses reference/{stem}.py (e.g. 1_allreduce.py). Custom backends use
# {stem}_{backend}.py (e.g. 1_allreduce_cuda.py) under solutions_<backend>/ or --solutions-dir.
def solution_path(
    project_root: Path,
    problem_stem: str,
    backend: str,
    *,
    solutions_dir: Path | None = None,
) -> Path:
    """
    Expected on-disk path for one problem/backend pair.

    - ``reference`` → ``reference/{stem}.py`` (e.g. ``1_allreduce.py``)
    - ``cuda`` / ``triton`` / ``parallelkittens`` → ``{stem}_{backend}.py`` under ``--solutions-dir`` if set, else ``solutions_{backend}/``
    """
    if backend == "reference":
        return project_root / "reference" / f"{problem_stem}.py"
    if solutions_dir is not None:
        return solutions_dir / f"{problem_stem}_{backend}.py"
    return project_root / f"solutions_{backend}" / f"{problem_stem}_{backend}.py"


# This function verifies that solution_path exists before launching a job; exit with a
# clear error (and nearby filenames) when the expected file is missing.
def check_solution(
    project_root: Path,
    problem_stem: str,
    backend: str,
    *,
    solutions_dir: Path | None = None,
) -> None:
    """
    Preflight check: the file from :func:`solution_path` must exist.

    Exits with code 1 if not found.
    """
    path = solution_path(project_root, problem_stem, backend, solutions_dir=solutions_dir)
    if path.is_file():
        return
    print(
        f"Error: no file for problem {problem_stem!r}, backend {backend!r}.",
        file=sys.stderr,
    )
    print(f"  Expected: {path}", file=sys.stderr)
    listing_dir = path.parent
    if listing_dir.is_dir():
        existing = sorted(listing_dir.glob("*.py"))
        if existing:
            names = [p.name for p in existing[:20]]
            suffix = "..." if len(existing) > 20 else ""
            print(f"  .py files here: {names}{suffix}", file=sys.stderr)
    sys.exit(1)


# Header: read solution module source for a non-reference backend (for job payloads
# or staging); reference jobs return None because the worker uses reference/ directly.
def read_solution_source(
    project_root: Path,
    problem_stem: str,
    backend: str,
    *,
    solutions_dir: Path | None = None,
) -> str | None:
    """Return full ``.py`` text for ``{stem}_{backend}.py``, or ``None`` for reference."""
    if backend == "reference":
        return None
    return solution_path(
        project_root, problem_stem, backend, solutions_dir=solutions_dir
    ).read_text(encoding="utf-8")


# Back-compat aliases (run_together / older scripts).
check_solution_exists = check_solution
read_solution_source_for_payload = read_solution_source


# Header: write in-memory job artifacts (relative path, bytes) to a local directory,
# preserving subpaths and printing one line per file when --download is used.
def write_artifacts_flat(artifacts: list[tuple[str, bytes]], dest_dir: Path) -> None:
    """
    Save downloaded worker outputs under ``dest_dir``.

    Each ``name`` is a relative path inside the job log tree (e.g. ``rank_0.pt``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts:
        out = dest_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"  Downloaded: {out}")


# Header: print torchrun failure info already captured on the outcome (stderr/stdout,
# elastic file blob, on-disk worker log directory). No extra log scraping.
def print_job_failure_diagnostics(out: JobCapture) -> None:
    """Print exit code, worker log path, and subprocess/elastic output to stderr."""
    print(f"  exit code: {out.returncode}", file=sys.stderr)
    if out.worker_logs_dir:
        print(f"  worker logs: {out.worker_logs_dir}", file=sys.stderr)
    if out.elastic_error_files:
        print("  --- torch elastic ---", file=sys.stderr)
        print(out.elastic_error_files, file=sys.stderr)
    if out.stderr:
        print("  --- stderr ---", file=sys.stderr)
        print(out.stderr, file=sys.stderr)
    if out.stdout:
        print("  --- stdout ---", file=sys.stderr)
        print(out.stdout, file=sys.stderr)


# Header: build a JSON-safe dict of truncated stdout/stderr (and related metadata)
# for one torchrun job, to embed under reference_job / solution_job in PKB_EVAL_JSON.
def captured_job_output_for_eval_json(out: JobCapture) -> dict:
    """
    Snapshot one job's captured stdout/stderr for PKB_EVAL_JSON.

    Long streams are truncated so the eval payload stays bounded.
    """
    payload: dict = {"returncode": out.returncode, "artifact_count": len(out.artifacts)}
    if out.stdout:
        payload["outputs_stdout"] = out.stdout[-8000:]
    if out.stderr:
        raw = out.stderr
        if len(raw) > 40000:
            payload["outputs_stderr"] = (
                raw[:32000]
                + f"\n\n... [middle {len(raw) - 40000} chars omitted] ...\n\n"
                + raw[-8000:]
            )
        else:
            payload["outputs_stderr"] = raw
    if out.elastic_error_files:
        payload["elastic_error_files"] = out.elastic_error_files[:24000]
    if out.worker_logs_dir:
        payload["worker_logs_dir"] = out.worker_logs_dir
    return payload


@dataclass
class EvalTrialStepResult:
    """Outcome of processing one eval trial (reference + solution jobs)."""

    trial_report: dict
    all_ok: bool
    stop_early: bool


# Header: after one reference+solution eval trial, fill trial_report, print the
# trial line, optionally download artifacts, and signal whether to stop more trials.
def log_and_print_trial_results(
    *,
    trial: int,
    ref_out: JobCapture,
    sol_out: JobCapture,
    solution_backend: str,
    measure_perf: bool,
    atol: float,
    rtol: float,
    n_trials: int,
    download: bool,
    ref_dir: Path,
    sol_dir: Path,
) -> EvalTrialStepResult:
    """
    Record one eval trial: compare rank outputs or report execution failure.

    Prints a ``trial N: ...`` line, updates ``trial_report`` for PKB_EVAL_JSON, and sets
    ``stop_early`` when the runner should not continue to further RNG trials.
    """
    trial_report: dict = {"trial": trial}
    ref_rc = 0 if ref_out.returncode == 0 and ref_out.artifacts else 1
    sol_rc = 0 if sol_out.returncode == 0 and sol_out.artifacts else 1
    trial_report["reference_returncode"] = ref_out.returncode
    trial_report["solution_returncode"] = sol_out.returncode
    trial_report["reference_artifact_names"] = [k for k, _ in ref_out.artifacts]
    trial_report["solution_artifact_names"] = [k for k, _ in sol_out.artifacts]

    if ref_rc != 0 or sol_rc != 0:
        trial_report["reference_job"] = captured_job_output_for_eval_json(ref_out)
        trial_report["solution_job"] = captured_job_output_for_eval_json(sol_out)
        rj, sj = trial_report["reference_job"], trial_report["solution_job"]
        harness = harness_environment_hint(
            str(rj.get("outputs_stderr") or ""),
            str(rj.get("outputs_stdout") or ""),
            str(sj.get("outputs_stderr") or ""),
            str(sj.get("outputs_stdout") or ""),
        )
        if harness:
            trial_report["result"] = "harness_environment"
            trial_report["harness_hint"] = harness
            tag = "harness_failure (not a kernel correctness issue)"
        elif ref_out.returncode == 130 or sol_out.returncode == 130:
            trial_report["result"] = "interrupted"
            tag = "interrupted (Ctrl+C)"
        else:
            trial_report["result"] = "no_artifacts"
            tag = "cannot_compare (missing rank outputs)"
        line = f"  trial {trial}: {tag}"
        if measure_perf:
            line += " | perf: n/a"
        print(line)
        if harness:
            print(f"    harness: {harness}", file=sys.stderr)
        print(
            f"Local eval: trial {trial}: no rank .pt "
            f"(reference={len(ref_out.artifacts)}, solution={len(sol_out.artifacts)}). "
            "See PKB_EVAL_JSON or failure reports below.",
            file=sys.stderr,
        )
        print("\n--- Reference job ---", file=sys.stderr)
        print_job_failure_diagnostics(ref_out)
        print("\n--- Solution job ---", file=sys.stderr)
        print_job_failure_diagnostics(sol_out)
        trial_report["early_stop"] = "execution_or_harness_failure"
        if n_trials > 1:
            print(
                f"  Stopping after trial {trial + 1}/{n_trials}: "
                "reference or solution failed; further trials skipped.",
                flush=True,
            )
        return EvalTrialStepResult(trial_report=trial_report, all_ok=False, stop_early=True)

    compare = rank_outputs_compare_details(
        ref_out.artifacts, sol_out.artifacts, atol=atol, rtol=rtol
    )
    trial_ok = bool(compare.get("ok"))
    trial_report["result"] = "match" if trial_ok else compare.get("kind", "mismatch")
    trial_report["compare"] = compare

    ref_perf = summarize_perf_artifacts(ref_out.artifacts) if measure_perf else None
    sol_perf = summarize_perf_artifacts(sol_out.artifacts) if measure_perf else None
    speedup = None
    if ref_perf and sol_perf and sol_perf["wall_time_ms_max"] > 0:
        speedup = ref_perf["wall_time_ms_max"] / sol_perf["wall_time_ms_max"]

    line = f"  trial {trial}: {'match' if trial_ok else 'mismatch'}"
    if not trial_ok and compare.get("errors"):
        err0 = str(compare["errors"][0])
        if len(err0) > 220:
            err0 = err0[:217] + "..."
        line += f" | {err0}"
    if measure_perf:
        if ref_perf and sol_perf and speedup is not None:
            line += (
                f" | ref {ref_perf['wall_time_ms_max']:.3f} ms | "
                f"{solution_backend} {sol_perf['wall_time_ms_max']:.3f} ms | speedup {speedup:.2f}x"
            )
        else:
            line += " | perf: n/a"
    print(line)
    if ref_out.returncode != 0 or sol_out.returncode != 0:
        print(
            f"    (reference exit={ref_rc}, solution exit={sol_rc})",
            file=sys.stderr,
        )
    if not trial_ok:
        for e in compare.get("errors") or []:
            print(f"    compare: {e}", file=sys.stderr)

    if download:
        sub = f"trial_{trial}"
        t_ref = ref_dir / sub
        t_sol = sol_dir / sub
        print(f"Downloading reference ({sub}) -> {t_ref}")
        write_artifacts_flat(ref_out.artifacts, t_ref)
        print(f"Downloading {solution_backend} ({sub}) -> {t_sol}")
        write_artifacts_flat(sol_out.artifacts, t_sol)

    stop_early = not trial_ok
    if stop_early:
        trial_report["early_stop"] = "mismatch"
        if n_trials > 1:
            print(
                f"  Stopping after trial {trial + 1}/{n_trials}: tensor mismatch; "
                "further trials skipped.",
                flush=True,
            )
    return EvalTrialStepResult(trial_report=trial_report, all_ok=trial_ok, stop_early=stop_early)


# Header: print the standard Perf: summary line from in-memory job artifacts
def print_perf_from_artifacts(
    artifacts: list[tuple[str, bytes]],
    measure_perf: bool = True,
) -> None:
    """Print max/mean wall time across ranks, or a short reason perf is unavailable."""
    if not measure_perf:
        return
    perf_files = [
        (name, data)
        for name, data in artifacts
        if os.path.basename(name.replace("\\", "/")).startswith("rank_")
        and name.endswith("_perf.json")
    ]
    if perf_files:
        summary = summarize_perf_artifacts(perf_files)
        if summary:
            print(
                f"Perf:      max wall time (across ranks) {summary['wall_time_ms_max']:.3f} ms "
                f"(mean {summary['wall_time_ms_mean']:.3f} ms, ranks={summary['n_ranks']})"
            )
            return
        print("Perf:      (no rank_*_perf.json found)", file=sys.stderr)
    elif artifacts:
        print("Perf:      (no rank_*_perf.json in job outputs)", file=sys.stderr)
    else:
        print("Perf:      (no job outputs; cannot read rank_*_perf.json)", file=sys.stderr)


# Header: remind the user how to persist perf JSON or profiler traces when they
# ran without --download.
def print_artifact_download_hints(*, measure_perf: bool, profile: bool) -> None:
    """Print follow-up hints after skipping artifact download."""
    if measure_perf:
        print(
            "Perf JSON not saved locally. Use --download together with --measure-perf "
            "to save rank_*_perf.json under logs/."
        )
    if profile:
        print(
            "Profiler traces not saved locally. Use --download together with --profile "
            "to save traces/ under logs/."
        )


# Header: aggregate rank-level performance JSON artifacts into max/mean wall time
# and rank count, ignoring malformed or unrelated files.
def summarize_perf_artifacts(files: list[tuple[str, bytes]]) -> dict | None:
    """
    Aggregate ``rank_*_perf.json`` artifacts from ``(relative_path, bytes)`` rows.

    Returns max wall time across ranks, mean wall time, and number of rank files
    parsed. Returns ``None`` when no valid perf JSON is present.
    """
    walls: list[float] = []
    for relpath, data in files:
        base = os.path.basename(relpath.replace("\\", "/"))
        if not (base.startswith("rank_") and base.endswith("_perf.json")):
            continue
        try:
            parsed = json.loads(data.decode("utf-8"))
            walls.append(float(parsed["wall_time_ms"]))
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if not walls:
        return None
    return {
        "wall_time_ms_max": max(walls),
        "wall_time_ms_mean": sum(walls) / len(walls),
        "n_ranks": len(walls),
    }


# Header: classify stderr/stdout blobs that indicate harness or infrastructure
# failures rather than kernel-correctness failures.
def harness_environment_hint(*texts: str) -> str | None:
    """
    Return a short harness/infra failure tag when worker logs match known patterns.

    These were all found empirically! Some common harness failure modes.

    The returned string is intended for eval reports so users know whether to fix
    the runtime/container/driver rather than the submitted kernel.
    """
    blob = "\n".join(t for t in texts if t).lower()
    if not blob:
        return None
    if "nvidia driver on your system is too old" in blob:
        return (
            "host_gpu_driver_mismatch: PyTorch's bundled CUDA needs a newer NVIDIA driver than the "
            "node exposes, OR pip upgraded torch inside the image. Rebuild/deploy the Together image "
            "from this repo's Dockerfile (do not install torch/triton from PyPI on top of NGC)."
        )
    if "cuda driver version is insufficient" in blob:
        return "host_gpu_driver_mismatch: CUDA driver insufficient for this PyTorch build (harness)."
    if "found no nvidia driver" in blob or "no cuda gpus are available" in blob:
        return "no_gpu_visible: NVIDIA driver/GPU not visible in container (harness)."
    if "no cuda runtime is found" in blob and "cuda" in blob:
        return "no_cuda_runtime: CUDA runtime not usable in container (harness)."
    if "error code: 803" in blob:
        return "nvml_gpu_error (harness)."
    if "'int' object has no attribute 'index'" in blob and "init_process_group" in blob:
        return (
            "distributed_init_device_id_type: PyTorch expected device_id=torch.device for NCCL init "
            "(harness; fixed in repo init_and_finalize_backends)."
        )
    if "filenotfounderror" in blob and "/solutions_" in blob and "_cuda.py" in blob:
        return (
            "solution_file_missing_in_container: image snapshot has no current solution module; "
            "run_together now sends solution_source in the job payload (redeploy worker if needed)."
        )
    if "cuda unknown error" in blob:
        return (
            "cuda_init_unknown_error: GPU init failed in the worker (often transient or bad node). "
            "Eval runs reference and solution as separate jobs-one can fail while the other succeeds. "
            "Retry; check Together GPU allocation / CUDA_VISIBLE_DEVICES; not a kernel-vs-reference math issue."
        )
    return None
