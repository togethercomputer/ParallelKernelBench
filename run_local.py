#!/usr/bin/env python3
"""
Local launcher for distributed kernel evaluation.

Use ``--download`` to keep a per-problem copy under ``logs/problem_<stem>/``).

Modes:
  dryrun  Run one job on a backend (e.g. reference, triton, cuda, parallelkittens, ...)
  eval    Run reference, then the given --solution, for each of --trials RNG trials
          (default 5); compare rank_*.pt outputs locally. Stops after the first trial when outputs
          differ (mismatch), or when either job produces no rank outputs (compile/runtime/harness). 
          
Requires: PyTorch + CUDA stack suitable for your node; ``torchrun`` on PATH.

Environment:
  PKB_PERF_THERMAL_COOLDOWN_SEC  Idle between ref and solution when measuring perf (default 0.5).

Example Usage (--nproc-per-node / --num-gpus is required):
    python run_local.py --nproc-per-node 4 --mode dryrun --problem 1 --solution reference --download --measure-perf
    python run_local.py --nproc-per-node 4 --mode dryrun --problem 1 --solution cuda --download --measure-perf --solutions-dir /home/simon/willyc/learning/ParallelKernelBench/solutions_cuda_bf16_h100_8_google_gemini-3-pro-preview
    python run_local.py --nproc-per-node 4 --mode dryrun --problem 1 --solution parallelkittens --download --measure-perf --solutions-dir /home/simon/willyc/learning/ParallelKernelBench/solutions_parallelkittens_bf16_h100_8_google_gemini-3-pro-preview
    python run_local.py --nproc-per-node 4 --mode dryrun --problem 1 --solution triton --download --measure-perf --solutions-dir /home/simon/willyc/learning/ParallelKernelBench/solutions_triton_bf16_h100_8_google_gemini-3-pro-preview

    python run_local.py --nproc-per-node 4 --mode eval --problem 1 --solution reference --solutions-dir /home/simon/willyc/learning/ParallelKernelBench/reference
    python run_local.py --nproc-per-node 4 --mode eval --problem 1 --solution cuda --solutions-dir /home/simon/willyc/learning/ParallelKernelBench/solutions_cuda_bf16_h100_8_google_gemini-3-pro-preview
    python run_local.py --nproc-per-node 4 --mode eval --problem 1 --solution parallelkittens --solutions-dir /path/to/kernels
    python run_local.py --nproc-per-node 4 --mode eval --problem 1 --solution triton --solutions-dir /path/to/kernels

"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from utils.problem_id import resolve_problem
from utils.run_utils import (
    check_solution,
    log_and_print_trial_results,
    print_artifact_download_hints,
    print_job_failure_diagnostics,
    print_perf_from_artifacts,
    read_solution_source,
    solution_path,
    write_artifacts_flat,
)

# Global constants for experimentation
_PERF_THERMAL_COOLDOWN_SEC = float(os.environ.get("PKB_PERF_THERMAL_COOLDOWN_SEC", "0.5"))
_DEFAULT_MEASURE_WARMUP_ITERS = 500
_DEFAULT_MEASURE_PROFILING_ITERS = 100


@dataclass
class LocalJobOutcome:
    returncode: int
    stdout: str
    stderr: str
    artifacts: list[tuple[str, bytes]]
    elastic_error_files: str | None
    worker_logs_dir: str | None = None  # preserved on disk when torchrun exits non-zero


def run_local_job(
    project_root: Path,
    *,
    problem_stem: str,
    solution_type: str,
    problem_id_int: int,
    m: int,
    n: int,
    dtype: str,
    measure_perf: bool,
    profile: bool,
    trial: int,
    measure_warmup_iters: int = _DEFAULT_MEASURE_WARMUP_ITERS,
    measure_profiling_iters: int = _DEFAULT_MEASURE_PROFILING_ITERS,
    solution_source: str | None,
    solutions_dir: Path | None,
    nproc_per_node: int,
    submit_label: str = "job",
) -> LocalJobOutcome:
    """
    Run one torchrun job given a particular PyTorch solution file.

    Essentially does `torchrun --nproc-per-node <NUM_GPUS> --master-addr 127.0.0.1 --master-port … scripts/worker.py --backend cuda --problem_py … --logs_dir …`
    """
    dest = solution_path(
        project_root, problem_stem, solution_type, solutions_dir=solutions_dir
    )
    if solution_source and solution_type != "reference":
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(solution_source, encoding="utf-8")
    problem_py = str(dest)

    # Staging for worker --logs_dir: under project logs/ (not /tmp) so paths are local; still removed
    # after artifacts are read into memory unless the job failed partway.
    sub = f"pkb_{problem_stem}_{solution_type}_t{trial}_{random.getrandbits(32):08x}"
    work_root = project_root / "logs" / "pkb_worker"
    work_root.mkdir(parents=True, exist_ok=True)
    logs_path = work_root / sub
    logs_path.mkdir(parents=True, exist_ok=True)
    logs_dir = str(logs_path)

    worker_script = project_root / "scripts" / "worker.py"

    num_gpus = max(1, int(nproc_per_node))
    master_port = random.randint(29500, 39500)

    cmd = [
        "torchrun",
        "--nproc-per-node",
        str(num_gpus),
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(master_port),
        str(worker_script),
        "--backend",
        solution_type,
        "--problem_py",
        problem_py,
        "--logs_dir",
        logs_dir,
        "--rows",
        str(m),
        "--cols",
        str(n),
        "--dtype",
        dtype,
        "--problem_id",
        str(problem_id_int),
    ]
    if measure_perf:
        cmd.append("--measure_perf")
        cmd.extend(["--measure-warmup-iters", str(measure_warmup_iters)])
        cmd.extend(["--measure-profiling-iters", str(measure_profiling_iters)])
    if profile:
        cmd.append("--profile")
    cmd.extend(["--trial", str(trial)])

    error_file = logs_path / "torch_elastic_error"
    env = {**os.environ, "TORCHELASTIC_ERROR_FILE": str(error_file)}

    print(f"Submitting {submit_label} ({solution_type})...")
    if os.environ.get("PKB_LOCAL_VERBOSE"):
        print(
            f"  (local) torchrun nproc={num_gpus} port={master_port} problem_py={problem_py} logs={logs_dir}",
            file=sys.stderr,
        )

    # Launch torchrun. We keep stdout/stderr in memory so the caller can print
    # concise diagnostics and embed bounded logs in PKB_EVAL_JSON.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )

    interrupted = False

    # Wait for torchrun to finish normally. Ctrl+C needs explicit cleanup because
    # torchrun has worker children; start_new_session=True lets us signal the
    # whole process group instead of leaving ranks alive.
    try:
        stdout, stderr = proc.communicate()
        returncode = proc.returncode if proc.returncode is not None else 1
    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interrupted; stopping torchrun...", file=sys.stderr)

        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, OSError):
                proc.terminate()

            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.wait()

        # Collect anything torchrun flushed before it died; if pipes are already
        # closed, fall back to empty strings and rely on the preserved log dir.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError):
            stdout, stderr = "", ""
        returncode = 130
        (logs_path / "interrupted.txt").write_text(
            f"Ctrl+C during {submit_label}\n"
            f"backend={solution_type} problem={problem_stem} trial={trial}\n",
            encoding="utf-8",
        )
    else:
        print("  Job completed.")

    stdout = stdout or ""
    stderr = stderr or ""

    # Torch elastic writes structured launcher failures to TORCHELASTIC_ERROR_FILE.
    # Read only that small set of files; do not walk and re-summarize the whole log tree.
    elastic_parts: list[str] = []
    for f in sorted(logs_path.glob("torch_elastic_error*")):
        try:
            elastic_parts.append(f"{f.name}:\n{f.read_text()}")
        except Exception as e:
            elastic_parts.append(f"{f.name}: (read failed: {e})")
    elastic_blob = "\n\n---\n\n".join(elastic_parts) if elastic_parts else None

    if returncode == 0 and not interrupted:
        # Successful jobs should have rank_*.pt and optional perf/profile JSON/GZ files.
        # Read those into memory, then remove the temporary worker log directory.
        artifacts: list[tuple[str, bytes]] = []
        for root, _dirs, filenames in sorted(os.walk(logs_path)):
            for filename in sorted(filenames):
                if filename.endswith((".pt", ".json", ".gz")):
                    full = Path(root) / filename
                    rel = str(full.relative_to(logs_path)).replace("\\", "/")
                    artifacts.append((rel, full.read_bytes()))
        artifacts.sort(key=lambda x: x[0])
        try:
            shutil.rmtree(logs_path, ignore_errors=True)
        except Exception:
            pass
        worker_logs_dir = None
    else:
        # Failed/interrupted jobs may contain useful partial logs. Leave them on disk
        # and point diagnostics at the directory instead of trying to package it all.
        artifacts = []
        worker_logs_dir = str(logs_path.resolve())

    return LocalJobOutcome(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        elastic_error_files=elastic_blob,
        worker_logs_dir=worker_logs_dir,
    )

def run_eval_single_problem(
    project_root: Path,
    problem_stem: str,
    problem_id_int: int,
    args: argparse.Namespace,
    logs_base: Path,
    solution_payload: str | None,
    solutions_dir: Path | None,
    solution_backend: str,
) -> tuple[bool, list[dict]]:
    """
    Run eval trials for one problem.

    Note: we stop after the FIRST trial if any of the following happen:
        - tensor compare fails (mismatch)
        - the reference or solution job fails (no artifacts)
    """
    ref_dir = logs_base / f"problem_{problem_stem}" / "reference"
    sol_dir = logs_base / f"problem_{problem_stem}" / solution_backend
    if args.download:
        if ref_dir.is_dir():
            shutil.rmtree(ref_dir)
        if sol_dir.is_dir():
            shutil.rmtree(sol_dir)

    all_ok = True
    trial_reports: list[dict] = []

    # For evaluation, we do a `torchrun reference` and `torchrun solution` pair per trial
    for trial in range(args.trials):
        ref_out = run_local_job(
            project_root,
            problem_stem=problem_stem,
            solution_type="reference",
            problem_id_int=problem_id_int,
            m=args.m,
            n=args.n,
            dtype=args.dtype,
            measure_perf=args.measure_perf,
            profile=args.profile,
            trial=trial,
            solution_source=None,
            solutions_dir=solutions_dir,
            nproc_per_node=args.nproc_per_node,
            submit_label=f"reference (trial {trial})",
        )

        if args.measure_perf and _PERF_THERMAL_COOLDOWN_SEC > 0:
            time.sleep(_PERF_THERMAL_COOLDOWN_SEC)

        sol_out = run_local_job(
            project_root,
            problem_stem=problem_stem,
            solution_type=solution_backend,
            problem_id_int=problem_id_int,
            m=args.m,
            n=args.n,
            dtype=args.dtype,
            measure_perf=args.measure_perf,
            profile=args.profile,
            trial=trial,
            solution_source=solution_payload,
            solutions_dir=solutions_dir,
            nproc_per_node=args.nproc_per_node,
            submit_label=f"solution (trial {trial})",
        )

        step = log_and_print_trial_results(
            trial=trial,
            ref_out=ref_out,
            sol_out=sol_out,
            solution_backend=solution_backend,
            measure_perf=args.measure_perf,
            atol=args.atol,
            rtol=args.rtol,
            n_trials=args.trials,
            download=args.download,
            ref_dir=ref_dir,
            sol_dir=sol_dir,
        )
        trial_reports.append(step.trial_report)
        all_ok = all_ok and step.all_ok
        if step.stop_early:
            break

    return all_ok, trial_reports


def run_local_main(
    args: argparse.Namespace,
    project_root: Path,
    logs_base: Path,
    problem_stem: str,
    problem_id_int: int,
    solution_backend: str,
    solutions_dir: Path | None,
    solution_payload: str | None,
) -> None:
    # dryrun logic
    if args.mode == "dryrun":
        out = run_local_job(
            project_root,
            problem_stem=problem_stem,
            solution_type=solution_backend,
            problem_id_int=problem_id_int,
            m=args.m,
            n=args.n,
            dtype=args.dtype,
            measure_perf=args.measure_perf,
            profile=args.profile,
            trial=0,
            solution_source=solution_payload,
            solutions_dir=solutions_dir,
            nproc_per_node=args.nproc_per_node,
            submit_label="job",
        )
        if out.returncode != 0:
            label = (
                "Interrupted (Ctrl+C)."
                if out.returncode == 130
                else "Job failed (torchrun / worker non-zero exit)."
            )
            print(label, file=sys.stderr)
            print_job_failure_diagnostics(out)
            sys.exit(130 if out.returncode == 130 else 1)
        if not out.artifacts:
            print("No output files to download.", file=sys.stderr)
            sys.exit(1)

        if args.measure_perf:
            print_perf_from_artifacts(out.artifacts)     # prints out perf information

        if not args.download:
            print("Skipping download. Pass --download to save .pt/.json under logs/.")
            print_artifact_download_hints(measure_perf=args.measure_perf, profile=args.profile)
            sys.exit(0)

        # Download problem artifacts (tensors + performance data)
        dest = logs_base / f"problem_{problem_stem}" / solution_backend
        print(f"Downloading {len(out.artifacts)} file(s) to {dest}")
        write_artifacts_flat(out.artifacts, dest)
        print(f"All files in: {dest}")
        sys.exit(0)

    # eval logic
    # Run and evaluation for a single problem:
    interrupted_run = False
    try:
        all_ok, trial_reports = run_eval_single_problem(
            project_root,
            problem_stem,
            problem_id_int,
            args,
            logs_base,
            solution_payload,
            solutions_dir,
            solution_backend,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        all_ok = False
        trial_reports = []
        interrupted_run = True
    print("all trials match" if all_ok else "mismatch")

    # We just ran evaluation for a single problem. `eval_payload` is a JSON/dict containing all the information we wish to save from said evaluation.
    eval_payload = {
        "version": 1,
        "backend": "local",
        "mode": "eval",
        "problem": problem_stem,
        "solution": solution_backend,
        "all_ok": all_ok,
        "interrupted": interrupted_run or any(isinstance(t, dict) and t.get("result") == "interrupted" for t in trial_reports),
        "harness_environment_failed": any(isinstance(t, dict) and t.get("result") == "harness_environment" for t in trial_reports),
        "trials": trial_reports,
    }
    if solutions_dir is not None:
        eval_payload["solutions_dir"] = str(solutions_dir)
    eval_payload["nproc_per_node"] = args.nproc_per_node


    eval_line = "PKB_EVAL_JSON:" + json.dumps(eval_payload, default=str, ensure_ascii=True)
    ### Optional: truncate the log if you don't want to save all of the logs
    # if len(eval_line) > 240_000:
    #     slim = {
    #         "version": 1,
    #         "backend": "local",
    #         "mode": "eval",
    #         "problem": problem_stem,
    #         "solution": solution_backend,
    #         "all_ok": all_ok,
    #         "trials": [
    #             {
    #                 "trial": t.get("trial"),
    #                 "result": t.get("result"),
    #                 "compare_errors": (t.get("compare") or {}).get("errors"),
    #                 "reference_returncode": t.get("reference_returncode"),
    #                 "solution_returncode": t.get("solution_returncode"),
    #             }
    #             for t in trial_reports
    #         ],
    #         "_truncated": "full PKB_EVAL_JSON exceeded 240k; inspect run_local stdout or use --download",
    #     }
    #     eval_line = "PKB_EVAL_JSON:" + json.dumps(slim, default=str, ensure_ascii=True)
    print(eval_line, flush=True)
    if not args.download:
        if args.measure_perf:
            print(
                "Perf JSON not saved locally. Use --download together with --measure-perf "
                "to save rank_*_perf.json under logs/."
            )
        if args.profile:
            print(
                "Profiler traces not saved locally. Use --download together with --profile "
                "to save traces/ under logs/."
            )
    if eval_payload["interrupted"]:
        sys.exit(130)
    sys.exit(0 if all_ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ParallelKernelBench locally (multi-GPU torchrun).")

    # (1) Evaluation config and problem essentials
    parser.add_argument(
        "--mode",
        choices=("dryrun", "eval"),
        default="dryrun",
        help=f"Can either be `dryrun` (run a single backend) or `eval` (run reference PyTorch & custom solution, and compare tensors and performance).",
    )
    parser.add_argument(
        "--problem",
        "-p",
        default="1",
        help="Problem: number, stem, or file.py (resolved against reference/).",
    )
    parser.add_argument(
        "--solution",
        "-s",
        default="reference",
        help="Worker backend: reference, triton, cuda, parallelkittens.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--num-gpus",
        type=int,
        required=True,
        dest="nproc_per_node",
        help="torchrun --nproc-per-node (number of GPU processes; e.g. 4 to use 4 GPUs). Required.",
    )

    # (2) Input Tensor Properties
    parser.add_argument("--m", type=int, default=1024, help="Custom Tensor Size Specified #1 (e.g. Rows). This is used for simplifying tensor sweeps.")
    parser.add_argument("--n", type=int, default=1024, help="Custom Tensor Size Specified #2 (e.g. Cols). This is used for simplifying tensor sweeps.")
    parser.add_argument("--dtype", default="float32", help="Tensor dtype")

    # (3) Correctness Evaluation Configuration
    parser.add_argument("--download", action="store_true", help="Write Tensor outputs under logs/problem_<stem>/{reference|<solution>}/ (default: do not download).")
    parser.add_argument("--atol", type=float, default=1e-2, help="eval mode: abs tolerance for tensor compare")
    parser.add_argument("--rtol", type=float, default=1e-2, help="eval mode: rel tolerance for tensor compare")
    parser.add_argument("--trials", type=int, default=5, help="eval mode only: number of random-input trials (default 5). Ignored for dryrun.")

    # (4) Performance Evaluation Configuration
    parser.add_argument("--measure-perf", action="store_true", help="Warmup + timed iterations; rank_*_perf.json")
    parser.add_argument("--profile", action="store_true", help="PyTorch profiler traces under traces/")

    # (5) Directories for logs / solutions
    parser.add_argument("--logs-dir", type=str, default=None, help="Base logs directory (default: project logs/)")
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=None,
        help="Directory of solution files named {stem}_{backend}.py (e.g. 1_allreduce_cuda.py). "
        "Overrides <project>/solutions_<backend>/.",
    )
    args = parser.parse_args()



    # Args checks
    if args.trials < 1:
        print("Error: --trials must be >= 1", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    logs_base = Path(args.logs_dir) if args.logs_dir else project_root / "logs"     # directory for logs
    solutions_dir: Path | None = (args.solutions_dir.resolve() if args.solutions_dir is not None else None)
    if solutions_dir is not None and not solutions_dir.is_dir():
        print("Error: --solutions-dir must be an existing directory", file=sys.stderr)
        sys.exit(1)
    solution_backend = args.solution



    # Find the problem in question
    try:
        problem_stem, problem_id_int = resolve_problem(
            project_root, str(args.problem), args.solution
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check that the files actually exist in the specified paths
    if args.mode == "eval":
        # (1) Eval Mode: Check both reference and backend .py files exist
        if solution_backend == "reference":
            print("Error: --mode eval requires a non-reference solution (-s triton|cuda|parallelkittens).", file=sys.stderr)
            sys.exit(1)
        check_solution(project_root, problem_stem, "reference")
        check_solution(project_root, problem_stem, solution_backend, solutions_dir=solutions_dir)
    else:
        # (2) DryRun Mode: Check just the backend .py file exists
        check_solution(project_root, problem_stem, solution_backend, solutions_dir=solutions_dir)

    # Print out the entire config so users can confirm everything looks right
    print("Local (torchrun)")
    print(f"  Mode:     {args.mode}")
    print(f"  Problem:  {problem_stem}")
    s_disp = f"{solution_backend}" + (f"  (from {solutions_dir})" if solutions_dir else "")
    print(f"  Solution: {s_disp}")
    print(f"  nproc:    {args.nproc_per_node}  (torchrun --nproc-per-node; same as --num-gpus)")
    print(f"  Shape:    ({args.m}, {args.n})  dtype={args.dtype}")
    print(f"  Download: {args.download}")
    if args.mode == "eval":
        print(f"  Trials:   {args.trials}")
    if args.measure_perf:
        print("  measure_perf: yes")
    if args.profile:
        print("  profile:      yes")
    print()

    # Read solution source code
    try:
        solution_payload = read_solution_source(
            project_root,
            problem_stem,
            solution_backend,
            solutions_dir=solutions_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the evaluation locally!
    try:
        run_local_main(
            args,
            project_root,
            logs_base,
            problem_stem,
            problem_id_int,
            solution_backend,
            solutions_dir,
            solution_payload,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()
