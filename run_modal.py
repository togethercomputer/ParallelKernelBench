#!/usr/bin/env python3
"""
Modal launcher for distributed kernel evaluation.

Modes (see --mode):
  dryrun  Run one backend (reference, triton, cuda, parallelkittens, ...). No local files unless --download.
  eval    Run reference and the given --solution, compare per-rank .pt outputs on Modal for
          --trials random-input trials (default 5). With --measure-perf, uses the repo benchmarking
          convention (500 warmup iters, 100 timed iters in one CUDA event pair, L2-sized input groups,
          500 ms idle between reference and solution). Prints per-trial wall times (max across ranks)
          and speedup. Saves logs locally only with --download; rank_*_perf.json and traces/ only with
          --download and --measure-perf / --profile respectively.

Examples:
    modal run run_modal.py --mode dryrun --problem 2
    modal run run_modal.py --mode dryrun --problem 2 --solution triton --download
    modal run run_modal.py --mode eval --problem 2 --solution triton
    modal run run_modal.py --mode eval --problem 2 --solution triton --trials 3 --download

Remote timeout: PKB_MODAL_TIMEOUT_SEC (default 300s). Eval pair runs use PKB_MODAL_EVAL_TIMEOUT_SEC
or a default scaled for multiple trials (see _MODAL_EVAL_PAIR_TIMEOUT_SEC).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import modal

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Brief idle between reference and solution GPU runs when measuring perf (thermal / power steadying).
_PERF_THERMAL_COOLDOWN_SEC = float(os.environ.get("PKB_PERF_THERMAL_COOLDOWN_SEC", "0.5"))


def _load_problem_id():
    """
    Load problem_id.py by absolute path first; fall back to normal import for local runs.

    Modal imports this module before the baked image exists: /workspace is missing then, so
    do not call this at import time — only from main() / run_eval_pair() when paths exist.
    """
    for path in (
        "/workspace/utils/problem_id.py",
        os.path.join(_script_dir, "utils", "problem_id.py"),
    ):
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("pkb_problem_id", path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.resolve_problem, mod.outputs_match_rank_outputs
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, _script_dir)
    from utils.problem_id import resolve_problem, outputs_match_rank_outputs

    return resolve_problem, outputs_match_rank_outputs


app = modal.App("parallel-kernel-bench")

_MODAL_REMOTE_TIMEOUT_SEC = int(os.environ.get("PKB_MODAL_TIMEOUT_SEC", str(60 * 5)))
_MODAL_PAIR_TIMEOUT_SEC = max(_MODAL_REMOTE_TIMEOUT_SEC * 2, 30 * 60)
# Eval runs reference+solution per trial; allow enough wall time for several trials (override via PKB_MODAL_EVAL_TIMEOUT_SEC).
_MODAL_EVAL_PAIR_TIMEOUT_SEC = int(
    os.environ.get("PKB_MODAL_EVAL_TIMEOUT_SEC", str(max(_MODAL_REMOTE_TIMEOUT_SEC * 20, 45 * 60)))
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "wget", "xz-utils", "gnupg", "software-properties-common", "git",
    )
    # Install NVSHMEM from tarball (avoids version mismatch issues with apt package)
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz -O /tmp/nvshmem.tar.xz",
        "mkdir -p /opt/nvshmem",
        "tar -xf /tmp/nvshmem.tar.xz -C /opt/nvshmem --strip-components=1",
        "rm /tmp/nvshmem.tar.xz",
    )
    # environment variables needed for using NVSHMEM
    .env({
        "NVSHMEM_HOME": "/opt/nvshmem",
        "LD_LIBRARY_PATH": "/opt/nvshmem/lib:/usr/local/cuda/lib64",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:${PATH}",
    })
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "mpi4py",
        "nvshmem4py-cu12",
        "cuda-python>=12.0",
        "cffi",
        "ninja",  # required for torch's load_inline
    )
    # Clone ThunderKittens (parallelkittens backend headers)
    .run_commands(
        "git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens",
    )
    .env({"THUNDERKITTENS_ROOT": "/opt/thunderkittens"})
)

# Volume to persist logs across runs (optional, for debugging)
volume = modal.Volume.from_name("pkb-logs", create_if_missing=True)

# Add local project files to the image; PYTHONPATH helps any Python subprocess resolve `utils`.
project_dir = os.path.dirname(os.path.abspath(__file__))
image_with_code = image.add_local_dir(
    project_dir, remote_path="/workspace", copy=True
).env({"PYTHONPATH": "/workspace"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def execute_worker(
    solution_type,
    problem_id,
    m,
    n,
    dtype,
    measure_perf,
    profile=False,
    problem_id_int=None,
    trial: int = 0,
    separate_trial_logs: bool = False,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
):
    """Run torchrun for one backend. Returns (logs_dir, exit_code)."""
    import subprocess

    if problem_id_int is None:
        m_match = re.match(r"^(\d+)", problem_id)
        problem_id_int = int(m_match.group(1)) if m_match else 1

    if solution_type == "reference":
        problem_py = f"/workspace/reference/{problem_id}.py"
    else:
        problem_py = f"/workspace/solutions_{solution_type}/{problem_id}_{solution_type}.py"

    if separate_trial_logs:
        logs_dir = f"/logs/problem_{problem_id}/{solution_type}/trial_{trial}"
    else:
        logs_dir = f"/logs/problem_{problem_id}/{solution_type}"

    # Clear stale logs from previous runs
    if os.path.isdir(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Running: {problem_py}")
    print(f"Output:  {logs_dir}")
    print(f"Shape:   ({m}, {n}), dtype={dtype}")
    print(f"GPUs:    8")
    print(f"Backend: {solution_type}")
    print("-" * 60)

    worker_script_path = "/workspace/scripts/worker.py"
    cmd = [
        "torchrun",
        "--nproc-per-node", "8",
        "--master-addr", "127.0.0.1",
        "--master-port", "29500",
        worker_script_path,
        "--backend", solution_type,
        "--problem_py", problem_py,
        "--logs_dir", logs_dir,
        "--rows", str(m),
        "--cols", str(n),
        "--dtype", dtype,
        "--problem_id", str(problem_id_int),
    ]
    if measure_perf:
        cmd.append("--measure_perf")
        cmd.extend(["--measure-warmup-iters", str(measure_warmup_iters)])
        cmd.extend(["--measure-profiling-iters", str(measure_profiling_iters)])
    if profile:
        cmd.append("--profile")
    cmd.extend(["--trial", str(trial)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Worker exited with code {result.returncode}")
    elif result.stderr:
        print("STDERR:", result.stderr)

    return logs_dir, result.returncode


def _is_perf_metrics_json(relpath: str) -> bool:
    """rank_*_perf.json from utils.performance.measure_solution_performance."""
    base = os.path.basename(relpath)
    return base.startswith("rank_") and base.endswith("_perf.json")


def _is_profile_artifact(relpath: str) -> bool:
    """Chrome trace JSON under traces/ from torch.profiler."""
    return relpath.startswith("traces" + os.sep) or "/traces/" in relpath.replace("\\", "/")


def collect_files(
    logs_dir,
    include_perf_artifacts: bool = True,
    include_profile_artifacts: bool = True,
):
    """(relpath, bytes) for .pt / .json / .gz under logs_dir, with optional perf/profile filtering."""
    files = []
    if not os.path.isdir(logs_dir):
        return files
    for root, _dirs, fnames in sorted(os.walk(logs_dir)):
        for fname in sorted(fnames):
            if not fname.endswith((".pt", ".json", ".gz")):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, logs_dir)
            if not include_perf_artifacts and _is_perf_metrics_json(rel):
                continue
            if not include_profile_artifacts and _is_profile_artifact(rel):
                continue
            with open(full, "rb") as f:
                files.append((rel, f.read()))
    return files


def summarize_perf_logs_dir(logs_dir: str) -> dict | None:
    """
    Aggregate rank_*_perf.json under logs_dir (Modal container or local).
    Uses max wall_time_ms across ranks as the conservative collective time for this GPU.
    """
    if not os.path.isdir(logs_dir):
        return None
    walls = []
    for root, _dirs, fnames in os.walk(logs_dir):
        for fname in fnames:
            if fname.startswith("rank_") and fname.endswith("_perf.json"):
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                    walls.append(float(d["wall_time_ms"]))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
    if not walls:
        return None
    return {
        "wall_time_ms_max": max(walls),
        "wall_time_ms_mean": sum(walls) / len(walls),
        "n_ranks": len(walls),
    }


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    image=image_with_code,
    gpu="H100:8",
    timeout=_MODAL_REMOTE_TIMEOUT_SEC,
    volumes={"/logs": volume},
)
def run_distributed_eval(
    problem_id: str = "1",
    solution_type: str = "reference",
    m: int = 1024,
    n: int = 1024,
    dtype: str = "bfloat16",
    measure_perf: bool = False,
    profile: bool = False,
    problem_id_int: int = 1,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
) -> dict:
    logs_dir, rc = execute_worker(
        solution_type,
        problem_id,
        m,
        n,
        dtype,
        measure_perf,
        profile,
        problem_id_int=problem_id_int,
        measure_warmup_iters=measure_warmup_iters,
        measure_profiling_iters=measure_profiling_iters,
    )
    volume.commit()
    perf_summary = None
    if measure_perf and rc == 0:
        perf_summary = summarize_perf_logs_dir(logs_dir)
    return {
        "problem_id": problem_id,
        "solution_type": solution_type,
        "logs_dir": logs_dir,
        "worker_exit_code": rc,
        "perf_summary": perf_summary,
    }


@app.function(
    image=image_with_code,
    gpu="H100:8",
    timeout=_MODAL_EVAL_PAIR_TIMEOUT_SEC,
    volumes={"/logs": volume},
)
def run_eval_pair(
    problem_id: str = "1",
    solution_type: str = "triton",
    m: int = 1024,
    n: int = 1024,
    dtype: str = "bfloat16",
    measure_perf: bool = False,
    profile: bool = False,
    problem_id_int: int = 1,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    include_files: bool = False,
    include_perf_in_download: bool = False,
    include_profile_in_download: bool = False,
    trials: int = 5,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
) -> dict:
    """
    Run reference then solution for each trial index; compare .pt outputs on the container.
    If include_files is True, return raw file bytes for local saving (with --download),
    with relpaths prefixed trial_<k>/...
    Perf JSON and profiler traces are only included when the corresponding *_in_download flags are True.
    """
    _, outputs_match_rank_outputs = _load_problem_id()
    trial_results = []
    ref_files_all = []
    sol_files_all = []
    all_ok = True

    for trial in range(trials):
        ref_dir, ref_rc = execute_worker(
            "reference",
            problem_id,
            m,
            n,
            dtype,
            measure_perf,
            profile,
            problem_id_int=problem_id_int,
            trial=trial,
            separate_trial_logs=True,
            measure_warmup_iters=measure_warmup_iters,
            measure_profiling_iters=measure_profiling_iters,
        )
        if measure_perf and _PERF_THERMAL_COOLDOWN_SEC > 0:
            time.sleep(_PERF_THERMAL_COOLDOWN_SEC)
        sol_dir, sol_rc = execute_worker(
            solution_type,
            problem_id,
            m,
            n,
            dtype,
            measure_perf,
            profile,
            problem_id_int=problem_id_int,
            trial=trial,
            separate_trial_logs=True,
            measure_warmup_iters=measure_warmup_iters,
            measure_profiling_iters=measure_profiling_iters,
        )

        ref_files = collect_files(ref_dir)
        sol_files = collect_files(sol_dir)
        compared_ok = (
            ref_rc == 0
            and sol_rc == 0
            and outputs_match_rank_outputs(ref_files, sol_files, atol, rtol)
        )
        ref_perf = summarize_perf_logs_dir(ref_dir) if measure_perf and ref_rc == 0 else None
        sol_perf = summarize_perf_logs_dir(sol_dir) if measure_perf and sol_rc == 0 else None
        speedup = None
        if (
            ref_perf is not None
            and sol_perf is not None
            and sol_perf["wall_time_ms_max"] > 0
        ):
            speedup = ref_perf["wall_time_ms_max"] / sol_perf["wall_time_ms_max"]

        trial_results.append(
            {
                "trial": trial,
                "tensors_match": compared_ok,
                "reference_exit_code": ref_rc,
                "solution_exit_code": sol_rc,
                "reference_perf": ref_perf,
                "solution_perf": sol_perf,
                "speedup": speedup,
            }
        )
        if not compared_ok:
            all_ok = False
        if include_files:
            prefix = f"trial_{trial}"
            ref_dl = collect_files(
                ref_dir,
                include_perf_artifacts=include_perf_in_download,
                include_profile_artifacts=include_profile_in_download,
            )
            sol_dl = collect_files(
                sol_dir,
                include_perf_artifacts=include_perf_in_download,
                include_profile_artifacts=include_profile_in_download,
            )
            ref_files_all.extend([(f"{prefix}/{r}", b) for r, b in ref_dl])
            sol_files_all.extend([(f"{prefix}/{r}", b) for r, b in sol_dl])

    volume.commit()

    out = {
        "problem_id": problem_id,
        "solution_type": solution_type,
        "trials": trials,
        "tensors_match": all_ok,
        "trial_results": trial_results,
        "reference_files": ref_files_all if include_files else [],
        "solution_files": sol_files_all if include_files else [],
    }
    return out


@app.function(
    image=image,
    volumes={"/logs": volume},
)
def download_logs(
    problem_id: str,
    solution_type: str,
    include_perf_artifacts: bool = True,
    include_profile_artifacts: bool = True,
) -> list:
    """Read /logs from the volume (cheap CPU container)."""
    logs_dir = f"/logs/problem_{problem_id}/{solution_type}"
    if not os.path.isdir(logs_dir):
        print(f"No logs found at {logs_dir}")
        return []
    return collect_files(
        logs_dir,
        include_perf_artifacts=include_perf_artifacts,
        include_profile_artifacts=include_profile_artifacts,
    )


def save_files_locally(files, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for relpath, data in files:
        local_path = os.path.join(local_dir, relpath)
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"  Downloaded: {local_path}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    mode: str = "dryrun",
    problem: str = "1",
    solution: str = "reference",
    m: int = 1024,
    n: int = 1024,
    dtype: str = "bfloat16",
    download: bool = False,
    measure_perf: bool = False,
    profile: bool = False,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    trials: int = 5,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
):
    project_root = os.path.dirname(os.path.abspath(__file__))
    resolve_problem, _ = _load_problem_id()
    if mode not in ("dryrun", "eval"):
        print(f"Error: --mode must be 'dryrun' or 'eval' (got {mode!r})", file=sys.stderr)
        sys.exit(1)

    try:
        problem_stem, problem_id_int = resolve_problem(Path(project_root), problem, solution)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if mode == "eval":
        if solution == "reference":
            print("Error: --mode eval needs a solution other than 'reference' to compare against reference.", file=sys.stderr)
            sys.exit(1)
        if trials < 1:
            print("Error: --trials must be >= 1", file=sys.stderr)
            sys.exit(1)

        include_perf_in_download = download and measure_perf
        include_profile_in_download = download and profile

        result = run_eval_pair.remote(
            problem_stem,
            solution,
            m,
            n,
            dtype,
            measure_perf,
            profile,
            problem_id_int=problem_id_int,
            atol=atol,
            rtol=rtol,
            include_files=download,
            include_perf_in_download=include_perf_in_download,
            include_profile_in_download=include_profile_in_download,
            trials=trials,
            measure_warmup_iters=measure_warmup_iters,
            measure_profiling_iters=measure_profiling_iters,
        )

        ok = result["tensors_match"]
        for tr in result["trial_results"]:
            t_ok = tr["tensors_match"]
            line = f"  trial {tr['trial']}: {'match' if t_ok else 'mismatch'}"
            if measure_perf:
                rp = tr.get("reference_perf")
                sp = tr.get("solution_perf")
                su = tr.get("speedup")
                if rp is not None and sp is not None and su is not None:
                    line += (
                        f" | ref {rp['wall_time_ms_max']:.3f} ms | "
                        f"{solution} {sp['wall_time_ms_max']:.3f} ms | speedup {su:.2f}x"
                    )
                else:
                    line += " | perf: n/a"
            print(line)
            if tr["reference_exit_code"] != 0 or tr["solution_exit_code"] != 0:
                print(
                    f"    (reference exit={tr['reference_exit_code']}, "
                    f"solution exit={tr['solution_exit_code']})",
                    file=sys.stderr,
                )
        print("all trials match" if ok else "mismatch")

        if download:
            ref_local = os.path.join(project_root, "logs", f"problem_{problem_stem}", "reference")
            sol_local = os.path.join(project_root, "logs", f"problem_{problem_stem}", solution)
            for d in (ref_local, sol_local):
                if os.path.isdir(d):
                    shutil.rmtree(d)
            print("Downloading reference and solution artifacts...")
            save_files_locally(result["reference_files"], ref_local)
            save_files_locally(result["solution_files"], sol_local)
            print(f"Wrote reference -> {ref_local}")
            print(f"Wrote {solution} -> {sol_local}")
        elif measure_perf or profile:
            if measure_perf and not download:
                print(
                    "Perf JSON not saved locally. Use --download together with --measure-perf "
                    "to save rank_*_perf.json under logs/."
                )
            if profile and not download:
                print(
                    "Profiler traces not saved locally. Use --download together with --profile "
                    "to save traces/ under logs/."
                )

        sys.exit(0 if ok else 1)

    # dryrun
    result = run_distributed_eval.remote(
        problem_id=problem_stem,
        solution_type=solution,
        m=m,
        n=n,
        dtype=dtype,
        measure_perf=measure_perf,
        profile=profile,
        problem_id_int=problem_id_int,
        measure_warmup_iters=measure_warmup_iters,
        measure_profiling_iters=measure_profiling_iters,
    )

    print(f"Problem:   {problem_stem}")
    print(f"Backend:   {solution}")
    print(f"Logs dir:  {result['logs_dir']}")
    print(f"Worker:    exit code {result['worker_exit_code']}")
    if measure_perf:
        ps = result.get("perf_summary")
        if ps:
            print(
                f"Perf:      max wall time (across ranks) {ps['wall_time_ms_max']:.3f} ms "
                f"(mean {ps['wall_time_ms_mean']:.3f} ms, ranks={ps['n_ranks']})"
            )
        elif result["worker_exit_code"] == 0:
            print("Perf:      (no rank_*_perf.json found)", file=sys.stderr)

    if download:
        local_logs_dir = os.path.join(
            project_root,
            "logs",
            f"problem_{problem_stem}",
            solution,
        )
        if os.path.isdir(local_logs_dir):
            shutil.rmtree(local_logs_dir)
        print("Downloading artifacts...")
        files = download_logs.remote(
            problem_stem,
            solution,
            include_perf_artifacts=download and measure_perf,
            include_profile_artifacts=download and profile,
        )
        save_files_locally(files, local_logs_dir)
        print(f"Wrote -> {local_logs_dir}")
    elif measure_perf:
        print("Skipping download. Pass --download to save .pt, rank_*_perf.json, traces/ under logs/.")

    sys.exit(0 if result["worker_exit_code"] == 0 else 1)
