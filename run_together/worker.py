"""
Sprocket worker for ParallelKernelBench on Together Containers.

Receives job payloads via Together's queue and runs distributed GPU kernel
evaluations using torchrun (8 processes, one per GPU).

Payload schema:
    {
        "problem_id":    "3",
        "solution_type": "triton",      # "reference" | "triton" | "cuda" | "parallelkittens"
        "m":             1024,           # rows  (default 1024)
        "n":             1024,           # cols  (default 1024)
        "dtype":         "float32",      # float32 | float16 | bfloat16 | float64
        "measure_perf":  true,           # optional: run perf measurement (default false)
        "measure_warmup_iters": 500,     # optional: warmup before timed region (default 500)
        "measure_profiling_iters": 100,  # optional: timed iters in one CUDA event pair (default 100)
        "profile":       false,          # optional: run PyTorch profiler (default false)
        "trial":           0,            # optional: RNG trial index (eval multi-trial; default 0)
        "solution_source": "<str>"       # optional: full Python source; worker writes solutions_<backend>/<stem>_<backend>.py
    }
"""

import glob
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import sprocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Jig copies project files relative to the repo root, so the container
# working directory contains scripts/, solutions_*/, reference/, utils/.
# Resolve once at import time.
_CONTAINER_ROOT = Path(os.getcwd())
_WORKER_SCRIPT = _CONTAINER_ROOT / "scripts" / "worker.py"
_SPROCKET_PID = os.getpid()


def _num_gpus() -> int:
    """World size for torchrun. Do not call torch.cuda here — parent CUDA init races torchrun children."""
    if "NUM_GPUS" in os.environ:
        return max(1, int(os.environ["NUM_GPUS"]))
    nvd = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if nvd and nvd not in ("all", "void"):
        parts = [p for p in nvd.split(",") if p.strip() and p.strip() != "void"]
        if parts:
            return len(parts)
    # Matches [tool.jig.deploy] gpu_count when env is not explicit.
    return 8


_STRAGGLER_CMDLINE_MARKERS = ("scripts/worker.py", "torchrun", "torch.distributed.run", "elastic_agent")


def _kill_straggler_processes() -> list[int]:
    """Kill all child/orphan processes related to torchrun, returning PIDs that were signalled."""
    killed: list[int] = []

    # 1. Walk /proc to kill any process whose cmdline matches known torchrun/worker markers.
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == _SPROCKET_PID or pid <= 2:
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
            except (OSError, PermissionError):
                continue
            if any(marker in cmdline for marker in _STRAGGLER_CMDLINE_MARKERS):
                logger.warning("Killing straggler pid=%d", pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except OSError:
                    pass
    except Exception as exc:
        logger.debug("Straggler /proc scan skipped: %s", exc)

    # 2. Kill any process still holding a GPU compute context (via nvidia-smi).
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            for line in smi.stdout.strip().splitlines():
                line = line.strip()
                if not line.isdigit():
                    continue
                gpu_pid = int(line)
                if gpu_pid == _SPROCKET_PID or gpu_pid <= 2:
                    continue
                if gpu_pid not in killed:
                    logger.warning("Killing GPU-holding pid=%d (nvidia-smi)", gpu_pid)
                    try:
                        os.kill(gpu_pid, signal.SIGKILL)
                        killed.append(gpu_pid)
                    except OSError:
                        pass
    except Exception as exc:
        logger.debug("nvidia-smi compute-apps query skipped: %s", exc)

    # 3. Wait for killed processes to actually exit (up to 2s).
    if killed:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            still_alive = [p for p in killed if Path(f"/proc/{p}").exists()]
            if not still_alive:
                break
            time.sleep(0.1)
        remaining = [p for p in killed if Path(f"/proc/{p}").exists()]
        if remaining:
            logger.warning("Processes still alive after SIGKILL wait: %s", remaining)

    return killed


def _remove_stale_files() -> None:
    """Remove shared-memory, rendezvous, IPC, and temp files left by previous torchrun jobs."""
    patterns = [
        "/dev/shm/nccl-*",
        "/dev/shm/cuda*",
        "/tmp/torch-distributed-*",
        "/tmp/c10d_*",
        "/tmp/pytorch_*",
        "/tmp/torchelastic_*",
    ]
    for pattern in patterns:
        for p in glob.glob(pattern):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Purge per-rank JIT caches to avoid corrupt .so files across runs.
    ext_base = os.environ.get(
        "TORCH_EXTENSIONS_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
    )
    for p in glob.glob(os.path.join(ext_base, "pkb_rank_*")):
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _gpu_health_check() -> bool:
    """Verify GPUs are responsive via nvidia-smi. Returns True if healthy."""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if smi.returncode != 0:
            logger.error("GPU health check FAILED: nvidia-smi rc=%d stderr=%s",
                         smi.returncode, (smi.stderr or "")[:500])
            return False
        n = sum(1 for ln in smi.stdout.splitlines() if ln.strip().startswith("GPU "))
        expected = _num_gpus()
        if n < expected:
            logger.warning("GPU health check: only %d/%d GPUs visible", n, expected)
            return False
        logger.info("GPU health check OK: %d GPU(s) visible", n)
        return True
    except Exception as exc:
        logger.error("GPU health check skipped: %s", exc)
        return False


def _cleanup_between_jobs() -> None:
    """Full GPU + process + filesystem reset between torchrun invocations."""
    _kill_straggler_processes()
    _remove_stale_files()
    # Give the kernel time to reclaim TCP sockets (TIME_WAIT), GPU contexts, and shm segments.
    time.sleep(2)
    _gpu_health_check()


class PKBWorker(sprocket.Sprocket):
    """Distributed kernel evaluation worker."""

    def setup(self) -> None:
        import torch

        # Log build-time CUDA version only — torch.cuda.* initializes the driver in *this* process
        # and can leave the GPU driver in a bad state for the next subprocess.run(torchrun ...),
        # causing intermittent "no cuda runtime" / solution job failures after a successful reference.
        logger.info("torch=%s  cuda_build=%s", torch.__version__, torch.version.cuda)
        try:
            smi = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if smi.returncode == 0 and smi.stdout:
                n = sum(1 for ln in smi.stdout.splitlines() if ln.strip().startswith("GPU "))
                logger.info("nvidia-smi -L: %d GPU(s) visible in container", n)
            else:
                logger.warning(
                    "nvidia-smi -L failed rc=%s stderr=%s",
                    smi.returncode,
                    (smi.stderr or "")[:500],
                )
        except Exception as exc:
            logger.warning("nvidia-smi -L skipped: %s", exc)
        logger.info("NVSHMEM_HOME=%s", os.environ.get("NVSHMEM_HOME"))
        logger.info("CUDA_HOME=%s", os.environ.get("CUDA_HOME"))

        # Sanity-check that the worker script is reachable
        if not _WORKER_SCRIPT.exists():
            logger.warning("Worker script not found at %s – listing cwd:", _WORKER_SCRIPT)
            for p in sorted(_CONTAINER_ROOT.iterdir()):
                logger.warning("  %s", p)

    def predict(self, args: dict) -> dict:
        """Run one eval job; never raise — return ``status: error`` so Sprocket can complete the message."""
        try:
            return self._predict_once(args)
        except Exception as exc:
            logger.exception("PKBWorker.predict crashed (structured error return): %s", exc)
            try:
                _cleanup_between_jobs()
            except Exception as cleanup_exc:
                logger.warning("Post-crash cleanup failed: %s", cleanup_exc)
            tb = traceback.format_exc()
            return {
                "status": "error",
                "returncode": -1,
                "stdout": "",
                "stderr": f"PKBWorker.predict exception: {exc}\n\n{tb}",
                "elastic_error_files": None,
            }

    def _predict_once(self, args: dict) -> dict:
        # Ensure a pristine GPU/process environment before every job.
        _cleanup_between_jobs()

        # -----------------------------------------------------------------
        # Parse payload
        # -----------------------------------------------------------------
        problem_id = str(args.get("problem_id", "1"))
        # Integer ID for scripts/worker.py (input tensor shape); from payload or leading digits of stem
        problem_id_int = args.get("problem_id_int")
        if problem_id_int is None:
            m = re.match(r"^(\d+)", problem_id)
            problem_id_int = int(m.group(1)) if m else 1
        else:
            problem_id_int = int(problem_id_int)
        solution_type = str(args.get("solution_type", "reference"))
        m = int(args.get("m", 1024))
        n = int(args.get("n", 1024))
        dtype = str(args.get("dtype", "float32"))
        measure_perf = bool(args.get("measure_perf", False))
        measure_warmup_iters = int(args.get("measure_warmup_iters", 500))
        measure_profiling_iters = int(args.get("measure_profiling_iters", 100))
        profile = bool(args.get("profile", False))
        trial = int(args.get("trial", 0))

        # Resolve solution file path
        if solution_type == "reference":
            problem_py = str(_CONTAINER_ROOT / "reference" / f"{problem_id}.py")
        else:
            problem_py = str(
                _CONTAINER_ROOT / f"solutions_{solution_type}" / f"{problem_id}_{solution_type}.py"
            )

        sol_inline = args.get("solution_source")
        if sol_inline and solution_type != "reference":
            dest = Path(problem_py)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(sol_inline), encoding="utf-8")

        # Temp dir for output .pt files (cleaned up automatically)
        logs_dir = tempfile.mkdtemp(prefix=f"pkb_p{problem_id}_{solution_type}_t{trial}_")

        logger.info(
            "Running problem=%s  solution=%s  trial=%d  shape=(%d,%d)  dtype=%s  perf=%s "
            "warmup=%d profile_iters=%d  profile=%s",
            problem_id,
            solution_type,
            trial,
            m,
            n,
            dtype,
            measure_perf,
            measure_warmup_iters,
            measure_profiling_iters,
            profile,
        )
        logger.info("Solution file: %s", problem_py)
        logger.info("Logs dir:      %s", logs_dir)

        # -----------------------------------------------------------------
        # Launch torchrun (mirrors run_modal.py)
        # -----------------------------------------------------------------
        num_gpus = _num_gpus()
        master_port = random.randint(29500, 39500)
        logger.info("Using nproc-per-node=%d  master-port=%d", num_gpus, master_port)
        cmd = [
            "torchrun",
            "--nproc-per-node", str(num_gpus),
            "--master-addr", "127.0.0.1",
            "--master-port", str(master_port),
            str(_WORKER_SCRIPT),
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

        # TORCHELASTIC_ERROR_FILE: child processes write tracebacks here when they crash
        error_file = Path(logs_dir) / "torch_elastic_error"
        env = {**os.environ, "TORCHELASTIC_ERROR_FILE": str(error_file)}

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        _cleanup_between_jobs()

        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.info("STDERR:\n%s", result.stderr)

        if result.returncode != 0:
            # Include elastic error files (contain actual Python tracebacks)
            error_details = []
            for f in sorted(Path(logs_dir).glob("torch_elastic_error*")):
                try:
                    error_details.append(f"{f.name}:\n{f.read_text()}")
                except Exception as e:
                    error_details.append(f"{f.name}: (read failed: {e})")

            # Child traceback is printed FIRST; elastic wrapper prints LAST. Return both.
            stderr_raw = result.stderr or ""
            stderr_first = stderr_raw[:32000]  # actual Python traceback
            stderr_last = stderr_raw[-8000:] if len(stderr_raw) > 8000 else ""
            if len(stderr_raw) > 40000:
                stderr_combined = f"{stderr_first}\n\n... [middle {len(stderr_raw)-40000} chars omitted] ...\n\n{stderr_last}"
            else:
                stderr_combined = stderr_raw

            return {
                "status": "error",
                "returncode": result.returncode,
                "stdout": (result.stdout or "")[-8000:],
                "stderr": stderr_combined,
                "elastic_error_files": "\n\n---\n\n".join(error_details) if error_details else None,
            }

        # -----------------------------------------------------------------
        # Collect outputs and return as FileOutput
        # Walk subdirectories so traces/ .json and .gz files are included.
        # -----------------------------------------------------------------
        outputs = {}
        logs_path = Path(logs_dir)
        for root, _dirs, fnames in sorted(os.walk(logs_path)):
            for fname in sorted(fnames):
                if fname.endswith((".pt", ".json", ".gz")):
                    full = Path(root) / fname
                    rel = full.relative_to(logs_path)
                    outputs[str(rel)] = sprocket.FileOutput(str(full))

        return {
            "status": "ok",
            "problem_id": problem_id,
            "solution_type": solution_type,
            "shape": [m, n],
            "dtype": dtype,
            "num_output_files": len(outputs),
            **outputs,
        }

    def shutdown(self) -> None:
        logger.info("Shutting down PKBWorker")


if __name__ == "__main__":
    deployment_name = os.environ.get("TOGETHER_DEPLOYMENT_NAME", "pkb-cuda-nvshmem")
    sprocket.run(PKBWorker(), deployment_name)
