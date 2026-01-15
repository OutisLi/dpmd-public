# SPDX-License-Identifier: LGPL-3.0-or-later
"""Schedule multi-GPU DPA4C capacity scans without GPU contention.

Each scan is executed by :mod:`lmp_scan`, which owns the logarithmic capacity
search and the 500,000-atom bisection. This module only allocates idle GPUs and
starts disjoint MPI jobs. Results are separated by process count:

``results/multigpu/ranks2/``, ``results/multigpu/ranks4/`` and
``results/multigpu/ranks8/``.

The scheduler runs one process-count batch at a time. Within a batch it starts
as many disjoint jobs as the available GPUs permit, so eight GPUs run four
two-rank models, two four-rank models, or one eight-rank model concurrently.
"""

from __future__ import (
    annotations,
)

import os
import subprocess
import time
from collections import (
    deque,
)
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)

HERE = Path(__file__).resolve().parent
PYTHON = Path("/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/python")
MODEL_ROOT = HERE / "models" / "sweep"
RESULT_ROOT = HERE / "results" / "multigpu"
WORK_ROOT = HERE / "work_multigpu"
LMP_SCAN = HERE / "lmp_scan.py"
MPIEXEC = Path(
    os.environ.get(
        "BENCH_MPIEXEC",
        "/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/mpiexec.gforker",
    )
)
GPU_COUNT = int(os.environ.get("BENCH_GPU_COUNT", "8"))
GPU_MEMORY_LIMIT_MIB = int(os.environ.get("BENCH_GPU_MEMORY_LIMIT_MIB", "512"))
POLL_SECONDS = int(os.environ.get("BENCH_GPU_POLL_SECONDS", "15"))
GRADES = ("nano", "mini", "neo", "air", "plus")
PROCESS_COUNTS = (2, 4, 8)


@dataclass
class ActiveJob:
    """A running scan and the GPUs reserved for it."""

    process: subprocess.Popen[bytes]
    gpus: tuple[int, ...]
    tag: str
    log_path: Path


def idle_gpus(reserved: set[int]) -> list[int]:
    """Return unreserved GPUs whose allocated memory is below the threshold."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    available = []
    for line in output.splitlines():
        index_text, memory_text = (field.strip() for field in line.split(",", 1))
        index = int(index_text)
        memory = int(memory_text)
        if index not in reserved and memory <= GPU_MEMORY_LIMIT_MIB:
            available.append(index)
    return sorted(available)


def start_scan(grade: str, nprocs: int, gpus: tuple[int, ...]) -> ActiveJob:
    """Start one disjoint MPI scan and return its process handle."""
    tag = f"dpa4c_{grade}_mpi{nprocs}"
    result_dir = RESULT_ROOT / f"ranks{nprocs}"
    work_root = WORK_ROOT / f"ranks{nprocs}"
    log_path = RESULT_ROOT / "logs" / f"{tag}.log"
    result_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment.update(
        {
            "BENCH_MODEL": str(MODEL_ROOT / f"dpa4c_{grade}.pt2"),
            "BENCH_TAG": tag,
            "BENCH_NPROCS": str(nprocs),
            "BENCH_GPUS": ",".join(str(gpu) for gpu in gpus),
            "BENCH_FLAVOR": "energy",
            # Resume completed points after a scheduler restart. Set
            # ``BENCH_FRESH=1`` in the parent environment for a clean sweep.
            "BENCH_FRESH": os.environ.get("BENCH_FRESH", "0"),
            "BENCH_INCLUDE_SMALL": "1",
            "BENCH_RESULT_DIR": str(result_dir),
            "BENCH_WORK_ROOT": str(work_root),
            "HWLOC_COMPONENTS": "-cuda,-nvml,-gl,-opencl,-rsmi",
        }
    )
    environment.pop("MPICH_GPU_SUPPORT_ENABLED", None)
    log_stream = log_path.open("wb")
    process = subprocess.Popen(
        [str(PYTHON), str(LMP_SCAN)],
        cwd=HERE,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    log_stream.close()
    print(  # noqa: T201
        f"[start] {tag} on GPUs {','.join(str(gpu) for gpu in gpus)}",
        flush=True,
    )
    return ActiveJob(process, gpus, tag, log_path)


def reap(active: list[ActiveJob]) -> list[ActiveJob]:
    """Remove finished jobs and fail after reporting their logs."""
    remaining = []
    for job in active:
        return_code = job.process.poll()
        if return_code is None:
            remaining.append(job)
            continue
        print(  # noqa: T201
            f"[done] {job.tag} exit={return_code} log={job.log_path}",
            flush=True,
        )
        if return_code != 0:
            raise RuntimeError(f"{job.tag} failed; see {job.log_path}")
    return remaining


def run_all() -> None:
    """Run all jobs whenever enough disjoint GPUs are idle.

    Two-rank jobs fill a free machine first. Once they release four GPUs, a
    four-rank job starts immediately without waiting for every two-rank job.
    Eight-rank jobs start as soon as the full device set is available.
    """
    pending = deque(
        (grade, nprocs) for nprocs in reversed(PROCESS_COUNTS) for grade in GRADES
    )
    active: list[ActiveJob] = []
    reserved: set[int] = set()
    while pending or active:
        active = reap(active)
        reserved = {gpu for job in active for gpu in job.gpus}
        while pending:
            available = idle_gpus(reserved)
            candidate = next(
                (
                    index
                    for index, (_, nprocs) in enumerate(pending)
                    if nprocs <= len(available)
                ),
                None,
            )
            if candidate is None:
                break
            grade, nprocs = pending[candidate]
            del pending[candidate]
            gpus = tuple(available[:nprocs])
            active.append(start_scan(grade, nprocs, gpus))
            reserved.update(gpus)
        if active:
            time.sleep(POLL_SECONDS)
        elif pending:
            print(  # noqa: T201
                f"[wait] pending jobs need an idle GPU group; "
                f"{len(pending)} jobs remain",
                flush=True,
            )
            time.sleep(POLL_SECONDS)


def main() -> None:
    """Run every grade at 2, 4 and 8 GPUs with dynamic allocation."""
    if GPU_COUNT < max(PROCESS_COUNTS):
        raise ValueError(
            f"BENCH_GPU_COUNT={GPU_COUNT} is smaller than the eight-GPU batch"
        )
    if not MPIEXEC.is_file():
        raise FileNotFoundError(f"MPI launcher does not exist: {MPIEXEC}")
    print("[scheduler] dynamic multi-GPU allocation", flush=True)  # noqa: T201
    run_all()


if __name__ == "__main__":
    main()
