# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime preflight and rebuild helpers for CUDA benchmarks."""

from __future__ import (
    annotations,
)

import os
import subprocess

from deepmd.env import (
    SHARED_LIB_DIR,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(os.path.dirname(HERE))
SOURCE_BUILD = os.path.join(REPOSITORY, "source", "build")
#: LAMMPS links against this standalone C++ tree.
DEEPMD_PREFIX = os.environ.get(
    "BENCH_DEEPMD_PREFIX", "/nas/outisli/Software/deepmd-kit_cpp"
)
#: The Python package resolves its operator library from its own ``lib``
#: directory. Installing there too keeps the interactive environment and the
#: benchmark on one build instead of letting the two copies drift apart.
PACKAGE_PREFIX = os.path.dirname(str(SHARED_LIB_DIR))
LAMMPS_BUILD = os.environ.get(
    "BENCH_LAMMPS_BUILD",
    "/nas/outisli/Software/lammps/lammps-patch_4Jul2026/build_kk",
)


def preflight(
    gpus: list[str], *, require_nep: bool, require_build_trees: bool = True
) -> None:
    """Validate GPU availability and external benchmark inputs.

    ``require_build_trees`` may be cleared when the caller does not intend to
    rebuild, which is the case for a deployment that carries only the installed
    libraries.
    """
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("At least one unique GPU index is required")
    for gpu in gpus:
        used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if used > 512:
            raise RuntimeError(f"GPU {gpu} is busy: {used} MiB allocated")
    required = (
        [
            os.path.join(SOURCE_BUILD, "CMakeCache.txt"),
            os.path.join(LAMMPS_BUILD, "CMakeCache.txt"),
        ]
        if require_build_trees
        else []
    )
    if require_nep:
        required.extend(
            (
                "/aisi-vepfs/outisli/Software/GPUMD/src/gpumd",
                os.path.join(HERE, "models", "nep89_20250409.txt"),
            )
        )
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing benchmark inputs: {missing}")


def build_runtime() -> None:
    """Rebuild and install DeePMD-kit, then relink LAMMPS.

    The build is installed to both the standalone tree that LAMMPS links and
    the Python package directory that the operator loader searches, so a
    benchmark can never run against a different build than an interactive
    session.
    """
    jobs = str(max(1, os.cpu_count() or 1))
    subprocess.run(
        ["cmake", "--build", SOURCE_BUILD, "-j", jobs],
        cwd=REPOSITORY,
        check=True,
    )
    for prefix in (DEEPMD_PREFIX, PACKAGE_PREFIX):
        subprocess.run(
            ["cmake", "--install", SOURCE_BUILD, "--prefix", prefix],
            cwd=REPOSITORY,
            check=True,
        )
    subprocess.run(
        ["cmake", "--build", LAMMPS_BUILD, "-j", jobs],
        cwd=REPOSITORY,
        check=True,
    )
