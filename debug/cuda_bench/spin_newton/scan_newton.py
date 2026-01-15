# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Capacity scan of Newtonian spin dynamics through ``lmp_scan.py``.

``lmp_scan.py`` is reused unchanged: this wrapper selects the OutisLi LAMMPS
fork with its DeePMD plugin and the development DeePMD libraries
before the module reads its environment, registers a ``spin_newton`` flavor
that pairs ``in_spin_newton.lammps`` with the spin data writer, and keeps the
data cache, results and work directories out of the shared benchmark tree.
Each point is one 10-step warm-up and one 100-step timed run, as in every
``lmp_scan.py`` flavor; ``in_spin_newton.lammps`` keeps both within its intact
window.

Usage
-----
    BENCH_MODEL=../models/spin_random/nano.pt2 BENCH_TAG=newton_nano \
        python scan_newton.py

Every environment variable of ``lmp_scan.py`` applies (``BENCH_TARGET_GRID``,
``BENCH_INCLUDE_SMALL``, ``BENCH_FRESH``, ...); ``BENCH_GPU`` defaults to 2.
"""

from __future__ import (
    annotations,
)

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import machine

# lmp_scan reads these when it is imported.
machine.load()
os.environ["BENCH_LMP"] = os.environ["BENCH_SPIN_LMP"]
os.environ["BENCH_PLUGIN_PATH"] = os.environ["BENCH_SPIN_PLUGIN_PATH"]
os.environ.setdefault("BENCH_GPU", os.environ["BENCH_SPIN_GPU"])
os.environ.setdefault("BENCH_RESULT_DIR", os.path.join(HERE, "results"))
os.environ.setdefault("BENCH_WORK_ROOT", "/tmp/spin_newton/scan")
os.environ["BENCH_FLAVOR"] = "spin_newton"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_claude_dpa4c"

import lmp_scan
from bench_newton import (
    wait_gpu_idle,
)

lmp_scan.FLAVORS["spin_newton"] = (
    os.path.join(HERE, "in_spin_newton.lammps"),
    "ensure_spin_data",
)
lmp_scan.SYSTEM_CACHE = os.path.join(HERE, "systems")
_run_case = lmp_scan.run_case


def _run_case_when_idle(*args, **kwargs) -> float:  # noqa: ANN002, ANN003
    """Run one ``lmp_scan`` point once the benchmark GPU is idle."""
    wait_gpu_idle(os.environ["BENCH_GPU"])
    return _run_case(*args, **kwargs)


lmp_scan.run_case = _run_case_when_idle

if __name__ == "__main__":
    if (
        os.environ["BENCH_GPU"] != os.environ["BENCH_SPIN_GPU"]
        or os.environ.get("BENCH_NPROCS", "1") != "1"
    ):
        raise SystemExit(
            f"this scan runs on the spin device {os.environ['BENCH_SPIN_GPU']} with one process only"
        )
    lmp_scan.main()
