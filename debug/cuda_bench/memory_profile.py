# SPDX-License-Identifier: LGPL-3.0-or-later
"""Attribute the device memory of one compressed DPA4C molecular-dynamics step.

Capacity, not throughput, decides the largest system a device can run, and
capacity is set by whatever scales with the atom count. This script samples the
device while LAMMPS runs a real step and compares the peak against the arrays
that scale, which is what a capacity proposal has to be judged against.

Sampling at several sizes separates the two parts of the budget. A linear fit
gives the per-atom slope and the fixed overhead, and the slope alone predicts
where the scan will run out of memory.

Because the pipeline is evaluated over node tiles, the descriptor, its
cotangent, the moment state and the fitting pre-activations no longer scale
with the system; they are bounded by the tile and belong to the fixed term.
What remains linear is the graph and the edge cotangent.

Usage
-----
    python memory_profile.py --grade neo --atoms 2985984 5025816 8998912
"""

from __future__ import (
    annotations,
)

import argparse
import os
import subprocess
import threading
import time

import gen_system
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = "/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/python"
LMP = "/aisi-vepfs/outisli/Software/lammps/lammps-patch_4Jul2026/build_kk/lmp"
DP_LIB = "/aisi-vepfs/outisli/Software/deepmd-kit_cpp/lib"
WARMUP, MEASURE = 10, 30
#: Mean neighbors of a diamond site within the 6 A model cutoff.
NEIGHBORS = 158.0
#: Usable device memory in bytes, which the largest successful run approaches.
DEVICE_BYTES = 95.22 * 2**30


def system_scale_terms(neighbors: float) -> dict[str, float]:
    """Return the bytes per atom of every array that scales with the system.

    Parameters
    ----------
    neighbors : float
        Mean neighbor count, which fixes the edge-to-node ratio.

    Returns
    -------
    dict[str, float]
        Bytes per atom, keyed by array.
    """
    word = 4.0
    return {
        "graph edge_vec": neighbors * 3 * word,
        "edge cotangent": neighbors * 3 * word,
        "graph source": neighbors * word,
        "graph source_order": neighbors * word,
        "graph row pointers": 2 * 8.0,
    }


def sample_peak(model: str, data_file: str, gpu: str) -> int:
    """Run one LAMMPS scan point and return the peak device memory in MiB.

    Parameters
    ----------
    model : str
        Frozen ``.pt2`` package.
    data_file : str
        LAMMPS data file of the diamond supercell.
    gpu : str
        CUDA device index.

    Returns
    -------
    int
        Highest device occupancy observed during the run, in MiB.

    Raises
    ------
    RuntimeError
        If LAMMPS does not complete.
    """
    peak = [0]
    stop = threading.Event()

    def poll() -> None:
        query = [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not stop.is_set():
            probe = subprocess.run(query, capture_output=True, text=True)
            value = probe.stdout.strip()
            if value.isdigit():
                peak[0] = max(peak[0], int(value))
            time.sleep(0.25)

    work = os.path.join(HERE, "work_memory_profile")
    os.makedirs(work, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["OMP_NUM_THREADS"] = "1"
    environment["LD_LIBRARY_PATH"] = (
        DP_LIB + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    completed = subprocess.run(
        [
            LMP,
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-in",
            os.path.join(HERE, "in.lammps"),
            "-var",
            "datafile",
            data_file,
            "-var",
            "model",
            model,
            "-var",
            "warmup",
            str(WARMUP),
            "-var",
            "nsteps",
            str(MEASURE),
        ],
        cwd=work,
        env=environment,
        capture_output=True,
        text=True,
    )
    stop.set()
    sampler.join()
    if completed.returncode != 0:
        raise RuntimeError(f"LAMMPS failed:\n{completed.stdout[-2000:]}")
    return peak[0]


def main() -> None:
    """Sample every requested size and report the fitted budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", default="neo")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--atoms",
        type=int,
        nargs="+",
        default=[2985984, 5025816, 8998912],
    )
    args = parser.parse_args()
    model = os.path.join(HERE, "models", "sweep", f"dpa4c_{args.grade}.pt2")

    counts, peaks = [], []
    for target in args.atoms:
        data_file, realized = gen_system.ensure_data(
            target, os.path.join(HERE, "systems")
        )
        peak = sample_peak(model, data_file, args.gpu) * 2**20
        counts.append(float(realized))
        peaks.append(float(peak))
        print(  # noqa: T201
            f"N={realized:>9d}  device peak {peak / 2**30:6.2f} GiB", flush=True
        )
    if len(counts) < 2:
        return

    slope, constant = np.polyfit(np.array(counts), np.array(peaks), 1)
    terms = system_scale_terms(NEIGHBORS)
    graph = sum(terms.values())
    print(  # noqa: T201
        f"\ndevice peak = {slope:.0f} B/atom * N + {constant / 2**20:.0f} MiB\n"
        f"predicted capacity = {(DEVICE_BYTES - constant) / slope / 1e6:.2f} M atoms\n"
    )
    print(f"{'array':22s} {'B/atom':>8s} {'share':>7s}")  # noqa: T201
    for name, value in sorted(terms.items(), key=lambda item: -item[1]):
        print(f"{name:22s} {value:8.0f} {value / slope * 100:6.1f}%")  # noqa: T201
    print(  # noqa: T201
        f"{'LAMMPS with Kokkos':22s} {slope - graph:8.0f} "
        f"{(slope - graph) / slope * 100:6.1f}%\n"
        f"{'fixed overhead':22s} {constant / 2**20:8.0f} MiB"
    )


if __name__ == "__main__":
    main()
