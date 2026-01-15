# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Check the long-range DPA4C fixtures in LAMMPS against their Python references.

The LAMMPS test of the long range (``source/lmp/tests/test_lammps_dpa4c_long_range_pt2.py``)
drives LAMMPS through the PyLAMMPS interface, which the Python package of the
fork does not include; this check runs the same fixtures of
``source/tests/infer/gen_dpa4c_long_range.py`` through the ``lmp`` executable
of the machine profile instead. For every
archive, under ``deepmd`` and ``deepmd/kk``, it compares the step-0 energy,
forces and per-atom virials with the references:

- the non-periodic module without a cell, at the default charge state and at
  ``charge_spin 1.0 2.0``;
- the periodic module in its triclinic cell (``p p p``) and as a cluster
  (``f f f``), the compressed archive on one and on two ranks;

and checks the rejections of a periodic box for a non-periodic module, of a
partially periodic box, and of an uncompressed periodic model under
``deepmd/kk``. It exits non-zero when a comparison exceeds its tolerance or a
rejection does not happen.

Environment: those of ``lmp_scan.py`` (machine profile of ``machine.sh``);
``CUDA_VISIBLE_DEVICES`` selects the devices, the ranks of a two-rank run
sharing one device when only one is visible.

Usage
-----
    python check_long_range_lammps.py [--work DIR]
"""

from __future__ import (
    annotations,
)

import argparse
import os
import subprocess
import sys
from typing import (
    NamedTuple,
)

import lmp_scan
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
INFER = os.path.join(REPO, "source", "tests", "infer")
sys.path.insert(0, os.path.join(REPO, "source", "lmp", "tests"))
from expected_ref import (
    read_expected_ref,
)

#: The six atoms of the fixtures: two groups about 9 A apart, beyond the cutoff
#: of the descriptor, which only the long-range module couples.
COORD = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
TYPES = (1, 2, 2, 1, 2, 2)
#: Box of the non-periodic fixtures, and the triclinic cell of the periodic ones
#: (gen_dpa4c_long_range.BOX): bounds, then the tilts xy xz yz.
CLUSTER_BOX = ("0 13 xlo xhi", "0 13 ylo yhi", "0 13 zlo zhi")
PERIODIC_BOX = (
    "0 13 xlo xhi",
    "0 12.5 ylo yhi",
    "0 13.5 zlo zhi",
    "0.6 -0.4 0.8 xy xz yz",
)
#: LAMMPS centroid/stress/atom columns xx yy zz xy xz yz yx zx zy, in the order
#: of the nine components of the reference virial.
TO_REFERENCE = [0, 6, 7, 3, 1, 8, 4, 5, 2]
NKTV2P = 1.6021765e6
#: Tolerances against the Python references: the uncompressed float64 archives
#: and the compressed float32 ones, whose evaluations in LAMMPS and in Python
#: round differently.
TOLERANCE = {
    False: {"energy": 1e-10, "force": 1e-10, "virial": 1e-10},
    True: {"energy": 2e-6, "force": 1e-6, "virial": 1e-6},
}
#: Pair-style keywords of the explicit charge state.
CHARGE_SPIN = "charge_spin 1.0 2.0"


class Case(NamedTuple):
    """One LAMMPS evaluation of a fixture archive."""

    #: Archive name in source/tests/infer, without the suffix.
    stem: str
    #: Section of the reference file, or a label of a rejection.
    section: str
    boundary: str
    data: str
    compressed: bool
    kokkos: bool
    keywords: str = ""
    ranks: int = 1


def write_data(path: str, box: tuple[str, ...]) -> None:
    """Write the six atoms in ``box``."""
    lines = ["six atoms", "", "6 atoms", "2 atom types", "", *box, "", "Masses"]
    lines += ["", "1 16", "2 2", "", "Atoms # atomic", ""]
    lines += [
        f"{i + 1} {t} {x:.10f} {y:.10f} {z:.10f}"
        for i, (t, (x, y, z)) in enumerate(zip(TYPES, COORD, strict=True))
    ]
    with open(path, "w") as stream:
        stream.write("\n".join(lines) + "\n")


def style(case: Case) -> str:
    """The pair style of a case."""
    return "deepmd/kk" if case.kokkos else "deepmd"


def run(work: str, tag: str, case: Case) -> tuple[int, str, str]:
    """Run one step-0 evaluation; return the exit status, the output and the dump."""
    dump = os.path.join(work, f"{tag}.dump")
    grid = f"{case.ranks} 1 1"
    script = f"""units metal
boundary {case.boundary}
atom_style atomic
atom_modify map yes
processors {grid}
read_data {case.data}
pair_style {style(case)} {os.path.join(INFER, case.stem + ".pt2")} {case.keywords}
pair_coeff * *
neighbor 1.0 bin
compute s all centroid/stress/atom NULL pair
dump d all custom 1 {dump} id fx fy fz c_s[1] c_s[2] c_s[3] c_s[4] c_s[5] c_s[6] c_s[7] c_s[8] c_s[9]
dump_modify d sort id format float %.17g
thermo_style custom step pe
thermo_modify format float %.17g
run 0
"""
    path = os.path.join(work, f"{tag}.in")
    with open(path, "w") as stream:
        stream.write(script)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = (
        f"{lmp_scan.DP_LIB}:{lmp_scan.TORCH_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    if lmp_scan.PLUGIN_PATH:
        env["LAMMPS_PLUGIN_PATH"] = lmp_scan.PLUGIN_PATH
    devices = len(env.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    # The Kokkos style is named in the input, so that a plugin without it
    # fails instead of falling back to the host style.
    command = [lmp_scan.LMP]
    if case.kokkos:
        command += ["-k", "on", "g", str(min(case.ranks, devices)), "-sf", "kk"]
        command += ["-pk", "kokkos", "newton", "off", "neigh", "full"]
    command += ["-in", path, "-log", "none"]
    if case.ranks > 1:
        lmp_scan.mpi_environment(env, lmp_scan.MPIEXEC)
        launcher = [lmp_scan.MPIEXEC, "-n", str(case.ranks), lmp_scan.MPI_RANK_EXEC]
        command = launcher + command
    try:
        result = subprocess.run(
            command, cwd=work, env=env, capture_output=True, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired as error:
        return -1, f"timed out: {error}", dump
    return result.returncode, result.stdout + result.stderr, dump


def outputs(text: str, dump: str) -> tuple[float, np.ndarray, np.ndarray]:
    """The potential energy, the forces and the per-atom virials of a run."""
    lines = text.splitlines()
    header = [
        i for i, line in enumerate(lines) if line.split()[:2] == ["Step", "PotEng"]
    ]
    energy = float(lines[header[-1] + 1].split()[1])
    table = np.loadtxt(dump, skiprows=9)
    virial = -table[:, 4:13][:, TO_REFERENCE] / NKTV2P
    return energy, table[:, 1:4], virial


def main() -> None:
    """Run every comparison and rejection; exit non-zero on a failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default=os.path.join(HERE, "work_long_range_check"))
    args = parser.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    cluster_data = os.path.join(work, "six.data")
    periodic_data = os.path.join(work, "six_periodic.data")
    write_data(cluster_data, CLUSTER_BOX)
    write_data(periodic_data, PERIODIC_BOX)
    failures = []

    # === Step 1. Energy, forces and virial against the references ===
    cases = []
    for compressed in (False, True):
        suffix = "_compressed" if compressed else ""
        cluster = f"deeppot_dpa4c_long_range{suffix}"
        periodic = f"deeppot_dpa4c_long_range_periodic{suffix}"
        for kokkos in (False, True):
            flags = (compressed, kokkos)
            cases += [
                Case(cluster, "default", "f f f", cluster_data, *flags),
                Case(cluster, "explicit", "f f f", cluster_data, *flags, CHARGE_SPIN),
            ]
            for section, boundary in (("periodic", "p p p"), ("cluster", "f f f")):
                # The device edge path of deepmd/kk passes an uncompressed model
                # no cell, which its pair style rejects (Step 2).
                if kokkos and not compressed and boundary == "p p p":
                    continue
                for ranks in (1, 2) if compressed else (1,):
                    case = Case(periodic, section, boundary, periodic_data, *flags)
                    cases.append(case._replace(ranks=ranks))
    for case in cases:
        variant = "kk" if case.kokkos else "host"
        name = case.stem[len("deeppot_dpa4c_long_range") :] or "_cluster"
        label = f"{name:22s} {style(case):9s} {case.section:8s} {case.ranks} rank"
        tag = f"{case.stem}_{variant}_{case.section}_{case.ranks}"
        status, text, dump = run(work, tag, case)
        if status != 0:
            failures.append(label)
            print(f"{label} FAILED\n{text[-1500:]}", flush=True)  # noqa: T201
            continue
        expected = os.path.join(INFER, case.stem + ".expected")
        reference = read_expected_ref(expected)[case.section]
        energy, force, virial = outputs(text, dump)
        target_force = reference["expected_f"].reshape(6, 3)
        target_virial = reference["expected_v"].reshape(6, 9)
        errors = {
            "energy": abs(energy - float(np.sum(reference["expected_e"]))),
            "force": float(np.abs(force - target_force).max()),
            "virial": float(np.abs(virial - target_virial).max()),
        }
        tolerance = TOLERANCE[case.compressed]
        bad = [key for key, value in errors.items() if value > tolerance[key]]
        if bad:
            failures.append(label)
        summary = " ".join(f"|d{k[0].upper()}| {v:.1e}" for k, v in errors.items())
        verdict = f"  EXCEEDS {','.join(bad)}" if bad else ""
        print(f"{label} {summary}{verdict}", flush=True)  # noqa: T201

    # === Step 2. Rejections ===
    cluster = "deeppot_dpa4c_long_range"
    periodic = f"{cluster}_periodic"
    rejections = [
        (
            Case(cluster, "", "p p p", cluster_data, False, False),
            "non-periodic module, p p p",
            "long-range module",
        ),
        (
            Case(f"{cluster}_compressed", "", "p p p", cluster_data, True, True),
            "non-periodic module kk, p p p",
            "long-range module",
        ),
        (
            Case(periodic, "", "p p f", periodic_data, False, False),
            "periodic module, p p f",
            "every dimension",
        ),
        (
            Case(periodic, "", "p p p", periodic_data, False, True),
            "uncompressed periodic kk, p p p",
            "pair_style deepmd for this model",
        ),
    ]
    for case, label, message in rejections:
        status, text, _ = run(work, "reject", case)
        rejected = status != 0 and message in text
        if not rejected:
            failures.append(f"rejection of {label}")
        print(f"rejected {label}: {rejected}", flush=True)  # noqa: T201

    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")  # noqa: T201
        sys.exit(1)
    print(f"all {len(cases)} comparisons and {len(rejections)} rejections pass")  # noqa: T201


if __name__ == "__main__":
    main()
