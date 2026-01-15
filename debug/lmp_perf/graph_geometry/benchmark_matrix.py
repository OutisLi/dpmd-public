# SPDX-License-Identifier: LGPL-3.0-or-later
"""Measure fixed and automatic geometry on the complete DPA4C matrix."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import (
    Path,
)

from artifact_snapshot import (
    capture_snapshot,
    verify_snapshot,
)


def input_text(
    model: Path,
    data: Path,
    elements: str,
    spin: bool,
    steps: int,
    warmup: int,
    verify: bool,
) -> str:
    """Use a short evolving trajectory for every potential and grade."""
    timestep = min(2e-6, 0.004 / (steps + warmup))
    pair = (
        f'dpa4spin "{model}" magnetic_force cartesian' if spin else f'deepmd "{model}"'
    )
    velocity = "spin yes spmass 0.1" if spin else "mom yes rot yes dist gaussian"
    integration = (
        "nve/spin lattice_flag yes spin_flag yes"
        if spin
        else "nvt temp 300.0 300.0 0.1"
    )
    columns = "id type x y z vx vy vz fx fy fz"
    atomic_energy = ""
    moment = ""
    if spin:
        moment = "compute moments all property/atom sp spx spy spz fmx fmy fmz\n"
        columns += " " + " ".join(f"c_moments[{index}]" for index in range(1, 8))
    before = after = ""
    if verify:
        atomic_energy = "compute atom_energy all pe/atom\n"
        columns += " c_atom_energy"
        before = f"run 0\nwrite_dump all custom before.dump {columns} modify sort id format float %.17g\n"
        after = f"write_dump all custom after.dump {columns} modify sort id format float %.17g\n"
    return f"""units metal
boundary p p p
atom_style {"spin" if spin else "atomic"}
atom_modify map yes sort {0 if verify else 1000} 0.0
read_data "{data}"
pair_style {pair}
pair_coeff * * {"B C N" if elements == "bcn" else "C"}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
velocity all create 300.0 12345 {velocity}
fix integrate all {integration}
timestep {timestep:.17g}
thermo_style custom step ke {"ske " if spin else ""}pe etotal temp press pxx pyy pzz pxy pxz pyz
thermo_modify format float %.17g
thermo {steps}
{moment}{atomic_energy}{before}run {warmup}
run {steps}
{after}"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_values(screen: str) -> list[dict]:
    return [
        {
            "selected": item[0],
            "vector_ms": float(item[1]),
            "indexed_ms": float(item[2]),
            "startup_s": float(item[3]),
        }
        for item in re.findall(
            r"DeePMD canonical geometry: (vector|indexed) \(vector ([\d.]+) ms, indexed ([\d.]+) ms; calibration ([\d.]+) s\)",
            screen,
        )
    ]


def thermo_values(screen: str) -> list[list[float]]:
    """Read every reported thermodynamic state, excluding timing statistics."""
    rows = []
    columns = 0
    for line in screen.splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            columns = len(fields)
        elif line.startswith("Loop time"):
            columns = 0
        elif columns and fields and fields[0].isdigit():
            assert len(fields) == columns, line
            values = list(map(float, fields))
            assert all(map(math.isfinite, values)), line
            rows.append(values)
    assert rows, screen
    return rows


def verify_trajectory(work: Path, spin: bool) -> dict:
    """Check finite evolving states and retain exact comparison evidence."""
    import numpy as np

    before = np.loadtxt(work / "before.dump", skiprows=9)
    after = np.loadtxt(work / "after.dump", skiprows=9)
    assert before.shape == after.shape, work
    assert np.isfinite(before).all() and np.isfinite(after).all(), work
    result = {
        "atoms": int(before.shape[0]),
        "before_sha256": digest(work / "before.dump"),
        "after_sha256": digest(work / "after.dump"),
    }
    if spin:
        moment_change = float(np.max(np.abs(after[:, 12:15] - before[:, 12:15])))
        magnetic_force = float(np.max(np.abs(after[:, 15:18])))
        assert moment_change > 0 and magnetic_force > 0, work
        result.update(moment_change=moment_change, magnetic_force=magnetic_force)
    return result


def numerical_difference(
    reference: object, candidate: object, atol: float, rtol: float
) -> dict:
    """Report absolute and relative errors with a componentwise acceptance bound."""
    import numpy as np

    left, right = np.asarray(reference), np.asarray(candidate)
    assert left.shape == right.shape
    difference = right - left
    rms = float(np.sqrt(np.mean(difference**2)))
    reference_rms = float(np.sqrt(np.mean(left**2)))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms_abs": rms,
        "relative_l2": rms / reference_rms if reference_rms else None,
        "atol": atol,
        "rtol": rtol,
        "passed": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
    }


def compare_trajectories(
    reference: Path,
    candidate: Path,
    spin: bool,
    atoms: int,
    reference_thermo: list[list[float]],
    candidate_thermo: list[list[float]],
    states: tuple[str, ...] = ("before", "after"),
) -> dict:
    """Compare observables in their physical units and retain every error metric."""
    import numpy as np

    result = {}
    fields = {
        "position_angstrom": (slice(2, 5), 5e-10, 1e-10),
        "velocity_angstrom_per_ps": (slice(5, 8), 1e-7, 1e-5),
        "force_ev_per_angstrom": (slice(8, 11), 8e-6, 1e-4),
        "atomic_energy_ev": (slice(-1, None), 8e-6, 1e-4),
    }
    if spin:
        fields.update(
            {
                "spin_magnitude": (slice(11, 12), 5e-9, 1e-7),
                "spin_direction": (slice(12, 15), 5e-9, 1e-7),
                "magnetic_force_ev_per_moment": (slice(15, 18), 8e-6, 1e-4),
            }
        )
    for state in states:
        left = np.loadtxt(reference / f"{state}.dump", skiprows=9)
        right = np.loadtxt(candidate / f"{state}.dump", skiprows=9)
        assert left.shape == right.shape and left.shape[0] == atoms
        assert np.array_equal(left[:, :2], right[:, :2]), (reference, candidate)
        for name, (columns, atol, rtol) in fields.items():
            result[f"{state}_{name}"] = numerical_difference(
                left[:, columns], right[:, columns], atol, rtol
            )
    left, right = np.asarray(reference_thermo), np.asarray(candidate_thermo)
    assert left.shape == right.shape and np.array_equal(left[:, 0], right[:, 0])
    temperature = 5 if spin else 4
    for column, name in enumerate(
        ("ke", "ske", "pe", "etotal") if spin else ("ke", "pe", "etotal"), start=1
    ):
        result[f"thermo_{name}_ev_per_atom"] = numerical_difference(
            left[:, column] / atoms, right[:, column] / atoms, 8e-6, 1e-4
        )
    result["thermo_temperature_kelvin"] = numerical_difference(
        left[:, temperature], right[:, temperature], 1e-5, 1e-4
    )
    result["thermo_pressure_bar"] = numerical_difference(
        left[:, temperature + 1 :], right[:, temperature + 1 :], 0.1, 1e-4
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--baseline-artifacts", type=Path)
    parser.add_argument("--original-artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--sizes", default="512,10296,100800")
    parser.add_argument("--elements", default="carbon,bcn")
    parser.add_argument("--flavors", default="energy,spin")
    parser.add_argument("--potentials", default="plain,zbl")
    parser.add_argument("--cases", help="Comma-separated complete case names")
    parser.add_argument(
        "--baseline-layout", choices=("vector", "indexed", "auto"), default="vector"
    )
    parser.add_argument(
        "--original-layout", choices=("vector", "indexed", "auto"), default="vector"
    )
    parser.add_argument("--sides", default="vector,indexed,auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "debug/cuda_bench"))
    import machine

    machine.load()
    root = args.artifacts.resolve()
    baseline_root = (
        args.baseline_artifacts.resolve() if args.baseline_artifacts else None
    )
    original_root = (
        args.original_artifacts.resolve() if args.original_artifacts else None
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = capture_snapshot(root)
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    baseline_snapshot = capture_snapshot(baseline_root) if baseline_root else None
    if baseline_snapshot:
        (output / "baseline").mkdir()
        (output / "baseline/snapshot.json").write_text(
            json.dumps(baseline_snapshot, indent=2) + "\n"
        )
    original_snapshot = capture_snapshot(original_root) if original_root else None
    if original_snapshot:
        (output / "original").mkdir()
        (output / "original/snapshot.json").write_text(
            json.dumps(original_snapshot, indent=2) + "\n"
        )
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
        DP_BACKEND_PLUGIN_PATH=str(root / "integrated"),
    )
    sides = args.sides.split(",")
    assert len(sides) == len(set(sides))
    assert set(sides) <= {
        "vector",
        "indexed",
        "auto",
        "before_contract",
        "original_vector",
    }
    assert ("before_contract" in sides) == (baseline_root is not None)
    assert ("original_vector" in sides) == (original_root is not None)
    variants = {side: (root, side) for side in ("vector", "indexed", "auto")}
    if baseline_root:
        variants["before_contract"] = (baseline_root, args.baseline_layout)
    if original_root:
        variants["original_vector"] = (original_root, args.original_layout)
    configuration = vars(args) | {
        "artifacts": str(root),
        "baseline_artifacts": str(baseline_root) if baseline_root else None,
        "original_artifacts": str(original_root) if original_root else None,
        "output": str(output),
        "snapshot_sha256": snapshot["sha256"],
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    summaries = []
    failed_cases = []
    selected_cases = set(args.cases.split(",")) if args.cases else None
    for grade, flavor, potential, elements, atoms in itertools.product(
        args.grades.split(","),
        args.flavors.split(","),
        args.potentials.split(","),
        args.elements.split(","),
        map(int, args.sizes.split(",")),
    ):
        spin = flavor == "spin"
        zbl_suffix = "_zbl" if potential == "zbl" else ""
        model_name = f"{grade}_{flavor}{zbl_suffix}.pt2"
        data = (
            root
            / "systems"
            / f"{elements}_{atoms}{zbl_suffix}{'_spin' if spin else ''}.data"
        )
        runtime_name = "fork" if spin else "public"
        steps = 4 if args.verify else args.steps or 64
        warmup = 2 if args.verify else max(100, steps // 10)
        case = f"{grade}_{flavor}{zbl_suffix}_{elements}_{atoms}"
        if selected_cases is not None and case not in selected_cases:
            continue
        samples = {side: [] for side in sides}
        records = []
        first_repeat = -1 if not args.verify and args.steps is None else 0
        for repeat in range(first_repeat, 1 if args.verify else args.repeats):
            pilot = repeat == -1
            # Cyclic passes give every layout each launch-order position.
            rotation = repeat % len(sides)
            order = sides[rotation:] + sides[:rotation]
            if pilot:
                order = ["vector"]
            for side in order:
                work = output / case / f"{repeat}_{side}"
                work.mkdir(parents=True)
                selected_root, plugin_side = variants[side]
                model = selected_root / "models" / model_name
                (work / "input.lmp").write_text(
                    input_text(model, data, elements, spin, steps, warmup, args.verify)
                )
                runtime = selected_root / f"{runtime_name}_lammps"
                plugin = selected_root / f"{runtime_name}_{plugin_side}_plugin"
                environment["DP_BACKEND_PLUGIN_PATH"] = str(
                    selected_root / "integrated"
                )
                environment["LD_LIBRARY_PATH"] = ":".join(
                    [
                        str(selected_root / "integrated"),
                        str(runtime),
                        environment["BENCH_TORCH_LIB"],
                        environment["BENCH_DP_LIB"],
                        os.environ.get("LD_LIBRARY_PATH", ""),
                    ]
                )
                environment["LAMMPS_PLUGIN_PATH"] = str(plugin)
                command = [
                    str(runtime / "lmp"),
                    "-k",
                    "on",
                    "g",
                    "1",
                    "-sf",
                    "kk",
                    "-in",
                    "input.lmp",
                    "-log",
                    "none",
                ]
                (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                (work / "libraries.json").write_text(
                    json.dumps(
                        {
                            key: environment[key]
                            for key in (
                                "LD_LIBRARY_PATH",
                                "DP_BACKEND_PLUGIN_PATH",
                                "LAMMPS_PLUGIN_PATH",
                            )
                        },
                        indent=2,
                    )
                    + "\n"
                )
                process = subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    capture_output=True,
                    timeout=900,
                )
                (work / "screen.txt").write_text(process.stdout + process.stderr)
                process.check_returncode()
                thermodynamics = thermo_values(process.stdout)
                loops = list(
                    re.finditer(
                        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms",
                        process.stdout,
                    )
                )
                last = loops[-1]
                assert int(last.group(2)) == steps and int(last.group(3)) == atoms, work
                milliseconds = 1000 * float(last.group(1)) / steps
                assert math.isfinite(milliseconds), work
                tail = process.stdout[last.end() :]
                builds = int(re.search(r"Neighbor list builds = (\d+)", tail)[1])
                danger = int(re.search(r"Dangerous builds = (\d+)", tail)[1])
                assert danger == 0, work
                calibration = calibration_values(process.stdout)
                if plugin_side == "auto":
                    assert calibration, work
                record = {
                    "case": case,
                    "side": side,
                    "repeat": repeat,
                    "pilot": pilot,
                    "steps": steps,
                    "step_ms": milliseconds,
                    "builds": builds,
                    "danger": danger,
                    "calibration": calibration,
                }
                if args.verify:
                    record.update(verify_trajectory(work, spin))
                    record["thermo"] = thermodynamics
                with (output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(  # noqa: T201
                    {
                        key: record[key]
                        for key in ("case", "side", "repeat", "step_ms", "calibration")
                    },
                    flush=True,
                )
                if pilot:
                    # Two seconds per sample amortize loop/output overhead.
                    # The same measured step count serves all three layouts.
                    steps = min(
                        10000, max(100, math.ceil(2000 / milliseconds / 50) * 50)
                    )
                    warmup = max(100, steps // 10)
                    continue
                records.append(record)
                samples[side].append(milliseconds)
        mismatches = []
        comparisons = {}
        if args.verify:
            for name in ("before_sha256", "after_sha256"):
                if len({record[name] for record in records}) != 1:
                    mismatches.append(name)
            if len({json.dumps(record["thermo"]) for record in records}) != 1:
                mismatches.append("thermo")
            reference = next(
                (record for record in records if record["side"] == "before_contract"),
                records[0],
            )
            for record in records:
                if record is reference:
                    continue
                pair = f"{record['side']}_vs_{reference['side']}"
                comparisons[pair] = compare_trajectories(
                    output / case / f"0_{reference['side']}",
                    output / case / f"0_{record['side']}",
                    spin,
                    atoms,
                    reference["thermo"],
                    record["thermo"],
                )
                failures = [
                    name
                    for name, metrics in comparisons[pair].items()
                    if not metrics["passed"]
                ]
                if failures:
                    failed_cases.append(
                        {"case": case, "comparison": pair, "observables": failures}
                    )
        else:
            for repeat in range(args.repeats):
                assert (
                    len(
                        {
                            record["builds"]
                            for record in records
                            if record["repeat"] == repeat
                        }
                    )
                    == 1
                ), case
        medians = {side: statistics.median(values) for side, values in samples.items()}
        summary = {
            "case": case,
            "grade": grade,
            "flavor": flavor,
            "zbl": potential == "zbl",
            "elements": elements,
            "atoms": atoms,
            "steps": steps,
            "medians_ms": medians,
            "samples_ms": samples,
            "records": records,
            "snapshot_sha256": snapshot["sha256"],
        }
        if baseline_snapshot:
            summary["baseline_snapshot_sha256"] = baseline_snapshot["sha256"]
        if original_snapshot:
            summary["original_snapshot_sha256"] = original_snapshot["sha256"]
        if args.verify:
            summary.update(
                bit_exact=not mismatches, mismatches=mismatches, comparisons=comparisons
            )
        if "vector" in medians:
            summary["reductions_percent"] = {
                side: 100 * (1 - value / medians["vector"])
                for side, value in medians.items()
            }
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    verify_snapshot(root, output, snapshot)
    if baseline_snapshot:
        verify_snapshot(baseline_root, output / "baseline", baseline_snapshot)
    if original_snapshot:
        verify_snapshot(original_root, output / "original", original_snapshot)
    if selected_cases is not None:
        assert {case["case"] for case in summaries} == selected_cases
    assert not failed_cases, failed_cases


if __name__ == "__main__":
    main()
