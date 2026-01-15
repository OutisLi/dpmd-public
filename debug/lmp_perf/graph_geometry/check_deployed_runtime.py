# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify an actual LAMMPS deployment and the libraries it loads."""

from __future__ import (
    annotations,
)

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import (
    Path,
)

from benchmark_matrix import (
    compare_trajectories,
    digest,
    input_text,
    thermo_values,
    verify_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--elements", choices=("carbon", "bcn"), required=True)
    parser.add_argument("--spin", action="store_true")
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "debug/cuda_bench"))
    import machine

    machine.load()
    library = args.libraries.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="0",
        DP_INTRA_OP_PARALLELISM_THREADS="0",
        DP_BACKEND_PLUGIN_PATH=str(library),
        LAMMPS_PLUGIN_PATH=str(args.plugin.resolve()),
        LD_DEBUG="libs",
        LD_LIBRARY_PATH=":".join(
            [
                str(library),
                str(args.lmp.resolve().parent),
                str(args.lmp.resolve().parent.parent / "lib"),
                environment["BENCH_TORCH_LIB"],
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        ),
    )
    (output / "input.lmp").write_text(
        input_text(
            args.model.resolve(),
            args.data.resolve(),
            args.elements,
            args.spin,
            4,
            2,
            True,
        )
    )
    command = [
        str(args.lmp.resolve()),
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
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    process = subprocess.run(
        command,
        cwd=output,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=900,
    )
    (output / "screen.txt").write_text(process.stdout)
    (output / "loader.txt").write_text(process.stderr)
    process.check_returncode()
    loaded: dict[str, set[str]] = {}
    plugins = set()
    for path in re.findall(r"calling init:\s+(\S+)", process.stderr):
        if Path(path).name in ("dpplugin.so", "libdeepmd_lmp.so"):
            plugins.add(str(Path(path).resolve()))
        if Path(path).name.startswith(("libdeepmd", "liblammps.so")):
            loaded.setdefault(Path(path).name, set()).add(str(Path(path).resolve()))
    expected_plugin = (args.plugin.resolve() / "dpplugin.so").resolve()
    assert plugins == {str(expected_plugin)}, (expected_plugin, plugins)
    runtime = args.lmp.resolve().parent / "liblammps.so.0"
    if not runtime.is_file():
        runtime = args.lmp.resolve().parent.parent / "lib/liblammps.so.0"
    assert loaded.get("liblammps.so.0") == {str(runtime.resolve())}, (runtime, loaded)
    for name in (
        "libdeepmd_c.so",
        "libdeepmd_cc.so",
        "libdeepmd_backend_ptexpt.so",
        "libdeepmd_op_pt.so",
    ):
        assert name in loaded, (name, loaded)
    for name, paths in loaded.items():
        if name == "libdeepmd_lmp.so":
            continue
        expected = library / name
        if expected.is_file():
            assert paths == {str(expected.resolve())}, (name, paths, expected)
    resolved = {name: sorted(paths) for name, paths in loaded.items()}
    result = verify_trajectory(output, args.spin) | {
        "thermo": thermo_values(process.stdout),
        "loaded_libraries": resolved,
        "loaded_plugin": str(expected_plugin),
    }
    if args.reference is not None:
        reference_thermo = thermo_values((args.reference / "screen.txt").read_text())
        result["bit_exact"] = (
            all(
                result[f"{name}_sha256"] == digest(args.reference / f"{name}.dump")
                for name in ("before", "after")
            )
            and result["thermo"] == reference_thermo
        )
        result["comparisons"] = compare_trajectories(
            args.reference,
            output,
            args.spin,
            result["atoms"],
            reference_thermo,
            result["thermo"],
        )
        assert all(value["passed"] for value in result["comparisons"].values()), result[
            "comparisons"
        ]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(resolved, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
