# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build a fixed-geometry reference from the current plugin sources."""

from __future__ import (
    annotations,
)

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import (
    Path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--geometry", choices=("vector", "indexed"), default="vector")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--runtime", choices=("public", "fork"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    build = args.build.resolve() if args.build else repo / "source/build"
    root = args.artifacts.resolve()
    label = f"{args.runtime}_" if args.runtime else ""
    work = root / f"{label}{args.geometry}_plugin"
    work.mkdir(exist_ok=True)
    automatic = root / (f"{label}auto_plugin" if args.runtime else "integrated")
    automatic.mkdir(exist_ok=True)
    shutil.copy2(build / "lmp/plugin/libdeepmd_lmp.so", automatic / "dpplugin.so")
    header = (repo / "source/lmp/compact_canonical_graph_kokkos.h").read_text()
    if args.geometry == "vector":
        assert header.count("supports_indexed = supported;") == 1
        header = header.replace(
            "supports_indexed = supported;", "supports_indexed = false;"
        )
    else:
        begin = header.index("    if (same_workload(workload)) {")
        end = header.index(" private:", begin)
        header = (
            header[:begin]
            + "    evaluate(DP_CANONICAL_INDEXED);\n  }\n\n"
            + header[end:]
        )
    (work / "compact_canonical_graph_kokkos.h").write_text(header)
    commands = json.loads((build / "compile_commands.json").read_text())
    objects = {}
    for stem in ("pair_deepmd_kokkos", "pair_dpa4spin_kokkos"):
        for suffix in (".h", ".cpp"):
            shutil.copy2(repo / f"source/lmp/{stem}{suffix}", work / f"{stem}{suffix}")
    shutil.copy2(repo / "source/lmp/plugin/deepmdplugin.cpp", work / "deepmdplugin.cpp")
    for stem in ("pair_deepmd_kokkos", "pair_dpa4spin_kokkos", "deepmdplugin"):
        entry = next(item for item in commands if item["file"].endswith(f"/{stem}.cpp"))
        command = shlex.split(entry["command"])
        lammps = next(
            Path(value[2:]).parent
            for value in command
            if value.startswith("-I")
            and value.endswith("/src")
            and (Path(value[2:]).parent / "lib/kokkos/bin/nvcc_wrapper").is_file()
        )
        command[0] = str(lammps / "lib/kokkos/bin/nvcc_wrapper")
        if "-include" in command:
            index = command.index("-include")
            del command[index : index + 2]
        if "-Winvalid-pch" in command:
            command.remove("-Winvalid-pch")
        index = command.index("-o")
        original = command[index + 1]
        command[index + 1] = str(work / f"{stem}.o")
        objects[original] = command[index + 1]
        command[command.index(entry["file"])] = str(work / f"{stem}.cpp")
        process = subprocess.run(command, cwd=build, text=True, capture_output=True)
        (work / f"{stem}.log").write_text(process.stdout + process.stderr)
        process.check_returncode()
    link = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "lmp/plugin/libdeepmd_lmp.so"],
        text=True,
    ).splitlines()[-1]
    command = [objects.get(value, value) for value in shlex.split(link)[2:-2]]
    command[command.index("-o") + 1] = str(work / "dpplugin.so")
    process = subprocess.run(command, cwd=build, text=True, capture_output=True)
    (work / "link.log").write_text(process.stdout + process.stderr)
    process.check_returncode()
    print(work / "dpplugin.so")  # noqa: T201


if __name__ == "__main__":
    main()
