# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage matching C API libraries, SDK headers and both LAMMPS runtimes."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import (
    Path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--deepmd-prefix", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "debug/cuda_bench"))
    import os

    import machine

    machine.load()
    root = args.artifacts.resolve()
    prefix = args.deepmd_prefix.resolve()
    library = root / "integrated"
    if not library.exists():
        shutil.copytree(root / "baseline/deepmd_lib", library)
    paths = (
        "libdeepmd.so",
        "libdeepmd_cc.so",
        "libdeepmd_c.so",
        "libdeepmd_backend_pt.so",
        "libdeepmd_backend_ptexpt.so",
        "libdeepmd_op_pt.so",
        "libdeepmd_op_cuda.so",
        "libdeepmd_dyn_cudart.so",
    )
    records = []
    for name in paths:
        # Installed RPATHs resolve sibling operators inside this snapshot.
        # Build-tree RPATHs can bypass LD_LIBRARY_PATH and load a live build.
        source = prefix / "lib" / name
        target = library / name
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source),
                "target": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    sdk = root / "sdk"
    shutil.copytree(prefix / "include", sdk / "include", dirs_exist_ok=True)
    (sdk / "lib").mkdir(exist_ok=True)
    for name in paths:
        shutil.copy2(library / name, sdk / "lib" / name)
    for flavor in ("public", "fork"):
        target = root / f"{flavor}_lammps"
        target.mkdir(exist_ok=True)
        source = root / "baseline" / f"{flavor}_runtime"
        executable, shared = source / "bin/lmp", source / "lib/liblammps.so.0"
        if flavor == "fork":
            build = Path(os.environ["BENCH_LAMMPS_FORK"]) / "build_kk"
            executable, shared = build / "lmp", build / "liblammps.so.0"
        for origin in (executable, shared):
            destination = target / origin.name
            shutil.copy2(origin, destination)
            records.append(
                {
                    "source": str(origin),
                    "target": str(destination),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
    edits = []
    plugins = []
    for flavor in ("public", "fork"):
        for side in ("vector", "indexed", "auto"):
            plugin = root / f"{flavor}_{side}_plugin/dpplugin.so"
            dynamic = subprocess.check_output(["readelf", "-d", str(plugin)], text=True)
            paths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic)
            assert len(paths) == 1, (plugin, dynamic)
            edits.append(
                f"file(RPATH_CHANGE FILE [==[{plugin}]==] OLD_RPATH [==[{paths[0]}]==] "
                "NEW_RPATH [==[$ORIGIN/../integrated]==])"
            )
            plugins.append(plugin)
    script = root / "stage_plugin_rpaths.cmake"
    script.write_text("\n".join(edits) + "\n")
    subprocess.run(["cmake", "-P", str(script)], check=True)
    for plugin in plugins:
        records.append(
            {
                "target": str(plugin),
                "sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
                "rpath": "$ORIGIN/../integrated",
            }
        )
    (root / "staged_runtime.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
