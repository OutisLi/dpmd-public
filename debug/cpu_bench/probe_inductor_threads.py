# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: TID253, T201
"""Probe which Inductor CPU options make the graph lower run in parallel.

The graph lower exports with symbolic node and edge axes whose size hints come
from the tiny synthetic trace system, so Inductor's C++ backend costs every
dynamic loop at roughly a hundred elements and emits it serially. This script
freezes one grade under a few option sets and reports how many generated
kernels carry an OpenMP region and what the baked thread count is.
"""

from __future__ import (
    annotations,
)

import os
import re
import zipfile

import torch
from harness import (
    GRADES,
    HERE,
    load_model,
)

OUT = os.path.join(HERE, "models")

OPTION_SETS: dict[str, dict] = {
    "default": {},
    "chunk1": {"cpp.min_chunk_size": 1},
    "chunk1_dyn": {"cpp.min_chunk_size": 1, "cpp.dynamic_threads": True},
}


def probe(grade: str, name: str, options: dict, threads: int) -> None:
    """Freeze one grade with one option set and report its parallel regions."""
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    torch.set_num_threads(threads)
    model = load_model(GRADES[grade])
    path = os.path.join(OUT, f"probe_{grade}_{name}.pt2")
    import deepmd.pt.utils.compile_compat as compile_compat

    original = compile_compat.build_inductor_compile_options

    def patched(*args: object, **kwargs: object) -> dict:
        merged = original(*args, **kwargs)
        merged.update(options)
        return merged

    compile_compat.build_inductor_compile_options = patched
    try:
        deserialize_to_file(
            path,
            {"model": model.serialize()},
            lower_kind="auto",
            do_atomic_virial=True,
        )
    finally:
        compile_compat.build_inductor_compile_options = original

    with zipfile.ZipFile(path) as archive:
        source = next(
            archive.read(entry).decode()
            for entry in archive.namelist()
            if entry.endswith("kernel.cpp")
        )
    kernels = len(re.findall(r'extern "C"\s+void\s+cpp_fused', source))
    parallel = len(re.findall(r"#pragma omp parallel", source))
    baked = sorted(set(re.findall(r"num_threads\((\w+)\)", source)))
    print(
        f"{name:12s} torch_threads={threads:3d} kernels={kernels:3d} "
        f"omp_regions={parallel:3d} num_threads={baked}"
    )


if __name__ == "__main__":
    for option_name, option_set in OPTION_SETS.items():
        probe("nano", option_name, option_set, threads=83)
