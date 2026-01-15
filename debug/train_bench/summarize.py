#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Tabulate the bench results directory.

Reads ``work/results/<backend>-<size>-<variant>.txt`` files produced by
``bench.py`` and prints one table per backend: median step time and peak VRAM
per size and variant, each accelerated variant followed by its speedup and
memory ratio against that size's ``base``.

Usage: summarize.py [results_dir]
"""

import re
import sys
from pathlib import (
    Path,
)

BACKENDS = ("pt", "pt_expt")
SIZES = ("nano", "mini", "neo", "air", "plus", "pro", "max", "ultra")
BASE = "base"


def parse(path: Path) -> tuple[float, float, int, int] | None:
    """Return time, memory, atom count, and frame count of one result."""
    text = path.read_text()
    median = re.search(r"median\s+([\d.]+) ms", text)
    vram = re.search(r"peak VRAM ([\d.]+) GiB", text)
    workload = re.search(r"natoms=(\d+) nframes=(\d+)", text)
    if not median or not vram or not workload:
        return None
    return (
        float(median.group(1)),
        float(vram.group(1)),
        int(workload.group(1)),
        int(workload.group(2)),
    )


results = Path(sys.argv[1] if len(sys.argv) > 1 else "work/results")
# measured[backend][size][variant] = (ms, GiB, atoms, frames)
measured: dict[str, dict[str, dict[str, tuple[float, float, int, int]]]] = {}
variants: list[str] = []
for entry in sorted(results.glob("*.txt")):
    parsed = parse(entry)
    if parsed is None:
        continue
    backend, _, rest = entry.stem.partition("-")
    size, _, variant = rest.partition("-")
    if backend not in BACKENDS or size not in SIZES or not variant:
        continue
    measured.setdefault(backend, {}).setdefault(size, {})[variant] = parsed
    if variant not in variants:
        variants.append(variant)

# The base column carries no ratio, so it is narrower than the rest.
variants.sort(key=lambda name: (name != BASE, name))
CELL = 30

for backend in BACKENDS:
    if backend not in measured:
        continue
    print(f"\n=== {backend} ===")
    header = f"{'size':<6}{'workload':>14}" + "".join(
        f"{name:>{CELL}}" for name in variants
    )
    print(header)
    for size in SIZES:
        if size not in measured[backend]:
            continue
        row = measured[backend][size]
        base = row.get(BASE)
        sample = base or next(iter(row.values()))
        line = f"{size:<6}{f'{sample[2]}a/{sample[3]}f':>14}"
        for name in variants:
            cell = row.get(name)
            if cell is None:
                line += f"{'-':>{CELL}}"
                continue
            text = f"{cell[0]:7.1f} ms {cell[1]:5.2f} GiB"
            if name != BASE and base is not None:
                text += f" {base[0] / cell[0]:4.2f}x {cell[1] / base[1]:4.2f}m"
            line += f"{text:>{CELL}}"
        print(line)

print("\nRatios are against that size's base: NNx speedup, NNm memory factor.")
