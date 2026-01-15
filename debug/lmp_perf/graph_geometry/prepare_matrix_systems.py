# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepare immutable matrix inputs with active ZBL pairs and native moments."""

from __future__ import (
    annotations,
)

import argparse
import hashlib
import json
import re
from pathlib import (
    Path,
)

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--sizes", default="512,10296,100800")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.artifacts.resolve() / "systems"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for atoms in map(int, args.sizes.split(",")):
        generator = np.random.default_rng(20261022)
        directions = generator.normal(size=(atoms, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for elements, suffix in (("carbon", ""), ("bcn", ".bcn")):
            source = repo / f"debug/cuda_bench/systems/diamond_v1_{atoms}{suffix}.data"
            text = source.read_text()
            header, body = re.split(r"\nAtoms[^\n]*\n", text, maxsplit=1)
            rows = np.array(
                [line.split() for line in body.splitlines() if line.strip()],
                dtype=float,
            )
            assert rows.shape == (atoms, 5)
            assert np.array_equal(rows[:, 0], np.arange(1, atoms + 1))
            box = np.array(
                [
                    float(match[1]) - float(match[0])
                    for match in re.findall(
                        r"([\d.eE+-]+) ([\d.eE+-]+) [xyz]lo [xyz]hi", header
                    )
                ]
            )
            for zbl in (False, True):
                positions = rows[:, 2:5].copy()
                anchors = np.arange(0, atoms - 4, 64) if zbl else np.empty(0, dtype=int)
                if zbl:
                    delta = positions[anchors + 4] - positions[anchors]
                    delta -= np.rint(delta / box) * box
                    delta *= 0.65 / np.linalg.norm(delta, axis=1, keepdims=True)
                    positions[anchors + 4] = (positions[anchors] + delta) % box
                for spin in (False, True):
                    name = (
                        f"{elements}_{atoms}"
                        + ("_zbl" if zbl else "")
                        + ("_spin" if spin else "")
                    )
                    path = output / f"{name}.data"
                    with path.open("w") as stream:
                        stream.write(header + "\nAtoms\n\n")
                        for index in range(atoms):
                            values = positions[index].tolist()
                            if spin:
                                values.extend([*directions[index], 2.0])
                            line = f"{int(rows[index, 0])} {int(rows[index, 1])} "
                            stream.write(
                                line
                                + " ".join(f"{value:.17g}" for value in values)
                                + "\n"
                            )
                    records.append(
                        {
                            "path": str(path),
                            "atoms": atoms,
                            "elements": elements,
                            "spin": spin,
                            "zbl": zbl,
                            "short_pairs": int(anchors.size),
                            "short_distance_angstrom": 0.65 if zbl else None,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Prepared", len(records), "matrix systems")  # noqa: T201


if __name__ == "__main__":
    main()
