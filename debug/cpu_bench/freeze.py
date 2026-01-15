# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Freeze released DPA4C grades into CPU AOTInductor ``.pt2`` packages.

Usage
-----
    DEVICE=cpu python freeze.py --grade nano --variant plain
    DEVICE=cpu python freeze.py --grade nano --variant compress --stride 0.002
"""

from __future__ import (
    annotations,
)

import argparse
import os
import time

from harness import (
    GRADES,
    HERE,
    load_model,
)

OUT = os.path.join(HERE, "models")


def freeze(grade: str, variant: str, stride: float) -> str:
    """Freeze one grade and return the artifact path."""
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = load_model(GRADES[grade])
    if variant == "compress":
        model.get_descriptor().enable_compression(0.0, table_stride_1=stride)
    os.makedirs(OUT, exist_ok=True)
    # The archive stores its compiled artifacts under a directory named after
    # the output file, and the strict loader rejects any root entry outside
    # ``model/``, so the artifact is written directly at its final name rather
    # than staged under another one.
    path = os.path.join(OUT, f"dpa4c_{grade}_{variant}.pt2")
    start = time.time()
    deserialize_to_file(
        path,
        {"model": model.serialize()},
        lower_kind="auto",
        do_atomic_virial=True,
    )
    print(f"[{grade}/{variant}] froze in {time.time() - start:.1f}s -> {path}")
    return path


def main() -> None:
    """Parse the requested grades and freeze them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", default="nano", choices=(*GRADES, "all"))
    parser.add_argument("--variant", default="plain", choices=("plain", "compress"))
    parser.add_argument("--stride", type=float, default=0.002)
    args = parser.parse_args()
    grades = list(GRADES) if args.grade == "all" else [args.grade]
    for grade in grades:
        freeze(grade, args.variant, args.stride)


if __name__ == "__main__":
    main()
