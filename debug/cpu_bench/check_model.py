# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Compare the compressed CPU inference path against the uncompressed model.

The two differ by the compression of the radial branch and by the operators
that evaluate the graph, so the comparison bounds the whole deployment
decision rather than any single kernel. The reference is the uncompressed
graph lower in the model's own float32 arithmetic.

Usage
-----
    OMP_NUM_THREADS=83 python check_model.py --atoms 512
"""

from __future__ import (
    annotations,
)

import argparse

from harness import (
    GRADES,
    build_lower_inputs,
    load_model,
)


def evaluate(grade: str, atoms: int, stride: float | None, jitter: float) -> dict:
    """Evaluate one grade, optionally compressed, and return its outputs."""
    model = load_model(GRADES[grade])
    if stride is not None:
        model.get_descriptor().enable_compression(0.0, table_stride_1=stride)
    sample = build_lower_inputs(model, atoms, jitter=jitter)
    output = model.forward_common_lower_graph(*sample.args)
    return {
        "energy": output["energy_redu"].detach().double(),
        "force": output["energy_derv_r"].detach().double().reshape(-1, 3),
        "virial": output["energy_derv_c_redu"].detach().double(),
        "atoms": sample.n_atom,
        "edges": sample.n_edge,
    }


def main() -> None:
    """Report the deviation of every grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--strides", default="0.002,0.01")
    parser.add_argument("--jitter", type=float, default=0.1)
    args = parser.parse_args()

    strides = [float(value) for value in args.strides.split(",")]
    print(
        f"{'grade':6s} {'stride':>7s} {'dE/atom (eV)':>14s} "
        f"{'dF max (eV/A)':>15s} {'|F| max':>10s} {'dVirial rel':>12s}"
    )
    for grade in args.grades.split(","):
        reference = evaluate(grade, args.atoms, None, args.jitter)
        for stride in strides:
            compressed = evaluate(grade, args.atoms, stride, args.jitter)
            energy_error = (
                float((compressed["energy"] - reference["energy"]).abs().max())
                / reference["atoms"]
            )
            force_error = float((compressed["force"] - reference["force"]).abs().max())
            force_scale = float(reference["force"].abs().max())
            virial_error = float(
                (compressed["virial"] - reference["virial"]).abs().max()
            ) / max(float(reference["virial"].abs().max()), 1e-12)
            print(
                f"{grade:6s} {stride:7.3f} {energy_error:14.3e} "
                f"{force_error:15.3e} {force_scale:10.3e} {virial_error:12.3e}"
            )


if __name__ == "__main__":
    main()
