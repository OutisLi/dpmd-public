# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Accuracy of the compressed DPA4C models on the multi-element systems.

Evaluates, on BCN configurations -- boron, carbon and nitrogen at random on the
sites of a jittered diamond supercell, as ``gen_system.write_bcn_data`` assigns
them -- or on their five-element counterparts with oxygen and fluorine
(``--flavor five``, as ``gen_system.write_five_data`` assigns them), the
energy, forces and virial of

1. the uncompressed trained model in fp32, as deployed without compression,
   and in fp64, the same weights evaluated exactly;
2. its compression as ``dp compress`` builds it, evaluated by the operator
   library that ``BENCH_OP_LIB`` names,

and reports the errors of 2 against both references of 1. Running it once with
each of two operator libraries compares their accuracy on systems of several
elements.

Usage
-----
    BENCH_OP_LIB=.../libdeepmd_op_pt.so python spin_newton/run_with_oplib.py \
        bcn_accuracy.py --grades air plus --atoms 4032 --jitter 0.05 0.3
"""

from __future__ import (
    annotations,
)

import argparse

import numpy as np
import torch  # noqa: TID253
from gen_system import (
    BCN_ELEMENTS,
    BCN_SEED,
    FIVE_ELEMENTS,
    build,
)
from lowrank_accuracy import (
    compare,
    compressed_outputs,
    load_trained,
    minimum_image_graph,
)
from lowrank_deployed import (
    compressed,
)

#: Elements of the systems of each flavor, as the data writers of
#: ``gen_system`` assign them.
FLAVOR_ELEMENTS = {"bcn": BCN_ELEMENTS, "five": FIVE_ELEMENTS}


def site_types(model: torch.nn.Module, atoms: int, elements: tuple) -> np.ndarray:
    """Model types of the sites, drawn as the data writers of ``gen_system`` do."""
    lammps_types = np.random.default_rng(BCN_SEED).integers(
        1, len(elements) + 1, size=atoms
    )
    type_map = model.get_type_map()
    model_types = np.array([type_map.index(name) for name, _ in elements])
    return model_types[lammps_types - 1]


def reference_outputs(
    model: torch.nn.Module,
    positions: np.ndarray,
    box: np.ndarray,
    types: np.ndarray,
    device: torch.device,
) -> dict:
    """Energy, forces and virial of the uncompressed model by autograd."""
    dtype = next(model.parameters()).dtype
    coord = torch.as_tensor(positions, dtype=dtype, device=device).reshape(1, -1, 3)
    cell = torch.as_tensor(np.diag(box), dtype=dtype, device=device).reshape(1, 9)
    atype = torch.as_tensor(types, dtype=torch.int64, device=device).reshape(1, -1)
    out = model(coord, atype, box=cell)
    return {
        "energy": out["energy"].reshape(-1).double().detach(),
        "force": out["force"].reshape(-1, 3).double().detach(),
        "virial": out["virial"].reshape(-1).double().detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grades", nargs="+", default=["air", "plus"])
    parser.add_argument("--flavor", default="bcn", choices=sorted(FLAVOR_ELEMENTS))
    parser.add_argument("--atoms", type=int, default=4032)
    parser.add_argument("--jitter", type=float, nargs="+", default=[0.05, 0.3])
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the jitter of the positions"
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    for jitter in args.jitter:
        positions, box = build(args.atoms, jitter=jitter, seed=args.seed)
        realized = positions.shape[0]
        for grade in args.grades:
            model32 = load_trained(grade, device, "float32")
            types = site_types(model32, realized, FLAVOR_ELEMENTS[args.flavor])
            descriptor = model32.atomic_model.fused_decomposition()[0].descriptor
            graph = minimum_image_graph(positions, box, float(descriptor.rcut), device)
            atype = torch.as_tensor(types, dtype=torch.int64, device=device)
            ref64 = reference_outputs(
                load_trained(grade, device, "float64"), positions, box, types, device
            )
            ref32 = reference_outputs(model32, positions, box, types, device)
            model, _ = compressed(model32, args.stride, low_rank=True)
            with torch.no_grad():
                out = compressed_outputs(model, graph, atype)
            print(  # noqa: T201
                f"== {grade}, {args.flavor}, {realized} atoms, jitter {jitter} A, "
                f"seed {args.seed}",
                flush=True,
            )
            print(  # noqa: T201
                f"  fp32 uncompressed vs fp64   {compare(ref32, ref64, realized)}",
                flush=True,
            )
            print(  # noqa: T201
                f"  compressed vs fp32 uncompressed {compare(out, ref32, realized)}",
                flush=True,
            )
            print(  # noqa: T201
                f"  compressed vs fp64          {compare(out, ref64, realized)}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
