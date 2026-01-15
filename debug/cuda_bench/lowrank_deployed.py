# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end accuracy of the deployed low-rank radial table of DPA4C.

``enable_compression`` adopts a low-rank basis of the scalar-only radial
channels when the basis reproduces them (see ``build_low_rank_table``). This
script evaluates, on a jittered diamond supercell of the trained released
checkpoints, the energy, forces and virial of

1. the uncompressed model in fp32, as deployed without compression, and in
   fp64, the same weights evaluated exactly;
2. the compression with the direct radial table, the low-rank table cleared;
3. the compression as deployed, with the low-rank table when it is adopted,

and reports the errors of 2 and 3 against both references of 1.

Usage
-----
    python lowrank_deployed.py --grades neo air plus --jitter 0.05 0.3
"""

from __future__ import (
    annotations,
)

import argparse
import copy

import torch  # noqa: TID253
from gen_system import (
    build,
)
from lowrank_accuracy import (
    compare,
    compressed_outputs,
    load_trained,
    minimum_image_graph,
    reference_outputs,
)


def compressed(
    model32: torch.nn.Module, stride: float, low_rank: bool
) -> tuple[torch.nn.Module, bool]:
    """Compress a copy of ``model32`` as ``dp compress`` does."""
    model = copy.deepcopy(model32)
    model.fold_vacuum_reference()
    descriptor = model.atomic_model.fused_decomposition()[0].descriptor
    descriptor.enable_compression(1.4, table_stride_1=stride)
    adopted = descriptor.compress_basis_table.numel() > 0
    if not low_rank:
        empty = descriptor.compress_basis_table.new_empty(0)
        descriptor._store_compression_buffer("basis_table", empty)
        descriptor._store_compression_buffer("basis_matrix", empty.clone())
    return model, adopted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", nargs="+", default=["neo", "air", "plus"])
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--jitter", type=float, nargs="+", default=[0.05, 0.3])
    parser.add_argument("--stride", type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device("cuda")
    for jitter in args.jitter:
        positions, box = build(args.atoms, jitter=jitter)
        graph = None
        for grade in args.grades:
            model32 = load_trained(grade, device, "float32")
            carbon = model32.get_type_map().index("C")
            descriptor = model32.atomic_model.fused_decomposition()[0].descriptor
            if graph is None:
                graph = minimum_image_graph(
                    positions, box, float(descriptor.rcut), device
                )
            atype = torch.full((args.atoms,), carbon, dtype=torch.int64, device=device)
            ref64 = reference_outputs(
                load_trained(grade, device, "float64"), positions, box, carbon, device
            )
            ref32 = reference_outputs(model32, positions, box, carbon, device)
            ref32 = {key: value.detach() for key, value in ref32.items()}
            ref64 = {key: value.detach() for key, value in ref64.items()}
            print(f"== {grade}, jitter {jitter} A", flush=True)  # noqa: T201
            print(  # noqa: T201
                f"  fp32 uncompressed vs fp64   {compare(ref32, ref64, args.atoms)}",
                flush=True,
            )
            for low_rank in (False, True):
                model, adopted = compressed(model32, args.stride, low_rank)
                with torch.no_grad():
                    out = compressed_outputs(model, graph, atype)
                label = "low-rank" if low_rank and adopted else "direct  "
                if low_rank and not adopted:
                    label = "no basis"
                print(  # noqa: T201
                    f"  {label} vs fp32 uncompressed {compare(out, ref32, args.atoms)}",
                    flush=True,
                )
                print(  # noqa: T201
                    f"  {label} vs fp64             {compare(out, ref64, args.atoms)}",
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
