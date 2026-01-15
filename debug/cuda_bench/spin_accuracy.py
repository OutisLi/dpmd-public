# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Accuracy of the compressed native-spin grades against the uncompressed model.

No trained magnetic checkpoint exists, so the spin grades are built from the
released presets with every zero-initialized parameter opened, as the spin
benchmark models are, from a fixed seed. Each is evaluated on a jittered diamond
supercell with random moments of magnitude two:

1. uncompressed, by autograd, in fp32 and in fp64 with the same weights;
2. compressed as ``dp compress`` does, through the compact canonical operator
   of the library loaded with ``BENCH_OP_LIB``.

The system is evaluated as carbon alone and with one atom changed to silicon,
a single-type and a two-type system. The errors of
the compressed outputs (energy per atom, force, magnetic force) against both
references are printed, so two operator libraries can be judged on the same
inputs.

Usage
-----
    BENCH_OP_LIB=.../libdeepmd_op_pt.so DP_CUDA_INFER=2 CUDA_VISIBLE_DEVICES=2 \
    python spin_newton/run_with_oplib.py spin_accuracy.py --grades mini air plus
"""

from __future__ import (
    annotations,
)

import argparse
import copy

import numpy as np
import torch  # noqa: TID253
from freeze_dpa4c import (
    TYPE_MAP,
    model_config,
)
from gen_system import (
    build,
)
from lowrank_accuracy import (
    compress,
    minimum_image_graph,
)

from deepmd.pt_expt.model.get_model import (  # noqa: TID253
    get_model,
)

#: Seed of the model initialization and of the moments.
SEED = 20261016


def spin_models(grade: str, device: torch.device) -> tuple[torch.nn.Module, ...]:
    """Return the fp32 and fp64 models of a spin grade with the same weights."""
    config = model_config(
        0, 0, 0, 2, 0, True, False, [0.0, 1.0], None, f"dpa4c-{grade}-v20260911"
    )
    torch.manual_seed(SEED)
    model32 = get_model(copy.deepcopy(config)).eval()
    with torch.no_grad():
        for parameter in model32.parameters():
            if parameter.numel() and not bool((parameter != 0).any()):
                parameter.fill_(0.3)
    config64 = copy.deepcopy(config)
    config64["descriptor"]["precision"] = "float64"
    config64["fitting_net"]["precision"] = "float64"
    model64 = get_model(config64).eval()
    model64.load_state_dict(model32.state_dict())
    return model32.to(device), model64.to(device)


def reference(
    model: torch.nn.Module,
    positions: np.ndarray,
    box: np.ndarray,
    atype: torch.Tensor,
    spin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Energy, force and magnetic force of the uncompressed model by autograd."""
    dtype = next(model.parameters()).dtype
    device = atype.device
    coord = torch.as_tensor(positions, dtype=dtype, device=device).reshape(1, -1, 3)
    cell = torch.as_tensor(np.diag(box), dtype=dtype, device=device).reshape(1, 9)
    out = model(coord, atype.reshape(1, -1), spin.to(dtype).reshape(1, -1, 3), box=cell)
    return {
        "energy": out["energy"].reshape(-1).double().detach(),
        "force": out["force"].reshape(-1, 3).double().detach(),
        "force_mag": out["force_mag"].reshape(-1, 3).double().detach(),
    }


def compressed(
    model: torch.nn.Module, graph: object, atype: torch.Tensor, spin: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Energy, force and magnetic force of a compressed model on the graph."""
    with torch.no_grad():
        out = model.forward_lower_canonical_graph(
            atype,
            graph.n_node,
            graph.n_local,
            graph.source,
            graph.edge_vec,
            graph.destination_row_ptr,
            graph.source_row_ptr,
            graph.source_order,
            do_atomic_virial=False,
            spin=spin,
        )
    return {
        "energy": out["energy"].reshape(-1).double(),
        "force": out["force"].reshape(-1, 3).double(),
        "force_mag": out["force_mag"].reshape(-1, 3).double(),
    }


def errors(test: dict, ref: dict, atoms: int) -> str:
    """Energy per atom and relative RMS force and magnetic-force errors."""
    energy = float((test["energy"] - ref["energy"]).abs().max()) / atoms
    parts = [f"dE/atom {energy:.2e} eV"]
    for key in ("force", "force_mag"):
        delta = (test[key] - ref[key]).pow(2).mean().sqrt()
        scale = ref[key].pow(2).mean().sqrt()
        parts.append(f"{key} rel rms {float(delta / scale):.2e}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grades", nargs="+", default=["nano", "mini", "neo", "air", "plus"]
    )
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--stride", type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device("cuda")
    positions, box = build(args.atoms, jitter=args.jitter)
    count = positions.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    direction = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    spin = (2.0 * direction / direction.norm(dim=1, keepdim=True)).float().to(device)
    carbon = torch.full((count,), TYPE_MAP.index("C"), dtype=torch.int64, device=device)
    mixed = carbon.clone()
    mixed[0] = TYPE_MAP.index("Si")
    graph = None
    for grade in args.grades:
        model32, model64 = spin_models(grade, device)
        descriptor = model32.atomic_model.fused_decomposition()[0].descriptor
        if graph is None:
            graph = minimum_image_graph(positions, box, float(descriptor.rcut), device)
        model = copy.deepcopy(model32)
        compress(model, args.stride)
        for label, atype in (("carbon", carbon), ("carbon+Si", mixed)):
            ref32 = reference(model32, positions, box, atype, spin)
            ref64 = reference(model64, positions, box, atype, spin)
            out = compressed(model, graph, atype, spin)
            print(f"== {grade} {label}", flush=True)  # noqa: T201
            print(  # noqa: T201
                f"  fp32 uncompressed vs fp64  {errors(ref32, ref64, count)}",
                flush=True,
            )
            print(  # noqa: T201
                f"  compressed vs fp32        {errors(out, ref32, count)}", flush=True
            )
            print(  # noqa: T201
                f"  compressed vs fp64        {errors(out, ref64, count)}", flush=True
            )
        del model, model32, model64
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
