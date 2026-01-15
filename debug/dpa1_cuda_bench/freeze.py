# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze a DPA1 (se_atten, attn_layer=0) model to a graph-lower ``.pt2``.

Inference throughput is independent of the trained weights, so a fresh
random-initialized model at the target architecture is a faithful speed
benchmark. The descriptor is emitted in either tebd input mode: ``concat``
(the graph-lower path today) or ``strip`` (with the smooth type-embedding
gate, the production form once graph-lower strip export lands). The fused CUDA
operator suite is baked into the package when ``DP_CUDA_INFER=1`` is set at
freeze time.

Usage
-----
    DP_CUDA_INFER=1 python freeze.py --shape L --mode concat --out dpa1_L.pt2

``--shape`` selects a named width preset (see :data:`SHAPES`); ``--neuron`` /
``--axis`` / ``--fitting`` override it for an arbitrary stack.
"""

from __future__ import (
    annotations,
)

import argparse

import torch  # noqa: TID253

from deepmd.pt_expt.entrypoints.main import freeze  # noqa: TID253
from deepmd.pt_expt.model.get_model import get_model  # noqa: TID253
from deepmd.pt_expt.train.wrapper import ModelWrapper  # noqa: TID253

# Full 118-element MatPES type map; the type count drives the layer-1 pair
# table size but not the per-edge kernel cost.
TYPE_MAP = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

# Named production width presets: (embedding neuron, axis_neuron, fitting neuron).
SHAPES = {
    "S": ([16, 16, 16], 16, [64, 64, 64]),
    "M": ([32, 32, 32], 16, [128, 128, 128]),
    "L": ([64, 64, 64], 16, [256, 256, 256]),
}


def model_params(
    neuron: list[int],
    axis: int,
    fitting: list[int],
    mode: str,
    act: str,
    lmax: int,
) -> dict:
    return {
        "type_map": TYPE_MAP,
        "descriptor": {
            "type": "se_atten",
            "sel": 181,
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "type_one_side": False,
            "neuron": neuron,
            "resnet_dt": False,
            "axis_neuron": axis,
            "lmax": lmax,
            "attn_layer": 0,
            "attn_dotr": True,
            "tebd_input_mode": mode,
            "smooth_type_embedding": mode == "strip",
            "set_davg_zero": False,
            "activation_function": act,
            "precision": "float32",
            "seed": 42,
        },
        "fitting_net": {
            "neuron": fitting,
            "resnet_dt": True,
            "activation_function": act,
            "precision": "float32",
            "seed": 42,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="L", choices=list(SHAPES))
    ap.add_argument("--neuron", type=int, nargs="+", default=None)
    ap.add_argument("--axis", type=int, default=None)
    ap.add_argument("--fitting", type=int, nargs="+", default=None)
    ap.add_argument("--mode", default="concat", choices=["concat", "strip"])
    ap.add_argument("--act", default="silu", choices=["silu", "tanh"])
    ap.add_argument("--lmax", type=int, choices=(1, 2, 3, 4), default=1)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--out", required=True, help="output .pt2 path")
    args = ap.parse_args()

    neuron, axis, fitting = SHAPES[args.shape]
    neuron = args.neuron or neuron
    axis = args.axis or axis
    fitting = args.fitting or fitting
    params = model_params(neuron, axis, fitting, args.mode, args.act, args.lmax)

    model = get_model(params)
    wrapper = ModelWrapper(model)
    state = wrapper.state_dict()
    state["_extra_state"] = {"model_params": params}
    ckpt = args.out.replace(".pt2", ".ckpt.pt")
    torch.save({"model": state}, ckpt)

    if args.compress:
        from deepmd.pt_expt.entrypoints.compress import (
            enable_compression,
        )
        from deepmd.pt_expt.utils.serialization import (
            deserialize_to_file,
        )

        intermediate = args.out.replace(".pt2", ".pte")
        deserialize_to_file(
            intermediate,
            {"model": model.serialize(), "min_nbor_dist": 1.4},
        )
        enable_compression(
            input_file=intermediate,
            output=args.out,
            stride=0.01,
            extrapolate=5.0,
            check_frequency=-1,
        )
    else:
        freeze(ckpt, args.out, lower_kind="graph")
    print(f"frozen -> {args.out}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
