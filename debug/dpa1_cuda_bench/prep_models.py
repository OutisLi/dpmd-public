# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze a MatPES DPA1-L0 checkpoint to two graph-lower pt_expt ``.pt2`` packages.

For one size (``S`` / ``M`` / ``L``) the pt-backend checkpoint is loaded, cast to
the backend-agnostic serialization, and frozen twice on the diamond graph lower:

* ``dpa1_<size>_uncompressed.pt2`` -- the uncompressed fused mega kernel
  (``DP_CUDA_INFER=2``), and
* ``dpa1_<size>_compress.pt2`` -- the geometrically compressed table mega kernel
  (``dp compress`` export, ``DP_CUDA_INFER=2``).

Both are strip ``se_atten_v2``, attention-free, and load in the C++ LAMMPS
``pair_style deepmd``. Usage: ``prep_models.py <S|M|L>``.
"""

from __future__ import (
    annotations,
)

import os
import sys

import torch  # noqa: TID253

MODELS = "/aisi-vepfs/outisli/Research/aisi_dp_script/MatPES/others"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OP_LIBRARY = os.environ.get(
    "BENCH_DP_OP",
    "/aisi-vepfs/outisli/Software/deepmd-kit_cpp/lib/libdeepmd_op_pt.so",
)
# Diamond nearest C-C is a*sqrt(3)/4 ~ 1.545 A; the table's extrapolate factor
# covers the thermal jitter below this floor.
MIN_NBOR = 1.4


def load_pt_model(ckpt: str, device: torch.device):  # noqa: ANN201
    """Load a pt-backend DPA1 checkpoint into an eval-mode model."""
    from deepmd.pt.model.model import (
        get_model,
    )
    from deepmd.pt.train.wrapper import (
        ModelWrapper,
    )

    sd = torch.load(ckpt, map_location=device, weights_only=True)
    sd = sd.get("model", sd)
    model = get_model(sd["_extra_state"]["model_params"]).to(device)
    wrapper = ModelWrapper(model)
    wrapper.load_state_dict(sd)
    model = wrapper.model["Default"]
    model.eval()
    return model


def main() -> None:
    size = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.ops.load_library(OP_LIBRARY)
    device = torch.device("cuda")

    from deepmd.pt_expt.entrypoints.compress import enable_compression as compress_entry
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = load_pt_model(f"{MODELS}/DPA1-L0-{size}/model_ema.ckpt.pt", device)
    data = model.serialize()

    # === Uncompressed fused mega kernel (DP_CUDA_INFER=2). ===
    os.environ["DP_CUDA_INFER"] = "2"
    uncompressed = f"{OUT}/dpa1_{size}_uncompressed.pt2"
    deserialize_to_file(
        uncompressed, {"model": data}, lower_kind="graph", do_atomic_virial=True
    )
    print(f"[{size}] froze uncompressed -> {uncompressed}", flush=True)  # noqa: T201

    # === Geometrically compressed table mega kernel (DP_CUDA_INFER=2). ===
    pte = f"{OUT}/dpa1_{size}.pte"
    deserialize_to_file(pte, {"model": data, "min_nbor_dist": MIN_NBOR})
    os.environ["DP_CUDA_INFER"] = "2"
    compressed = f"{OUT}/dpa1_{size}_compress.pt2"
    compress_entry(
        input_file=pte,
        output=compressed,
        stride=0.01,
        extrapolate=5,
        check_frequency=-1,
    )
    print(f"[{size}] froze compressed -> {compressed}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
