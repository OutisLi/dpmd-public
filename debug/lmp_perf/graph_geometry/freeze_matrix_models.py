# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze the five-grade energy, native-spin and ZBL geometry matrix."""

from __future__ import (
    annotations,
)

import argparse
import copy
import hashlib
import json
import os
import sys
import zipfile
from pathlib import (
    Path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--grades", default="nano,mini,neo,air,plus")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    bench = repo / "debug/cuda_bench"
    sys.path.insert(0, str(bench))
    sys.path.insert(0, str(bench / "spin_newton"))
    import machine

    machine.load()
    import torch

    torch.ops.load_library(str(repo / "source/build/op/pt/libdeepmd_op_pt.so"))
    from freeze_dpa4c import (
        load_checkpoint,
        model_config,
        show_route_measurements,
    )
    from freeze_spin_random import (
        REDRAW_SCALE,
        REDRAW_SEED,
        redraw_zero_parameters,
    )

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    show_route_measurements()
    root = args.artifacts.resolve()
    output = root / "models"
    output.mkdir(parents=True, exist_ok=True)
    for grade in args.grades.split(","):
        checkpoint = Path(os.environ["DPA4C_CHECKPOINTS"].format(grade.capitalize()))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        energy_config = copy.deepcopy(state["_extra_state"]["model_params"])
        energy = load_checkpoint(str(checkpoint))
        spin_config = model_config(
            channels=None,
            fitting_width=64,
            fitting_depth=3,
            lmax=2,
            radial_modes=0,
            spin=True,
            charge_state=False,
            default_charge_spin=[0.0, 1.0],
            preset=f"dpa4c-{grade}-v20260911",
        )
        spin = get_model(spin_config).eval()
        redraw_zero_parameters(spin, REDRAW_SEED, REDRAW_SCALE)
        for flavor, config, base in (
            ("energy", energy_config, energy),
            ("spin", spin_config, spin),
        ):
            for zbl in (False, True):
                name = f"{grade}_{flavor}" + ("_zbl" if zbl else "")
                target = output / f"{name}.pt2"
                if target.exists():
                    raise FileExistsError(target)
                selected_config = copy.deepcopy(config)
                if zbl:
                    selected_config.update(
                        bridging_method="zbl",
                        bridging_r_inner=0.5,
                        bridging_r_outer=0.8,
                    )
                model = get_model(selected_config).eval()
                learned, pair = model.atomic_model.fused_decomposition()
                learned.load_state_dict(base.atomic_model.state_dict(), strict=True)
                assert (pair is not None) == zbl
                descriptor = learned.descriptor
                if zbl:
                    assert descriptor.bridging_scale == "absolute"
                    assert descriptor.bridging_f_inner == 0.5
                    assert descriptor.bridging_f_outer == 0.8
                if flavor == "spin":
                    assert descriptor.spin.spin_gate.count_nonzero() > 0
                descriptor.enable_compression(0.4 if zbl else 1.4, table_stride_1=0.01)
                deserialize_to_file(
                    str(target),
                    {"model": model.serialize()},
                    lower_kind="dpa4c_canonical",
                    do_atomic_virial=True,
                )
                with zipfile.ZipFile(target) as archive:
                    metadata = json.loads(archive.read("model/extra/metadata.json"))
                assert metadata["canonical_indexed_geometry"]
                record = {
                    "grade": grade,
                    "flavor": flavor,
                    "zbl": zbl,
                    "trained": flavor == "energy",
                    "channels": descriptor.channels,
                    "lmax": descriptor.lmax,
                    "radial_modes": descriptor.radial_modes,
                    "path": str(target),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "metadata": metadata,
                }
                (output / f"{name}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                print(name, target.stat().st_size, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
