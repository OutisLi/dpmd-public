# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze isolated geometry-test archives from existing weights or test models."""

from __future__ import (
    annotations,
)

import argparse
import json
import sys
import zipfile
from pathlib import (
    Path,
)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--charge", action="store_true")
    args = parser.parse_args()
    root = args.artifacts.resolve()
    repo = Path(__file__).resolve().parents[3]
    torch.ops.load_library(str(root / "integrated/libdeepmd_op_pt.so"))
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
        serialize_from_file,
    )

    output = root / ("models_charge" if args.charge else "models_shape_restored")
    output.mkdir(exist_ok=False)
    if args.charge:
        sys.path.insert(0, str(repo / "source/tests/infer"))
        import gen_dpa4c_canonical as fixture

        models = [
            ("nano", {"model": fixture._build_model().to("cpu").serialize()}),
            ("spin_nano", {"model": fixture._build_spin_model().to("cpu").serialize()}),
        ]
    else:
        models = []
        for label, archive in (
            ("nano", "v20260911_trained_rcf/nano.pt2"),
            ("spin_nano", "spin_random/nano.pt2"),
        ):
            data = serialize_from_file(str(repo / "debug/cuda_bench/models" / archive))
            model = BaseModel.deserialize(data["model"]).to("cpu")
            descriptor = model.atomic_model.fused_decomposition()[0].descriptor
            # JSON's empty list loses the archived table's second dimension.
            assert descriptor.compress_coupling_meta.shape == (0,)
            descriptor.compress_coupling_meta = (
                descriptor.compress_coupling_meta.reshape(0, 8)
            )
            data["model"] = model.serialize()
            models.append((label, data))
    for label, data in models:
        target = output / f"{label}.pt2"
        deserialize_to_file(
            str(target), data, do_atomic_virial=True, lower_kind="dpa4c_canonical"
        )
        with zipfile.ZipFile(target) as archive:
            metadata = json.loads(archive.read("model/extra/metadata.json"))
        assert metadata["canonical_indexed_geometry"]
        if args.charge:
            assert metadata["charge_state_constants"]
        print(target, target.stat().st_size)  # noqa: T201


if __name__ == "__main__":
    main()
