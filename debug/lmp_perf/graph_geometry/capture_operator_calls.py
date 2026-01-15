# SPDX-License-Identifier: LGPL-3.0-or-later
"""Record real validated operator inputs for focused CUDA memory checking."""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)
from typing import (
    Any,
)


def main(
    indexed_cases: tuple[tuple[int, int, int, int, bool, bool, int], ...] | None = None,
    generic_channels: tuple[int, ...] = (8, 64),
) -> None:
    import pytest
    import torch
    from torch.utils._python_dispatch import (
        TorchDispatchMode,
    )
    from torch.utils._pytree import (
        tree_map,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-library", type=Path)
    parser.add_argument("--index-dtype", choices=("uint32", "int64"), default="int64")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[3]
    library = args.operator_library or repo / "source/build/op/pt/libdeepmd_op_pt.so"
    torch.ops.load_library(str(library.resolve(strict=True)))
    from source.tests.pt_expt.model import test_dpa4c_graph_lower as suite

    calls = []
    label = ""

    def to_cpu(value: Any) -> Any:
        return value.detach().cpu() if isinstance(value, torch.Tensor) else value

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            function: Any,
            types: Any,
            args: tuple = (),
            kwargs: dict | None = None,
        ) -> Any:
            result = function(*args, **(kwargs or {}))
            name = function._schema.name
            call_label = label
            wanted = name == "deepmd::dpa4c_graph_compress_backward"
            if name == "deepmd::dpa4c_canonical_compress_energy_force":
                values = dict(
                    zip(
                        (arg.name for arg in function._schema.arguments),
                        args,
                        strict=False,
                    )
                )
                values.update(kwargs or {})
                packed = values.get("packed")
                layout = (
                    "indexed"
                    if packed is not None and packed.numel() != 0
                    else "vector"
                )
                call_label = f"{label}_{layout}"
                wanted = True
            if wanted and not any(call["label"] == call_label for call in calls):
                calls.append(
                    {
                        "name": name.split("::")[1],
                        "label": call_label,
                        "args": tree_map(to_cpu, args),
                        "kwargs": tree_map(to_cpu, kwargs or {}),
                        "expected": tree_map(to_cpu, result),
                    }
                )
            return result

    with pytest.MonkeyPatch.context() as patch, Capture():
        for channels in generic_channels:
            label = f"generic_{channels}"
            suite.test_compressed_level_two_matches_autograd(patch, channels)
        indexed = suite.test_indexed_canonical_geometry_preserves_outputs
        cases = indexed_cases
        if cases is None:
            cases = next(
                mark.args[1]
                for mark in indexed.pytestmark
                if mark.name == "parametrize" and isinstance(mark.args[0], tuple)
            )
        for case in cases:
            label = "canonical_" + "_".join(map(str, case))
            indexed(patch, *case, index_dtype=getattr(torch, args.index_dtype))
        label = "canonical_atom_virial"
        suite.test_canonical_atom_virial_sums_to_frame(
            patch, getattr(torch, args.index_dtype)
        )
    assert len(calls) == len(generic_channels) + 2 * len(cases) + 1, [
        (call["label"], call["name"]) for call in calls
    ]
    torch.save(calls, args.output)
    print([(call["label"], call["name"]) for call in calls])  # noqa: T201


if __name__ == "__main__":
    main()
