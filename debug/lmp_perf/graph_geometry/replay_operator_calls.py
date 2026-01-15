# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay recorded operators without rebuilding models or neighbor graphs."""

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


def main() -> None:
    import torch
    from torch.utils._pytree import (
        tree_flatten,
        tree_map,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--operator-library", type=Path)
    args = parser.parse_args()
    calls = torch.load(args.calls, weights_only=True, map_location="cpu")
    torch.cuda.init()
    repo = Path(__file__).resolve().parents[3]
    library = args.operator_library or repo / "source/build/op/pt/libdeepmd_op_pt.so"
    torch.ops.load_library(str(library.resolve(strict=True)))

    def to_device(value: Any) -> Any:
        return value.to("cuda") if isinstance(value, torch.Tensor) else value

    for call in calls:
        inputs = tree_map(to_device, call["args"])
        keywords = tree_map(to_device, call["kwargs"])
        result = getattr(torch.ops.deepmd, call["name"])(*inputs, **keywords)
        values, _ = tree_flatten(result)
        expected, _ = tree_flatten(call["expected"])
        torch.cuda.synchronize()
        assert len(values) == len(expected)
        for actual, reference in zip(values, expected, strict=True):
            assert torch.equal(actual.cpu(), reference), call["label"]
        print("PASS", call["label"])  # noqa: T201
    print("PASS all", len(calls), "captured operator calls")  # noqa: T201


if __name__ == "__main__":
    main()
