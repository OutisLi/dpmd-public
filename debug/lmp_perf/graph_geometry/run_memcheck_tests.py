# SPDX-License-Identifier: LGPL-3.0-or-later
"""Initialize CUDA before loading the operator library for sanitizer attachment."""

from __future__ import (
    annotations,
)

import sys
from pathlib import (
    Path,
)


def main() -> int:
    import pytest
    import torch

    torch.cuda.init()
    repo = Path(__file__).resolve().parents[3]
    torch.ops.load_library(str(repo / "source/build/op/pt/libdeepmd_op_pt.so"))
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
