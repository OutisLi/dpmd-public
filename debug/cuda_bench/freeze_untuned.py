# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run a freeze script with the freeze-time fitting-route measurement removed.

The archive then carries no fitting-route table, and the fused energy-force
operator of compressed DPA4C takes the routes of its rules, which makes it the
reference of an A/B of the measured routes. The arguments are those of the
freeze script.

Usage
-----
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so python freeze_untuned.py \
        freeze_dpa4c.py --checkpoint ... --out ...
"""

import importlib
import os
import runpy
import sys

import torch  # noqa: TID253


def main() -> None:
    """Remove the fitting-route measurement, then run the freeze script."""
    torch.ops.load_library(os.environ["BENCH_OP_LIB"])
    # The measurement registers itself when its module is imported, so it is
    # imported first and then removed from the registry that the freeze runs.
    importlib.import_module("deepmd.pt_expt.kernels.fitting_routes")
    from deepmd.pt_expt.kernels.autotune import (
        _AUTOTUNERS,
    )

    _AUTOTUNERS.pop("dpa4c_fitting_routes")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
