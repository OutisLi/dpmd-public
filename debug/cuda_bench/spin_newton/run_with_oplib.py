# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run a Python script after loading the DeePMD-kit PyTorch operator library.

The library given by ``BENCH_OP_LIB`` registers the custom operators of the
installed C++ build, so a model frozen by the script records the operator
schema of that build rather than of the editable Python package.

Usage
-----
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so python run_with_oplib.py SCRIPT [ARGS...]
    BENCH_OP_LIB=/path/to/libdeepmd_op_pt.so python run_with_oplib.py -m MODULE [ARGS...]
"""

import os
import runpy
import sys

import torch  # noqa: TID253

torch.ops.load_library(os.environ["BENCH_OP_LIB"])
if sys.argv[1] == "-m":
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
