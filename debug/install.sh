#!/bin/bash
# Reinstall deepmd-kit into the local conda environment and verify the DPA4/SeZM
# fused CUDA operators.
#
# The install is the editable scikit-build-core install of doc/outisli/install.md
# (PyTorch backend with the CUDA custom operators, TensorFlow and Paddle off).
# The C++ and CUDA operators are rebuilt incrementally in the build tree that
# pyproject.toml declares; the checks afterwards run against the installed
# library and exit non-zero when any of them fails.
#
#   bash debug/install.sh            # incremental rebuild and install, then verify
#   bash debug/install.sh --clean    # recompile every operator object first
#   bash debug/install.sh --verify   # skip the install, run the checks only
#
# Machine-specific settings: ENV_PREFIX is the conda/mamba environment that
# receives the install and CUDA_HOME the toolkit that compiles the operators.
# On another machine change these two lines (or export CUDA_HOME beforehand).
set -euo pipefail

ENV_PREFIX=/nas/outisli/Software/miniforge3/envs/dpmd
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

case "${1:-}" in
"" | --clean | --verify) MODE="${1:-install}" ;;
*)
	echo "usage: $0 [--clean|--verify]"
	exit 2
	;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ENV_PREFIX/bin/python"
BUILD_DIR="$REPO/build/py37-none-linux_x86_64" # build-dir = "build/{wheel_tag}" in pyproject.toml
LOG="$REPO/build/install.log"
cd "$REPO"

if [ "$MODE" != "--verify" ]; then
	export CUDAToolkit_ROOT="$CUDA_HOME" CUDA_PATH="$CUDA_HOME"
	CUDA_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
	export CUDA_VERSION
	export DP_VARIANT=cuda DP_ENABLE_PYTORCH=1 DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PADDLE=0
	# Every nvcc job needs a few GB of memory, so the parallelism stays well below
	# the core count.
	export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
	mkdir -p "$REPO/build"
	if [ "$MODE" = "--clean" ] && [ -f "$BUILD_DIR/build.ninja" ]; then
		ninja -C "$BUILD_DIR" -t clean
	fi
	echo "installing $REPO into $ENV_PREFIX (CUDA $CUDA_VERSION); follow the build with: tail -f $LOG"
	if ! "$PY" -m pip install -e . -v >"$LOG" 2>&1; then
		tail -n 40 "$LOG"
		echo "INSTALL FAILED (full log: $LOG)"
		exit 1
	fi
	# The Inductor/AOTI caches bake the custom-operator schemas into generated
	# dispatcher calls; a rebuilt operator library makes every cached graph stale.
	rm -rf "/tmp/torchinductor_$(whoami)"
	echo "install finished"
fi

echo "verifying the installed package and the DPA4 CUDA operators"
export OMP_NUM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0
DP_INSTALL_REPO="$REPO" "$PY" - <<'EOF'
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

import deepmd
from deepmd.env import SHARED_LIB_DIR
from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP

repo = os.environ["DP_INSTALL_REPO"]
failures = []


def check(condition: bool, message: str) -> None:
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


commit = subprocess.run(
    ["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()
print(f"deepmd-kit {deepmd.__version__} at commit {commit}, torch {torch.__version__}")

# === Step 1. Package location and operator library ===
check(
    os.path.realpath(deepmd.__file__).startswith(os.path.realpath(repo)),
    f"deepmd imports from the working tree ({deepmd.__file__})",
)
lib = os.path.join(SHARED_LIB_DIR, "libdeepmd_op_pt.so")
check(ENABLE_CUSTOMIZED_OP and os.path.exists(lib), f"operator library loaded ({lib})")
if os.path.exists(lib):
    built = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lib)))
    print(f"      library built {built}")

# === Step 2. Registered DPA4 operators ===
OPS = (
    "dpa4_edge_radial",
    "dpa4_edge_radial_backward",
    "dpa4_fp32_ridge",
    "dpa4_grid_pair",
    "dpa4_grid_pair_backward",
    "dpa4_so2_conv",
    "dpa4_so2_conv_backward",
    "dpa4_wigner_dense",
    "dpa4_wigner_dense_backward",
    "dpa4_wigner_runs",
    "dpa4_wigner_runs_backward",
    "dpa4_zonal_scatter",
    "dpa4_zonal_scatter_backward",
)
missing = [name for name in OPS if not hasattr(torch.ops.deepmd, name)]
check(not missing, "all DPA4 operators registered" + (f" (missing: {missing})" if missing else ""))

# === Step 3. Device code for the local GPU ===
check(torch.cuda.is_available(), "CUDA device visible to torch")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"      device {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    listing = subprocess.run(["cuobjdump", "--list-elf", lib], capture_output=True, text=True)
    if listing.returncode == 0:
        archs = sorted({int(m) for m in re.findall(r"sm_(\d+)", listing.stdout)})
        check(major * 10 + minor in archs, f"library carries SASS for sm_{major}{minor} (built for {archs})")
    else:
        print("      cuobjdump unavailable; architecture check skipped")

# === Step 4. Operator availability as the descriptor sees it ===
from deepmd.pt_expt.kernels.cuda.dpa4 import op_available as cuda_infer_available
from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    op_available as cuda_value_available,
)

check(cuda_infer_available(), "DPA4 CUDA inference operators available (DP_CUDA_INFER)")
check(cuda_value_available(), "DPA4 CUDA training value path available (DP_CUDA_TRAIN)")

# === Step 5. Numerical parity of the fused paths against the dense reference ===
# A small SeZM descriptor in the deployed layout evaluates one random cluster.
# Fresh output projections are zero, which would leave the output independent
# of the geometry, so those parameters are moved off zero first while the rest
# of the network keeps its designed initialization. The gates are read at
# construction time; each variant is deserialized from the same weights after
# setting the environment.
#
# The fused paths bind for float32 only. Training (second order like a force
# loss: the coordinate gradient of a quadratic objective) is judged against
# the float64 evaluation of the dense path: the dense float32 path itself sits
# at the rounding floor of the double backward, and the fused path must stay
# within a small factor of that floor, as the operator unit tests require.
# Inference compares the fused paths with the dense float32 path directly, with
# the tolerances of the repository tests.
import copy

from deepmd.pt.model.descriptor.sezm import DescrptSeZM
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution

GATES = (
    "DP_TRITON_TRAIN",
    "DP_CUDA_TRAIN",
    "DP_TRITON_INFER",
    "DP_CUDA_INFER",
    "DP_CUTILE_INFER",
    "DP_CUTE_INFER",
)


def set_gates(**levels: int) -> None:
    for name in GATES:
        os.environ[name] = str(levels.get(name, 0))


def as_float32(obj):
    """Return a serialized descriptor with every float64 leaf cast to float32."""
    if isinstance(obj, dict):
        return {
            key: "float32" if key == "precision" and value == "float64" else as_float32(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [as_float32(value) for value in obj]
    if isinstance(obj, np.ndarray) and obj.dtype == np.float64:
        return obj.astype(np.float32)
    return obj


device = torch.device("cuda:0")
nloc, nnei, rcut = 12, 16, 4.0
generator = torch.Generator().manual_seed(7)
coord = torch.rand((nloc, 3), generator=generator, dtype=torch.float64) * 3.0
atype = torch.randint(0, 2, (nloc,), generator=generator)
distance = torch.cdist(coord, coord).fill_diagonal_(float("inf"))
order = torch.argsort(distance, dim=1)[:, : nloc - 1]
within = torch.gather(distance, 1, order) < rcut
nlist = torch.full((nloc, nnei), -1, dtype=torch.int64)
nlist[:, : nloc - 1] = torch.where(within, order, torch.full_like(order, -1))


def inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extended_coord = coord.reshape(1, nloc, 3).to(device=device, dtype=dtype).requires_grad_(True)
    return extended_coord, atype.reshape(1, nloc).to(device), nlist.reshape(1, nloc, nnei).to(device)


def train_step(descriptor: DescrptSeZM, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    extended_coord, extended_atype, neighbors = inputs(dtype)
    output = descriptor(extended_coord, extended_atype, neighbors)[0]
    objective = (output**2).sum()
    gradient = torch.autograd.grad(objective, extended_coord)[0]
    return objective.detach().cpu().double().numpy(), gradient.detach().cpu().double().numpy()


def infer_step(descriptor: DescrptSeZM, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    extended_coord, extended_atype, neighbors = inputs(dtype)
    output = descriptor(extended_coord, extended_atype, neighbors)[0]
    gradient = torch.autograd.grad(output.sum(), extended_coord)[0]
    return output.detach().cpu().double().numpy(), gradient.detach().cpu().double().numpy()


def convolution(descriptor: DescrptSeZM) -> SO2Convolution:
    return next(m for m in descriptor.modules() if isinstance(m, SO2Convolution))


def relative_error(value: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(value - truth).max() / np.abs(truth).max())


set_gates()
descriptor = DescrptSeZM(
    ntypes=2,
    sel=[nnei],
    rcut=rcut,
    channels=32,
    n_radial=8,
    lmax=2,
    mmax=1,
    n_blocks=2,
    mixing_layers=3,
    radial_so2_mode="degree_channel",
    radial_so2_rank=1,
    n_atten_head=1,
    grid_branch=[1, 1, 1],
    s2_activation=[False, True],
    random_gamma=False,
    precision="float64",
    seed=7,
)
generator = torch.Generator().manual_seed(11)
with torch.no_grad():
    for parameter in descriptor.parameters():
        if parameter.abs().max() == 0:
            noise = torch.randn(parameter.shape, dtype=parameter.dtype, generator=generator)
            parameter.add_(0.1 * noise.to(parameter.device))
data64 = descriptor.serialize()
data32 = as_float32(copy.deepcopy(data64))

# --- training, float32, judged against the float64 dense evaluation ---
set_gates()
_, truth_gradient = train_step(DescrptSeZM.deserialize(data64).to(device).train(), torch.float64)
check(np.abs(truth_gradient).max() > 0, "dense reference responds to the geometry (non-zero coordinate gradient)")
_, dense_gradient = train_step(DescrptSeZM.deserialize(data32).to(device).train(), torch.float32)
dense_error = relative_error(dense_gradient, truth_gradient)
for levels, label in (
    ({"DP_CUDA_TRAIN": 1}, "DP_CUDA_TRAIN=1"),
    ({"DP_CUDA_TRAIN": 1, "DP_TRITON_TRAIN": 1}, "DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1"),
):
    set_gates(**levels)
    fused = DescrptSeZM.deserialize(data32).to(device).train()
    check(convolution(fused)._cuda_value_train is not None, f"{label} binds the fused CUDA value path")
    _, gradient = train_step(fused, torch.float32)
    fused_error = relative_error(gradient, truth_gradient)
    check(
        fused_error <= 3.0 * dense_error + 1e-6,
        f"training step with {label} stays within the dense float32 rounding floor "
        f"(gradient error vs float64: fused {fused_error:.1e}, dense {dense_error:.1e})",
    )

# --- inference, float32 ---
set_gates()
dense_output, dense_gradient = infer_step(DescrptSeZM.deserialize(data32).to(device).eval(), torch.float32)
for levels, label in (
    ({"DP_TRITON_INFER": 2, "DP_CUDA_INFER": 1}, "freeze operating point (DP_TRITON_INFER=2 DP_CUDA_INFER=1)"),
    ({"DP_TRITON_INFER": 1, "DP_CUDA_INFER": 2}, "fused SO(2) convolution (DP_CUDA_INFER=2)"),
):
    set_gates(**levels)
    accelerated = DescrptSeZM.deserialize(data32).to(device).eval()
    if levels["DP_CUDA_INFER"] == 1:
        check(accelerated._cuda_radial_fn is not None, "DP_CUDA_INFER=1 binds the CUDA edge-radial operator")
    else:
        check(convolution(accelerated)._cuda_conv_fn is not None, "DP_CUDA_INFER=2 binds the fused CUDA SO(2) convolution")
    output, gradient = infer_step(accelerated, torch.float32)
    check(
        np.allclose(output, dense_output, rtol=2e-4, atol=2e-5)
        and np.allclose(gradient, dense_gradient, rtol=2e-4, atol=2e-5),
        f"float32 inference at the {label} matches the dense path "
        f"(relative error {relative_error(gradient, dense_gradient):.1e} in the gradient)",
    )

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
EOF
