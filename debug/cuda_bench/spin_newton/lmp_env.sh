#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Environment of every LAMMPS run of the Newtonian spin benchmark: the fork
# executable with its DeePMD plugin, the installed DeePMD libraries and the
# spin device of the machine profile (machine.sh).
# shellcheck source=../machine.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/machine.sh"
export LMP=${LMP:-${BENCH_SPIN_LMP}}
export LAMMPS_PLUGIN_PATH=${BENCH_SPIN_PLUGIN_PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${BENCH_SPIN_GPU}}
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_claude_dpa4c
export LD_LIBRARY_PATH=${BENCH_DP_LIB}:${BENCH_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
