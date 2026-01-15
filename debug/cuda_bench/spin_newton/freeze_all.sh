#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Freeze the five randomized native-spin DPA4C grades on the spin device of
# the machine profile (machine.sh) with the installed operator library.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${HERE}/../models/spin_random
# shellcheck source=../machine.sh
. "${HERE}/../machine.sh"
PYTHON=${BENCH_PYTHON}
GPU=${GPU:-${BENCH_SPIN_GPU}}
mkdir -p "${OUT}" "${HERE}/logs"
for grade in ${GRADES:-nano mini neo air plus}; do
	bash "${HERE}/gpu_idle.sh" "${GPU}"
	DP_CUDA_INFER=2 CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=8 \
		TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_claude_dpa4c \
		BENCH_OP_LIB="${BENCH_DP_LIB}/libdeepmd_op_pt.so" \
		"${PYTHON}" "${HERE}/run_with_oplib.py" "${HERE}/freeze_spin_random.py" \
		--preset "dpa4c-${grade}-v20260911" --out "${OUT}/${grade}.pt2" \
		>"${HERE}/logs/freeze_${grade}.log" 2>&1
	grep -E "redrawn|frozen" "${HERE}/logs/freeze_${grade}.log"
done
