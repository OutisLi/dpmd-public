#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# A/B timing of two LAMMPS library builds with bench_newton.py. The builds
# alternate round by round (one repeat each), so a slow drift of the device or
# the host does not favor one of them; round i writes the results
# newton_<label>_r<i>.csv.
# Usage: bash run_ab_bench.sh ROUNDS GRADES LABEL_A LIBDIR_A LABEL_B LIBDIR_B [bench_newton options]
#   LIBDIR "installed" runs the installed build.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUNDS=$1 GRADES=$2 LABEL_A=$3 LIB_A=$4 LABEL_B=$5 LIB_B=$6
shift 6
# shellcheck source=../machine.sh
. "${HERE}/../machine.sh"
PYTHON=${BENCH_PYTHON}
cd "${HERE}"
for round in $(seq 0 $((ROUNDS - 1))); do
	for side in A B; do
		if [[ ${side} == A ]]; then label=${LABEL_A} lib=${LIB_A}; else label=${LABEL_B} lib=${LIB_B}; fi
		libargs=()
		if [[ ${lib} != installed ]]; then libargs=(--lib "${lib}"); fi
		"${PYTHON}" bench_newton.py --label "${label}_r${round}" --grades "${GRADES}" \
			--repeats 1 "${libargs[@]}" "$@" | grep -E "ms/step"
	done
done
