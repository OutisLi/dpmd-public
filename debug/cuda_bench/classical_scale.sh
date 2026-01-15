#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Scan pure-carbon MEAM and Tersoff through lmp_scan.py. The scanner owns the
# common logarithmic small-to-large grid and the OOM bisection; this launcher
# only supplies the potential flavor and the MPI device groups.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/python
RESULT_ROOT="$HERE/results/classical"
WORK_ROOT="$HERE/work_classical"
MODEL_PLACEHOLDER=/tmp/classical_potential

run_scan() {
	potential="$1"
	nprocs="$2"
	gpus="$3"
	result_dir="$RESULT_ROOT/$potential/ranks${nprocs}"
	work_root="$WORK_ROOT/$potential/ranks${nprocs}"
	mkdir -p "$result_dir" "$work_root"
	BENCH_MODEL="$MODEL_PLACEHOLDER" \
		BENCH_TAG="classical_${potential}_mpi${nprocs}" \
		BENCH_GPU=0 \
		BENCH_GPUS="$gpus" \
		BENCH_NPROCS="$nprocs" \
		BENCH_KOKKOS=1 \
		BENCH_LOG_SCALE=1 \
		BENCH_FLAVOR="$potential" \
		BENCH_FRESH=1 \
		BENCH_INCLUDE_SMALL=1 \
		BENCH_RESULT_DIR="$result_dir" \
		BENCH_WORK_ROOT="$work_root" \
		"$PYTHON" "$HERE/lmp_scan.py"
}

for potential in meam tersoff; do
	run_scan "$potential" 1 0 >"/tmp/classical_${potential}_mpi1.log" 2>&1 &
	run_scan "$potential" 2 0,1 >"/tmp/classical_${potential}_mpi2.log" 2>&1 &
	run_scan "$potential" 4 2,3,4,5 >"/tmp/classical_${potential}_mpi4.log" 2>&1 &
	wait
	run_scan "$potential" 8 0,1,2,3,4,5,6,7 >"/tmp/classical_${potential}_mpi8.log" 2>&1
done
