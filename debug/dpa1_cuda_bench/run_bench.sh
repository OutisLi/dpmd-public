#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Launch the seven throughput scans across four GPUs: six LAMMPS+Kokkos deepmd
# curves (three MatPES sizes x {uncompressed mega, compressed table}) and the
# GPUMD NEP reference. Each scan runs to its first out-of-memory. Longest
# scans (small model + compress reach the largest N) are balanced against the
# shorter uncompressed scans.
set -u
PY=/aisi-vepfs/outisli/miniforge3/envs/dpmd/bin/python
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$DIR/results"

lmp() { # tag gpu   (model = models/dpa1_<tag>.pt2)
	BENCH_TAG="$1" BENCH_GPU="$2" BENCH_MODEL="$DIR/models/dpa1_$1.pt2" \
		"$PY" "$DIR/lmp_scan.py" >"$DIR/results/lmp_$1.log" 2>&1
}
nep() { BENCH_GPU="$1" "$PY" "$DIR/nep_scan.py" >"$DIR/results/nep.log" 2>&1; }

(lmp S_compress 0) &
(
	lmp M_compress 1
	lmp L_uncompressed 1
) &
(
	lmp L_compress 2
	lmp M_uncompressed 2
) &
(
	nep 3
	lmp S_uncompressed 3
) &
wait
echo "ALL_MD_DONE"
