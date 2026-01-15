#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Evaluate matched checkpoints on the complete, fixed OMat24 sub001 validation set.
# Usage: matrix_accuracy.sh <gpu> <step> <run> [<run> ...]
set -Eeuo pipefail
gpu=$1
step=$2
shift 2
directory=$(cd "$(dirname "$0")" && pwd)
python=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0
cd "$directory"
for run in "$@"; do
	checkpoint="runs/$run/ema_$step.pt"
	output="runs/$run/accuracy_sub001_$step"
	if [ -f "$output.done" ]; then
		continue
	fi
	[ -f "$checkpoint" ]
	[ ! -e "$output.npz" ]
	echo "$(date -Is) starting $run at $step on GPU $gpu"
	"$python" -u pro_scan_fast.py "$checkpoint" \
		--lmdb /data/Datasets/LMDB/OMat24/sub001val.lmdb \
		--out "$output.npz" >"$output.log" 2>&1
	"$python" - "$output.npz" "runs/$run/input.json" <<'PY'
import json
import sys
import numpy as np
from deepmd.dpmodel.utils.lmdb_data import LmdbDataReader

with open(sys.argv[2]) as stream:
    type_map = json.load(stream)["model"]["type_map"]
reader = LmdbDataReader("/data/Datasets/LMDB/OMat24/sub001val.lmdb", type_map, batch_size=1)
with np.load(sys.argv[1]) as data:
    expected = len(reader.frame_nlocs)
    assert np.array_equal(np.sort(data["idx"]), np.arange(expected)), "incomplete frame coverage"
    assert all(np.isfinite(data[key]).all() for key in data.files), "non-finite evaluation result"
    print(f"Verified {expected} distinct frames in {sys.argv[1]}", flush=True)
PY
	touch "$output.done"
	echo "$(date -Is) completed $run at $step"
done
