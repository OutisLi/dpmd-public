#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Per-atom energies and forces of the 512-atom diamond with the trained Nano
# model at the given neighbor skins (in_fill.lammps, run 0), written into
# work_fill/ as dump.TAG_sSKIN and log.TAG_sSKIN. The skin sets the length of
# the candidate lists: the skins 0.5, 1, 2 and 2.5 Å keep them within the 512
# entries of the single-pass fill of the compact canonical graph, and 4 and
# 18 Å take the counted fill. The runs of two builds compare with
# spin_newton/compare_text_runs.py.
# Usage: [GPU=0] [LMP=...] [DP_LIB=...] [PLUGIN=...] bash run_fill.sh TAG SKIN...
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=machine.sh
. "${HERE}/machine.sh"
LMP=${LMP:-${BENCH_LMP}}
DP_LIB=${DP_LIB:-${BENCH_DP_LIB}}
PLUGIN=${PLUGIN-${BENCH_PLUGIN_PATH}}
tag=$1
shift
mkdir -p "${HERE}/work_fill"
cd "${HERE}/work_fill" || exit 1
for skin in "$@"; do
	LAMMPS_PLUGIN_PATH=${PLUGIN} LD_LIBRARY_PATH=${DP_LIB}:${BENCH_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
		CUDA_VISIBLE_DEVICES=${GPU:-${BENCH_GPUS%% *}} OMP_NUM_THREADS=1 timeout 900 "${LMP}" -k on g 1 -sf kk \
		-in "${HERE}/in_fill.lammps" -var datafile "${HERE}/systems/diamond_v1_512.data" \
		-var model "${HERE}/models/v20260911_trained/nano.pt2" -var skin "${skin}" \
		-var tag "${tag}_s${skin}" -log "log.${tag}_s${skin}" -screen none >"out.${tag}_s${skin}" 2>&1
	echo "skin ${skin} exit=$?"
done
