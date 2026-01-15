#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Single-point dumps (energy, force, magnetic force) of every randomized spin
# model on the 512-atom diamond spin system, written to
# /tmp/spin_newton/points/<label>. Two labels compare with
#   python compare_text_runs.py --dumps a/<grade>.dump b/<grade>.dump \
#       --logs a/log_<grade>.lammps b/log_<grade>.lammps
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "${HERE}/lmp_env.sh"
LABEL=${1:?usage: run_points.sh LABEL}
OUT=/tmp/spin_newton/points/${LABEL}
mkdir -p "${OUT}"
bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}"
for grade in nano mini neo air plus; do
	"${LMP}" -k on g 1 -sf kk -in "${HERE}/in_spin_point.lammps" \
		-log "${OUT}/log_${grade}.lammps" -screen none \
		-var datafile "${HERE}/systems/diamond_v1_512.spin.data" \
		-var model "${HERE}/../models/spin_random/${grade}.pt2" \
		-var dumpfile "${OUT}/${grade}.dump"
	grep -A1 -E "^\s*Step\s+PotEng" "${OUT}/log_${grade}.lammps" | tail -1 | sed "s/^/${grade}: /"
done
