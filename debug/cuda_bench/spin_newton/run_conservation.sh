#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# NVE energy conservation of Newtonian lattice + spin dynamics on the 512-atom
# diamond. Every grade integrates one fixed physical window at four timesteps,
# each half the previous one, from the same initial state; thermo every 10 steps.
# The window ends before the randomized model collapses the lattice (nano: 20 fs,
# plus and the other grades: 10 fs), and the smallest timestep takes 4,000 steps.
# Usage: bash run_conservation.sh [grade ...]   (default: nano plus)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/lmp_env.sh"
WORK=${WORK:-/tmp/spin_newton/conservation}
DATA=${DATA:-${HERE}/systems/diamond_v1_512.spin.data}
mkdir -p "${WORK}" "${HERE}/logs"
grades=("$@")
[[ ${#grades[@]} -gt 0 ]] || grades=(nano plus)
for grade in "${grades[@]}"; do
	if [[ ${grade} == nano ]]; then window=0.02; else window=0.01; fi # ps
	for steps in 500 1000 2000 4000; do
		dt=$(python3 -c "print(${window} / ${steps})")
		bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null
		log="${HERE}/logs/nve_${grade}_512_n${steps}.log"
		(cd "${WORK}" && "${LMP}" -k on g 1 -sf kk -in "${HERE}/in_spin_newton.lammps" \
			-log "${log}" -var datafile "${DATA}" \
			-var model "${HERE}/../models/spin_random/${grade}.pt2" \
			-var dt "${dt}" -var nsteps "${steps}" -var every 10 >/dev/null 2>&1) ||
			echo "${grade} ${steps}: LAMMPS failed, see ${log}"
		echo "${grade} dt=${dt} ps steps=${steps}: $(grep -E '^Loop time' "${log}" | tail -1)"
	done
done
