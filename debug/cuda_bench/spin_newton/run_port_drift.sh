#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# NVE energy conservation of the device integrator nve/spin/kk against the host
# fix nve/spin of the saved build before it, on the 512-atom diamond with the
# models of models/spin_random_st: nano over 20 fs and plus over 10 fs, each at
# the smallest timestep of run_conservation.sh (0.01 fs, 2,000 steps, and
# 0.0025 fs, 4,000 steps); thermo every 10 steps. energy_drift.py summarizes
# both runs, and the thermo rows of the two integrators are compared.
# Usage: bash run_port_drift.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/lmp_env.sh"
OPLIB=${OPLIB:-/tmp/dpSR/lib}
OLDLIB=${OLDLIB:-/tmp/spin_newton/install_7d62aeb/lib}
MODELS=${HERE}/../models/spin_random_st
PYTHON=${BENCH_PYTHON}
WORK=/tmp/spin_newton/port_drift
BASE_LD=${LD_LIBRARY_PATH}
mkdir -p "${WORK}"
thermo_rows() { awk '/^ *Step +KinEng/{f++} f==2 && NF==7 && $1 ~ /^[0-9]+$/ {print}' "$1"; }
for spec in "nano 2000 0.02" "plus 4000 0.01"; do
	read -r grade steps window <<<"${spec}"
	dt=$(python3 -c "print(${window} / ${steps})")
	for side in host_old device; do
		if [[ ${side} == host_old ]]; then libs=${OLDLIB}:${OPLIB}; else libs=${OPLIB}; fi
		bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
			echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
			exit 1
		}
		log=${WORK}/nve_${grade}_512_n${steps}_${side}.log
		(cd "${WORK}" && LD_LIBRARY_PATH=${libs}:${BASE_LD} "${LMP}" -k on g 1 -sf kk \
			-in "${HERE}/in_spin_newton.lammps" -log "${log}" \
			-var datafile "${HERE}/systems/diamond_v1_512.spin.data" \
			-var model "${MODELS}/${grade}.pt2" -var dt "${dt}" -var nsteps "${steps}" \
			-var every 10 >/dev/null 2>&1) || echo "${grade} ${side}: LAMMPS failed, see ${log}"
	done
	"${PYTHON}" "${HERE}/energy_drift.py" "${WORK}/nve_${grade}_512_n${steps}_host_old.log" \
		"${WORK}/nve_${grade}_512_n${steps}_device.log" | cut -c1-190
	if diff <(thermo_rows "${WORK}/nve_${grade}_512_n${steps}_host_old.log") \
		<(thermo_rows "${WORK}/nve_${grade}_512_n${steps}_device.log") >/dev/null; then
		echo "${grade}: thermo of nve/spin and nve/spin/kk identical ($(thermo_rows "${WORK}/nve_${grade}_512_n${steps}_device.log" | wc -l) rows)"
	else
		echo "${grade}: thermo of nve/spin and nve/spin/kk differ"
	fi
done
