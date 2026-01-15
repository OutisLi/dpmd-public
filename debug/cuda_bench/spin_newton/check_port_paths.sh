#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Paths of the Newtonian spin state outside the integration step, run with the
# device integrator nve/spin/kk (models/spin_random_st, operator library OPLIB):
#   exchange  run_exchange_check.sh: one rank, two ranks with the legacy
#             exchange, two ranks with comm/exchange host (switched to legacy)
#   lost      in_spin_lost.lammps: an atom lost through a fixed boundary,
#             nve/spin/kk against the host fix of the saved build OLDLIB
#   restart   100 steps, write_restart, read_restart and 100 more steps against
#             200 uninterrupted steps (in_spin_restart.lammps)
# Usage: bash check_port_paths.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPLIB=${OPLIB:-/tmp/dpSR/lib}
OLDLIB=${OLDLIB:-/tmp/spin_newton/install_7d62aeb/lib}
MODELS=${HERE}/../models/spin_random_st
thermo_rows() { # log first-column-header
	awk -v h="$2" '$1 == "Step" && $2 == h {f++; next} f && $1 ~ /^[0-9]+$/ && NF > 3 {print}' "$1"
}

echo "== exchange"
rm -rf /tmp/spin_newton/exchange
MODELS=${MODELS} OPLIB=${OPLIB} bash "${HERE}/run_exchange_check.sh" | tail -3
E=/tmp/spin_newton/exchange
for name in legacy kokkos; do
	if diff <(thermo_rows "${E}/log_serial.lammps" KinEng) <(thermo_rows "${E}/log_${name}.lammps" KinEng) >/dev/null; then
		echo "exchange ${name}: thermo identical to one rank ($(thermo_rows "${E}/log_${name}.lammps" KinEng | wc -l) rows)"
	else
		echo "exchange ${name}: thermo differs from one rank"
	fi
done
grep -h "Newtonian spin state" "${E}/out_kokkos.txt" | sort -u | cut -c1-110

source "${HERE}/lmp_env.sh"
BASE_LD=${LD_LIBRARY_PATH}

echo "== lost"
W=/tmp/spin_newton/lost_port
rm -rf /tmp/spin_newton/lost_port
mkdir -p "${W}"
for side in host_old device; do
	if [[ ${side} == host_old ]]; then
		libs=${OLDLIB}:${OPLIB}
		extra=(-pk kokkos comm/exchange no)
	else
		libs=${OPLIB}
		extra=()
	fi
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
		echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
		exit 1
	}
	(cd "${W}" && LD_LIBRARY_PATH=${libs}:${BASE_LD} "${LMP}" -k on g 1 -sf kk -in "${HERE}/in_spin_lost.lammps" \
		-log "log_${side}.lammps" -screen none "${extra[@]}" \
		-var datafile "${HERE}/systems/diamond_v1_512.spin.data" -var model "${MODELS}/nano.pt2" \
		>"out_${side}.txt" 2>&1) || echo "lost ${side}: LAMMPS failed"
done
if diff <(thermo_rows "${W}/log_host_old.lammps" Atoms) <(thermo_rows "${W}/log_device.lammps" Atoms) >/dev/null; then
	echo "lost: nve/spin/kk identical to the host fix ($(thermo_rows "${W}/log_device.lammps" Atoms | wc -l) rows, $(thermo_rows "${W}/log_device.lammps" Atoms | tail -1 | awk '{print $2}') atoms at the end)"
else
	echo "lost: nve/spin/kk differs from the host fix"
fi

echo "== restart"
W=/tmp/spin_newton/restart_port
rm -rf /tmp/spin_newton/restart_port
mkdir -p "${W}"
for part in full first second; do
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
		echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
		exit 1
	}
	(cd "${W}" && LD_LIBRARY_PATH=${OPLIB}:${BASE_LD} "${LMP}" -k on g 1 -sf kk -in "${HERE}/in_spin_restart.lammps" \
		-log "log_${part}.lammps" -screen none -var part "${part}" \
		-var datafile "${HERE}/systems/diamond_v1_512.spin.data" -var model "${MODELS}/nano.pt2" \
		>"out_${part}.txt" 2>&1) || echo "restart ${part}: LAMMPS failed"
done
full_end=$(thermo_rows "${W}/log_full.lammps" KinEng | tail -1)
second_end=$(thermo_rows "${W}/log_second.lammps" KinEng | tail -1)
first_end=$(thermo_rows "${W}/log_first.lammps" KinEng | tail -1)
second_start=$(thermo_rows "${W}/log_second.lammps" KinEng | head -1)
echo "restart: state written at step 100: ${first_end}"
echo "restart: state read back:           ${second_start}"
echo "restart: step 200 uninterrupted:    ${full_end}"
echo "restart: step 200 after restart:    ${second_end}"
