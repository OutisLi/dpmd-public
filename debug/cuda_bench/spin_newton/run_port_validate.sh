#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Device Newtonian integrator nve/spin/kk against the host fix nve/spin.
#
# Every case runs in_spin_newton.lammps three times on the spin device (lmp_env.sh) with the models of
# models/spin_random_st and the operator library in OPLIB:
#   host_old   the saved build before the device integrator, host fix nve/spin
#   device     the installed build, nve/spin/kk selected by -sf kk
#   host_new   the installed build, host fix nve/spin (hostfix 1), which reads
#              and writes the spin state through its dual views
# and compares the states dumped by atom ID (x, v, S, f, fm at %.17g) and the
# thermo output with compare_runs.py. The runs take 200 steps of 0.04 fs with
# sorting every 20 and reneighboring every 10 steps, so the spin state passes
# through device sorting and neighbor rebuilds.
# Further LAMMPS -var pairs in EXTRA override these settings for all runs, e.g.
# EXTRA="-var sort 0"; TAG names the output subdirectory.
# Usage: bash run_port_validate.sh [CASE ...]   (default: nano:512 plus:512)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/lmp_env.sh"
OPLIB=${OPLIB:-/tmp/dpSR/lib}
OLDLIB=${OLDLIB:-/tmp/spin_newton/install_7d62aeb/lib}
MODELS=${HERE}/../models/spin_random_st
PYTHON=${BENCH_PYTHON}
WORK=/tmp/spin_newton/port_validate/${TAG:-default}
BASE_LD=${LD_LIBRARY_PATH}
mkdir -p "${WORK}"
cases=("$@")
[[ ${#cases[@]} -gt 0 ]] || cases=(nano:512 plus:512)
run() { # name grade atoms libdirs extra-args...
	local name=$1 grade=$2 atoms=$3 libs=$4
	shift 4
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
		echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
		return 1
	}
	(cd "${WORK}" && LD_LIBRARY_PATH=${libs}:${BASE_LD} "${LMP}" -k on g 1 -sf kk \
		-in "${HERE}/in_spin_newton.lammps" -log "log_${name}.lammps" "$@" ${EXTRA:-} \
		-var datafile "${HERE}/systems/diamond_v1_${atoms}.spin.data" \
		-var model "${MODELS}/${grade}.pt2" \
		-var dt 0.00004 -var nsteps 200 -var every 20 -var sort 20 \
		-var rebuild 10 -var check no -var dump 20 -var dumpfile "state_${name}.dump" -var info 1 \
		>"out_${name}.txt" 2>&1) ||
		{
			echo "${name}: LAMMPS failed"
			tail -4 "${WORK}/out_${name}.txt"
			return 1
		}
	echo "${name}: $(grep -h -o -E 'integrate, +style = [^,]+' "${WORK}/out_${name}.txt" | tr -s ' ' | head -1); $(grep -E '^Loop time' "${WORK}/log_${name}.lammps" | tail -1)"
}
for point in "${cases[@]}"; do
	grade=${point%%:*}
	atoms=${point##*:}
	tag=${grade}_${atoms}
	run "${tag}_host_old" "${grade}" "${atoms}" "${OLDLIB}:${OPLIB}"
	run "${tag}_device" "${grade}" "${atoms}" "${OPLIB}"
	run "${tag}_host_new" "${grade}" "${atoms}" "${OPLIB}" -var hostfix 1
	for other in device host_new; do
		echo "${tag} host_old vs ${other}: $(${PYTHON} "${HERE}/compare_runs.py" \
			"${WORK}/state_${tag}_host_old.dump" "${WORK}/state_${tag}_${other}.dump" \
			--logs "${WORK}/log_${tag}_host_old.lammps" "${WORK}/log_${tag}_${other}.lammps" | tail -1)"
	done
done
