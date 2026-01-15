#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Cases in which the KOKKOS fix nvt must keep its thermostat chain on the host,
# run with one LAMMPS library build into /tmp/spin_newton/chain_checks/TAG:
#   dyn_first   in_nvt_dynamic.lammps, the timed run is the first run
#   dyn_second  the same after a "run 0", when the temperature compute already
#               knows that its group is dynamic
#   sphere_kk   in_nvt_sphere.lammps with -sf kk (fix nvt/sphere/kk)
#   sphere_host in_nvt_sphere.lammps without the KOKKOS package (fix nvt/sphere)
# Usage: bash run_chain_checks.sh TAG [LIBDIR]   (LIBDIR: saved liblammps directory)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=$1
source "${HERE}/lmp_env.sh"
if [[ $# -gt 1 ]]; then export LD_LIBRARY_PATH=$2:${LD_LIBRARY_PATH}; fi
WORK=/tmp/spin_newton/chain_checks/${TAG}
rm -rf "/tmp/spin_newton/chain_checks/${TAG:?}"
mkdir -p "${WORK}"
cd "${WORK}" || exit 1
run() { # name args...
	local name=$1
	shift
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
		echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
		return 1
	}
	"${LMP}" "$@" -screen none >"out_${name}.txt" 2>&1 ||
		{
			echo "${TAG} ${name}: LAMMPS failed"
			tail -3 "out_${name}.txt"
			return 1
		}
	echo "${TAG} ${name}: done"
}
model=${HERE}/../models/v20260911_trained/nano.pt2
data=${HERE}/../systems/diamond_v1_512.data
for prerun in 0 1; do
	if [[ ${prerun} == 0 ]]; then name=dyn_first; else name=dyn_second; fi
	run "${name}" -k on g 1 -sf kk -in "${HERE}/in_nvt_dynamic.lammps" -log "init_${name}.lammps" \
		-var datafile "${data}" -var model "${model}" -var prerun "${prerun}" \
		-var logfile "log_${name}.lammps" -var dumpfile "${name}.dump"
done
run sphere_kk -k on g 1 -sf kk -in "${HERE}/in_nvt_sphere.lammps" -log log_sphere_kk.lammps
run sphere_host -in "${HERE}/in_nvt_sphere.lammps" -log log_sphere_host.lammps
