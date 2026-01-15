#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Atom migration between MPI ranks in Newtonian spin dynamics under KOKKOS.
#
# The Newtonian spin velocity and inertia are host arrays that the KOKKOS
# exchange functors do not know, while the legacy exchange packs every field
# registered by AtomVecSpin. Two ranks share the spin device and run the same 200 steps
# (0.1 fs, rebuild and exchange every 10 steps) of nano on 10,296 atoms with
#   legacy   package kokkos comm/exchange no    (host AtomVec::pack_exchange)
#   kokkos   package kokkos comm/exchange host  (KOKKOS exchange functors)
# and one rank as the reference; the thermo output (every 10 steps) includes
# the spin kinetic energy ske and the total energy.
# MODELS (directory of the <grade>.pt2 models) and OPLIB (library directories put
# first on the library path) select another model set and its operator library.
# Usage: bash run_exchange_check.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/lmp_env.sh"
MPIRUN=${MPIRUN:-$(dirname "${BENCH_MPIEXEC}")/mpirun}
MODELS=${MODELS:-${HERE}/../models/spin_random}
if [[ -n "${OPLIB:-}" ]]; then export LD_LIBRARY_PATH=${OPLIB}:${LD_LIBRARY_PATH}; fi
WORK=/tmp/spin_newton/exchange
mkdir -p "${WORK}"
common=(-k on g 1 -sf kk -in "${HERE}/in_spin_newton.lammps"
	-var datafile "${HERE}/systems/diamond_v1_10296.spin.data"
	-var model "${MODELS}/nano.pt2"
	-var dt 0.0001 -var nsteps 200 -var every 10 -var rebuild 10 -var check no)
run() { # name ranks package-args...
	local name=$1 ranks=$2
	shift 2
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null || {
		echo "GPU ${CUDA_VISIBLE_DEVICES} busy"
		return 1
	}
	(cd "${WORK}" && "${MPIRUN}" -np "${ranks}" "${LMP}" "${common[@]}" "$@" \
		-log "log_${name}.lammps" >"out_${name}.txt" 2>&1) ||
		{
			echo "${name}: LAMMPS failed"
			tail -5 "${WORK}/out_${name}.txt"
			return 1
		}
	echo "${name}: $(grep -E '^Loop time' "${WORK}/log_${name}.lammps" | tail -1)"
}
run serial 1
run legacy 2 -pk kokkos comm/exchange no
run kokkos 2 -pk kokkos comm/exchange host
