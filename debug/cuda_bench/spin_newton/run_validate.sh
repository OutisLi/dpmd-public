#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Short Newtonian spin runs that exercise atom sorting and neighbor rebuilds,
# for before/after comparisons of LAMMPS changes with compare_runs.py.
# Neighbor lists are rebuilt every 10 steps and atoms are sorted every 20
# steps, so 400 steps pass through the sort path 20 times; the state is dumped
# by atom ID every 50 steps.
# Usage: bash run_validate.sh TAG [LIBDIR]   (LIBDIR: saved liblammps directory)
# Extra LAMMPS -var pairs can be given in EXTRA, e.g. EXTRA="-var sort 0". They
# precede the settings above on the command line, and LAMMPS keeps the first
# definition of an index variable, so they override them.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=$1
source "${HERE}/lmp_env.sh"
if [[ $# -gt 1 ]]; then export LD_LIBRARY_PATH=$2:${LD_LIBRARY_PATH}; fi
WORK=/tmp/spin_newton/validate/${TAG}
mkdir -p "${WORK}"
for case in "nano 512" "nano 10296" "plus 10296"; do
	set -- ${case}
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null
	(cd "${WORK}" && "${LMP}" -k on g 1 -sf kk -in "${HERE}/in_spin_newton.lammps" \
		-log "log_$1_$2.lammps" ${EXTRA:-} \
		-var datafile "${HERE}/systems/diamond_v1_$2.spin.data" \
		-var model "${HERE}/../models/spin_random/$1.pt2" \
		-var dt 0.00001 -var nsteps 400 -var every 50 -var sort 20 \
		-var rebuild 10 -var check no -var dump 50 -var dumpfile "state_$1_$2.dump" \
		>"out_$1_$2.txt" 2>&1)
	echo "${TAG} $1 $2: $(grep -E '^Loop time' "${WORK}/log_$1_$2.lammps" | tail -1); $(grep -cE 'WARNING.*sort' "${WORK}/out_$1_$2.txt") sort warnings; builds $(grep -E 'Neighbor list builds' "${WORK}/log_$1_$2.lammps" | tail -1)"
done
