#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Run in_nvt_validate.lammps for nano and plus at 512 and 10296 atoms plus
# the thermostat variants, with one LAMMPS build, into /tmp/spin_newton/nvtval/TAG.
# Usage: bash run_nvt_validate.sh TAG LMP LIBDIR
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=$1
export LMP=$2
source "${HERE}/lmp_env.sh"
export LD_LIBRARY_PATH=$3:${LD_LIBRARY_PATH}
SYS=${HERE}/../systems
MOD=${HERE}/../models/v20260911_trained
WORK=/tmp/spin_newton/nvtval/${TAG}
mkdir -p "${WORK}"
run() { # name grade atoms extra-args...
	local name=$1 grade=$2 atoms=$3
	shift 3
	bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null
	(cd "${WORK}" && "${LMP}" -k on g 1 -sf kk -in "${HERE}/in_nvt_validate.lammps" \
		-log "log_${name}.lammps" -var datafile "${SYS}/diamond_v1_${atoms}.data" \
		-var model "${MOD}/${grade}.pt2" -var dumpfile "${name}.dump" "$@" >"out_${name}.txt" 2>&1) ||
		{
			echo "${TAG} ${name}: LAMMPS failed"
			tail -5 "${WORK}/out_${name}.txt"
			return 1
		}
	echo "${TAG} ${name}: $(grep -E '^Loop time' "${WORK}/log_${name}.lammps" | tail -1)"
}
run nano_512 nano 512
run nano_10296 nano 10296
run plus_512 plus 512
run plus_10296 plus 10296
run nano_512_chain nano 512 -var nhopts "tchain 5 tloop 3 drag 0.2"
run nano_512_ramp nano 512 -var tstop 600.0 -var nhopts "tchain 1"
run nano_512_split nano 512 -var split 1
run nano_512_every1 nano 512 -var nsteps 500 -var every 1 -var dump 50
