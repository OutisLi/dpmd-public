#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Loop time per step of the non-spin DPA4C NVT (in.lammps) and NVE
# (in_nve_bench.lammps) benchmark inputs for several LAMMPS builds on the spin device (lmp_env.sh).
# Builds alternate inside every repeat. An optional fourth field loads the
# DeePMD styles of a plugin-mode build from that plugin directory. Usage:
#   bash bench_nvt.sh REPEATS NAME:LMP:LIBDIR[:PLUGINDIR] [...]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(cd "${HERE}/.." && pwd)
REPEATS=$1
shift
source "${HERE}/lmp_env.sh"
BASE_LD=${LD_LIBRARY_PATH}
WORK=/tmp/spin_newton/bench_nvt
mkdir -p "${WORK}"
for repeat in $(seq 1 "${REPEATS}"); do
	for atoms in 512 10296; do
		if [[ ${atoms} == 512 ]]; then steps=2000; else steps=1000; fi
		for grade in nano plus; do
			for integ in nvt nve; do
				if [[ ${integ} == nvt ]]; then input=${BENCH}/in.lammps; else input=${HERE}/in_nve_bench.lammps; fi
				for spec in "$@"; do
					IFS=: read -r name lmp lib plugin <<<"${spec}"
					plugin_env=()
					if [[ -n "${plugin}" ]]; then plugin_env=(LAMMPS_PLUGIN_PATH="${plugin}"); fi
					bash "${HERE}/gpu_idle.sh" "${CUDA_VISIBLE_DEVICES}" >/dev/null
					out=${WORK}/${name}_${grade}_${atoms}_${integ}_r${repeat}.log
					(cd "${WORK}" && env "${plugin_env[@]}" LD_LIBRARY_PATH="${lib}:${BASE_LD}" "${lmp}" -k on g 1 -sf kk -in "${input}" \
						-log none -var datafile "${BENCH}/systems/diamond_v1_${atoms}.data" \
						-var model "${BENCH}/models/v20260911_trained/${grade}.pt2" \
						-var warmup $((steps / 10)) -var nsteps ${steps} >"${out}" 2>&1)
					ms=$(grep -E '^Loop time' "${out}" | tail -1 | awk '{printf "%.4f", $4 * 1000 / $9}')
					echo "${name} ${grade} ${atoms} ${integ} r${repeat} ${ms}"
				done
			done
		done
	done
done
