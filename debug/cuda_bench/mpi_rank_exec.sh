#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# mpiexec.gforker exports PMI_RANK but not PMI_LOCAL_RANK. LAMMPS Kokkos uses
# the latter to assign one visible CUDA device to each local MPI rank.
set -eu

if [ -z "${PMI_LOCAL_RANK:-}" ]; then
	if [ -n "${PMI_RANK:-}" ]; then
		export PMI_LOCAL_RANK="${PMI_RANK}"
	elif [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
		export PMI_LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
	else
		printf '%s\n' "MPI local rank is unavailable" >&2
		exit 2
	fi
fi

exec "$@"
