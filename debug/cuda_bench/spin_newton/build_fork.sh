#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Configure, build and install the OutisLi LAMMPS fork for the GPU of this
# machine, with the paths of the machine profile (machine.sh). The fork loads
# the DeePMD styles from the plugin that the DeePMD-kit C++ interface builds
# against it (doc/outisli/install.md, section 3.1), so a change of the
# DeePMD-kit sources needs that plugin rebuilt, not the fork.
#
# Usage: bash build_fork.sh [configure|build|install|all]   (default: all)
#
# BUILD_DIR and INSTALL_ROOT select another build of the same fork without
# touching the installed one:
#   BUILD_DIR=/tmp/lammps_fork_b/build INSTALL_ROOT=/tmp/lammps_fork_b \
#   bash build_fork.sh all
#
# The MPI library of BENCH_MPICXX enters the RUNPATH of the installed
# executable and library (LAMMPS_INSTALL_RPATH), and the plugin must be built
# against the same MPI library.
set -euo pipefail

# shellcheck source=../machine.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/machine.sh"
SOURCE_ROOT=${BENCH_LAMMPS_FORK}
BUILD_DIR=${BUILD_DIR:-${SOURCE_ROOT}/build_kk}
INSTALL_ROOT=${INSTALL_ROOT:-${SOURCE_ROOT}}
JOBS=${JOBS:-${BENCH_JOBS}}
STEP=${1:-all}

configure() {
	cmake -S "${SOURCE_ROOT}/cmake" -B "${BUILD_DIR}" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=ON \
		-DBUILD_MPI=ON \
		-DBUILD_OMP=ON \
		-DPKG_SPIN=ON \
		-DPKG_KSPACE=ON \
		-DPKG_MANYBODY=ON \
		-DPKG_MEAM=ON \
		-DPKG_PLUGIN=ON \
		-DPKG_EXTRA-FIX=ON \
		-DPKG_KOKKOS=ON \
		-DKokkos_ENABLE_SERIAL=ON \
		-DKokkos_ENABLE_CUDA=ON \
		-DKokkos_ARCH_"${BENCH_KOKKOS_ARCH}"=ON \
		-DCMAKE_CXX_COMPILER="${SOURCE_ROOT}/lib/kokkos/bin/nvcc_wrapper" \
		-DMPI_CXX_COMPILER="${BENCH_MPICXX}" \
		-DCMAKE_INSTALL_PREFIX="${INSTALL_ROOT}" \
		-DLAMMPS_INSTALL_RPATH=ON
}

build() {
	cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
}

install() {
	cmake --install "${BUILD_DIR}"
}

case "${STEP}" in
configure) configure ;;
build) build ;;
install) install ;;
all) configure && build && install ;;
*)
	echo "unknown step: ${STEP}" >&2
	exit 2
	;;
esac
