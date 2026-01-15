# SPDX-License-Identifier: LGPL-3.0-or-later
# shellcheck shell=bash
# Paths and device facts of the machine the benchmark harness runs on.
#
# Every shell script of debug/cuda_bench sources this file with bash, as does
# debug/install.sh; the Python scripts load it through machine.load(). The
# profile machines/<name>.sh sets the values that differ between machines:
# DP_BENCH_MACHINE names it, or the software root that exists on the machine
# selects it. This file derives the rest. A variable that the caller has set
# keeps its value.
#
#   BENCH_MACHINE             profile name
#   BENCH_SOFTWARE            software root of the machine
#   BENCH_PYTHON              Python of the DeePMD-kit environment
#   BENCH_DEEPMD_SRC          this DeePMD-kit checkout
#   BENCH_DEEPMD_PREFIX       install prefix of the C++ interface
#   BENCH_DP_LIB              its lib directory
#   BENCH_TORCH_LIB           lib directory of the torch of BENCH_PYTHON
#   BENCH_LAMMPS_FORK         the OutisLi LAMMPS fork: source tree and install prefix
#   BENCH_SPIN_LMP            its executable
#   BENCH_SPIN_PLUGIN_PATH    DeePMD plugin directory of BENCH_SPIN_LMP (default BENCH_PLUGIN_PATH)
#   BENCH_LMP                 LAMMPS executable of the plain-model benchmarks
#   BENCH_PLUGIN_PATH         DeePMD plugin directory of BENCH_LMP; empty for built-in styles
#   BENCH_MPIEXEC             MPI launcher of the multi-rank scans
#   BENCH_MPICXX              MPI C++ compiler of the LAMMPS build
#   BENCH_HWLOC_COMPONENTS    HWLOC_COMPONENTS of the multi-rank scans: the hwloc discovery
#                             components that the MPI launcher and the ranks skip (default
#                             the GPU plugins, -cuda,-nvml,-gl,-opencl,-rsmi)
#   BENCH_GPUS                devices the harness may use, separated by spaces
#   BENCH_SPIN_GPU            device of the single-device spin scripts
#   BENCH_JOBS                build parallelism
#   BENCH_CUDA_ARCH           compute capability of the first device of BENCH_GPUS, without the dot
#   BENCH_KOKKOS_ARCH         Kokkos architecture of that capability (Kokkos_ARCH_<name>)
#   BENCH_CUDA_ARCHITECTURES  CMAKE_CUDA_ARCHITECTURES of the operator builds: empty for the
#                             project default; a profile value "native" stands for BENCH_CUDA_ARCH
#   BENCH_NATIVE_OPTIMIZATION ENABLE_NATIVE_OPTIMIZATION of the C++ builds (ON or OFF): OFF
#                             where an image of the machine runs on other host CPUs; empty
#                             keeps what the build tree holds
#   DPA4C_CHECKPOINTS         trained DPA4C checkpoints, "{}" standing for the capitalized grade
#   LD_LIBRARY_PATH           led by the CUDA libraries of the torch wheel when torch was built
#                             for a newer CUDA than the toolkit in CUDA_HOME (default
#                             /usr/local/cuda), and otherwise unchanged
#
# A new machine adds a profile under machines/ and its line in bench_profile.

_bench_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Profile of this machine: the one DP_BENCH_MACHINE names, or the first whose
# software root exists.
bench_profile() {
	if [ -n "${DP_BENCH_MACHINE:-}" ]; then
		echo "${DP_BENCH_MACHINE}"
	elif [ -d /nas/outisli/Software ]; then
		echo nas
	elif [ -d /aisi-vepfs/outisli/Software ]; then
		echo vepfs
	elif [ -d /root/Software ] && [ -x /root/miniconda3/bin/python ]; then
		echo bohrium
	fi
}

# Kokkos architecture name of a compute capability written without the dot.
bench_kokkos_arch() {
	case "$1" in
	70) echo VOLTA70 ;;
	72) echo VOLTA72 ;;
	75) echo TURING75 ;;
	80) echo AMPERE80 ;;
	86) echo AMPERE86 ;;
	87) echo AMPERE87 ;;
	89) echo ADA89 ;;
	90) echo HOPPER90 ;;
	100) echo BLACKWELL100 ;;
	103) echo BLACKWELL103 ;;
	120) echo BLACKWELL120 ;;
	121) echo BLACKWELL121 ;;
	esac
}

BENCH_MACHINE=$(bench_profile)
if [ -z "${BENCH_MACHINE}" ] || [ ! -f "${_bench_dir}/machines/${BENCH_MACHINE}.sh" ]; then
	echo "machine.sh: no profile matches this machine; set DP_BENCH_MACHINE to the name of one of ${_bench_dir}/machines/*.sh" >&2
	return 1
fi
# shellcheck source=/dev/null
. "${_bench_dir}/machines/${BENCH_MACHINE}.sh"
: "${BENCH_DEEPMD_SRC:=$(cd "${_bench_dir}/../.." && pwd)}"
: "${BENCH_DP_LIB:=${BENCH_DEEPMD_PREFIX}/lib}"
: "${BENCH_SPIN_LMP:=${BENCH_LAMMPS_FORK}/bin/lmp}"
: "${BENCH_SPIN_PLUGIN_PATH=${BENCH_PLUGIN_PATH}}"
: "${BENCH_HWLOC_COMPONENTS=-cuda,-nvml,-gl,-opencl,-rsmi}"
: "${BENCH_TORCH_LIB:=$("${BENCH_PYTHON}" -I -c 'import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin), "lib"))')}"
# A Python process loads the CUDA libraries of the torch wheel before anything
# else, while a LAMMPS process binds the CUDA runtime of the toolkit first,
# through LD_LIBRARY_PATH, when it links the operators, before torch loads. On
# a machine whose torch was built for a newer CUDA than the toolkit, torch
# would then run on an older runtime than it was built for, and the libraries
# of the wheel would bind the older toolkit copies of their dependencies
# (libnvJitLink of libcusparse), which lack their symbols. The libraries of
# the wheel then lead the search path, so that every process runs on the newer
# of the two runtimes, as a Python process does.
_bench_torch_cuda=$(sed -n "s/^cuda[^=]*= *'\([0-9.]*\)'.*/\1/p" "$(dirname "${BENCH_TORCH_LIB}")/version.py")
_bench_toolkit_cuda=$("${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
if [ -n "${_bench_torch_cuda}" ] && [ -n "${_bench_toolkit_cuda}" ] &&
	[ "$(printf '%s\n' "${_bench_torch_cuda}" "${_bench_toolkit_cuda}" | sort -V | tail -n 1)" != "${_bench_toolkit_cuda}" ]; then
	for _bench_lib in "$(dirname "$(dirname "${BENCH_TORCH_LIB}")")"/nvidia/*/lib; do
		if [ -d "${_bench_lib}" ]; then
			case ":${LD_LIBRARY_PATH:-}:" in
			*":${_bench_lib}:"*) ;;
			*) LD_LIBRARY_PATH=${_bench_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ;;
			esac
		fi
	done
	export LD_LIBRARY_PATH
fi
if [ -z "${BENCH_CUDA_ARCH:-}" ] && command -v nvidia-smi >/dev/null; then
	BENCH_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${BENCH_GPUS%% *}" | tr -d '.[:space:]')
fi
: "${BENCH_KOKKOS_ARCH:=$(bench_kokkos_arch "${BENCH_CUDA_ARCH:-}")}"
if [ "${BENCH_CUDA_ARCHITECTURES:-}" = native ]; then
	BENCH_CUDA_ARCHITECTURES=${BENCH_CUDA_ARCH:-}
fi
# The harness puts BENCH_DP_LIB on LD_LIBRARY_PATH, which the loader searches
# before the RUNPATH of a LAMMPS executable: a LAMMPS library installed into
# the C++ prefix would replace the one of the LAMMPS under test.
for _bench_lib in "${BENCH_DP_LIB}"/liblammps*; do
	if [ -e "${_bench_lib}" ]; then
		echo "machine.sh: ${_bench_lib} would take the place of the library of every LAMMPS the harness runs; move the LAMMPS install out of ${BENCH_DEEPMD_PREFIX}" >&2
		return 1
	fi
done
export BENCH_MACHINE BENCH_SOFTWARE BENCH_PYTHON BENCH_DEEPMD_SRC BENCH_DEEPMD_PREFIX BENCH_DP_LIB \
	BENCH_TORCH_LIB BENCH_LAMMPS_FORK BENCH_SPIN_LMP BENCH_SPIN_PLUGIN_PATH BENCH_LMP BENCH_PLUGIN_PATH \
	BENCH_MPIEXEC BENCH_MPICXX BENCH_HWLOC_COMPONENTS BENCH_GPUS BENCH_SPIN_GPU BENCH_JOBS BENCH_CUDA_ARCH \
	BENCH_KOKKOS_ARCH BENCH_CUDA_ARCHITECTURES BENCH_NATIVE_OPTIMIZATION DPA4C_CHECKPOINTS
unset _bench_dir _bench_lib _bench_torch_cuda _bench_toolkit_cuda
