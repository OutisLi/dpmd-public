# SPDX-License-Identifier: LGPL-3.0-or-later
# shellcheck shell=bash
# The Bohrium development containers (one V100 or L20 each), whose base conda
# environment holds DeePMD-kit. The OutisLi fork, with the DeePMD plugin of
# the C++ prefix, is the only LAMMPS of the containers, and the operators
# compile for the architecture of the container alone. The containers are
# imaged for Bohrium jobs on hosts whose CPUs may differ, so host code is not
# tuned to the build host. Open MPI runs as root here.
: "${BENCH_SOFTWARE:=/root/Software}"
: "${BENCH_PYTHON:=/root/miniconda3/bin/python}"
: "${BENCH_DEEPMD_PREFIX:=${BENCH_SOFTWARE}/deepmd-kit_cpp}"
: "${BENCH_LAMMPS_FORK:=${BENCH_SOFTWARE}/lammps/lammps-outisli}"
: "${BENCH_LMP:=${BENCH_LAMMPS_FORK}/bin/lmp}"
: "${BENCH_PLUGIN_PATH=${BENCH_DEEPMD_PREFIX}/lib/deepmd_lmp}"
: "${BENCH_MPIEXEC:=/usr/bin/mpiexec}"
: "${BENCH_MPICXX:=/usr/bin/mpicxx}"
: "${BENCH_GPUS:=0}"
: "${BENCH_SPIN_GPU:=0}"
: "${BENCH_JOBS:=$(nproc)}"
: "${BENCH_CUDA_ARCHITECTURES=native}"
: "${BENCH_NATIVE_OPTIMIZATION=OFF}"
# The images do not carry the trained checkpoints; a campaign copies them
# into its scratch directory.
if [ -z "${DPA4C_CHECKPOINTS:-}" ]; then
	DPA4C_CHECKPOINTS='/root/gpu_port/checkpoints/{}.ckpt.pt'
fi
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
