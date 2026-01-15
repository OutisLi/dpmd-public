# SPDX-License-Identifier: LGPL-3.0-or-later
# shellcheck shell=bash
# The H20 nodes of the vePFS cluster (/aisi-vepfs). The OutisLi fork, with the
# DeePMD plugin of the C++ prefix, is the default LAMMPS. Public-runtime scans
# select their separate installation and matching plugin explicitly. Operators
# compile for the architecture of the node alone. The fork, the C++ interface
# and the plugin are built against the CUDA-aware Open MPI of the
# conda environment openmpi, which runs as root here. The hwloc 2.13.0 of that
# Open MPI crashes in the topology export of its launcher when its Linux I/O
# discovery runs on these VMs, so the multi-rank scans skip that component too.
: "${BENCH_SOFTWARE:=/aisi-vepfs/outisli/Software}"
: "${BENCH_PYTHON:=/aisi-vepfs/outisli/miniforge3/envs/dpmd-new/bin/python}"
: "${BENCH_DEEPMD_PREFIX:=${BENCH_SOFTWARE}/deepmd-kit_cpp}"
: "${BENCH_LAMMPS_FORK:=${BENCH_SOFTWARE}/lammps/lammps-outisli}"
: "${BENCH_LMP:=${BENCH_LAMMPS_FORK}/bin/lmp}"
: "${BENCH_PLUGIN_PATH=${BENCH_DEEPMD_PREFIX}/lib/deepmd_lmp}"
: "${BENCH_MPIEXEC:=/aisi-vepfs/outisli/miniforge3/envs/openmpi/bin/mpiexec}"
: "${BENCH_MPICXX:=/aisi-vepfs/outisli/miniforge3/envs/openmpi/bin/mpicxx}"
: "${BENCH_HWLOC_COMPONENTS=-cuda,-nvml,-gl,-opencl,-rsmi,-linuxio}"
: "${BENCH_GPUS:=0 1 2 3}"
: "${BENCH_SPIN_GPU:=0}"
: "${BENCH_JOBS:=96}"
: "${BENCH_CUDA_ARCHITECTURES=native}"
: "${BENCH_NATIVE_OPTIMIZATION=}"
if [ -z "${DPA4C_CHECKPOINTS:-}" ]; then
	DPA4C_CHECKPOINTS="${BENCH_SOFTWARE}/dpa4c_checkpoints/{}.ckpt.pt"
fi
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
