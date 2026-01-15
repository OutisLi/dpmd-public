# SPDX-License-Identifier: LGPL-3.0-or-later
# shellcheck shell=bash
# The RTX PRO 6000 workstation and the Pro6000 Slurm nodes, which share /nas.
# Plain models are timed on the public LAMMPS release with the DeePMD plugin,
# so that a comparison measures the deepmd-kit side alone
# (doc/outisli/dpa4c_cuda.md, section 2.3). The OutisLi fork loads a DeePMD
# plugin of its own. Both LAMMPS builds and the C++ interface use the Open MPI
# of the DeePMD-kit environment, whose CUDA support the multi-rank scans enable.
# GPUs 4-7 of the workstation run the user's own jobs.
: "${BENCH_SOFTWARE:=/nas/outisli/Software}"
: "${BENCH_PYTHON:=${BENCH_SOFTWARE}/miniforge3/envs/dpmd/bin/python}"
: "${BENCH_DEEPMD_PREFIX:=${BENCH_SOFTWARE}/deepmd-kit_cpp}"
: "${BENCH_LAMMPS_FORK:=${BENCH_SOFTWARE}/lammps/lammps-outisli}"
: "${BENCH_LMP:=${BENCH_SOFTWARE}/lammps/lammps-patch_4Jul2026/bin/lmp}"
: "${BENCH_PLUGIN_PATH=${BENCH_DEEPMD_PREFIX}/lib/deepmd_lmp}"
: "${BENCH_SPIN_PLUGIN_PATH=${BENCH_LAMMPS_FORK}/lib/deepmd_lmp}"
: "${BENCH_MPIEXEC:=${BENCH_SOFTWARE}/miniforge3/envs/dpmd/bin/mpiexec}"
: "${BENCH_MPICXX:=${BENCH_SOFTWARE}/miniforge3/envs/dpmd/bin/mpicxx}"
: "${BENCH_GPUS:=0 1 2 3}"
: "${BENCH_SPIN_GPU:=2}"
: "${BENCH_JOBS:=64}"
: "${BENCH_CUDA_ARCHITECTURES=}"
: "${BENCH_NATIVE_OPTIMIZATION=}"
if [ -z "${DPA4C_CHECKPOINTS:-}" ]; then
	DPA4C_CHECKPOINTS='/nas/outisli/Research/aisi_workspace/MatPES/2025.1/DPA4C/{}/model_ema.ckpt.pt'
fi
