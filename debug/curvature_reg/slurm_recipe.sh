#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=125000M
#SBATCH --output=/nas/outisli/Software/deepmd-kit/debug/curvature_reg/slurm-%j.out
set -Eeuo pipefail
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:$PATH
export DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1 DP_TUNE_TRAIN=0
export OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8 NUM_WORKERS=8
export TORCHINDUCTOR_COMPILE_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /nas/outisli/Software/deepmd-kit/debug/curvature_reg
python -u run_recipe.py "$1"
