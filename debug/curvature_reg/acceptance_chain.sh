#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Acceptance chain: run the recipe to its horizon (100000) with the observer, then validate
# the listed checkpoints. Usage: acceptance_chain.sh <run> <gpu> [steps...]
set -Eeuo pipefail
run=$1
gpu=$2
shift 2
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(40000 60000 80000 100000)
cd /nas/outisli/Software/deepmd-kit/debug/curvature_reg
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python -u run_recipe.py "$run"
for step in "${steps[@]}"; do bash matrix_accuracy.sh "$gpu" "$step" "$run"; done
