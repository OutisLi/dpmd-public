#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# End-to-end speed and peak memory of every zoo model against the Triton
# baseline, one GPU per model in parallel. The cell size falls with the model
# size so the wider models still fit, so the rows compare paths within a
# model, not models against each other.
#
# The baseline is Triton level 2, not 3. Level 3 adds fp16x3 split-compensated
# GEMMs whose launch tables only cover swept shape keys, so on an unswept part
# it silently degrades to the level-2 float32 path and a "level 3" baseline is
# then level 2 under another name. Pinning the baseline at 2 keeps the
# comparison float32 against float32 on every part.
#
# GPUS lists the devices to spread over (default 1-6, one per model); LEVEL
# and TRITON select the paths. Per-model logs land next to LOG.
set -u

BENCH=$(cd "$(dirname "$0")" && pwd)
PYTHON="${PYTHON:-/nas/outisli/Software/miniforge3/envs/dpmd/bin/python}"
GPUS=(${GPUS:-1 2 3 4 5 6})
LEVEL="${LEVEL:-2}"
TRITON="${TRITON:-2}"
LOG="${LOG:-/tmp/zoo_bench}"

MODELS=(nano mini neo air plus pro)
# Both paths are built in one process, and the Triton side alone peaks at
# 49 GiB for the pro checkpoint at 1000 atoms.
ATOMS=(8000 8000 4096 4096 2000 1000)

pids=()
for i in "${!MODELS[@]}"; do
	model="${MODELS[$i]}"
	gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
	CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 \
		DP_TRITON_INFER="$TRITON" DP_COMPILE_INFER=1 \
		timeout 3600 "$PYTHON" "$BENCH/compare_paths.py" \
		--ckpt "$BENCH/models/checkpoints/dpa4-$model.pt" \
		--atoms "${ATOMS[$i]}" --iters 20 --level "$LEVEL" \
		>"$LOG.$model.log" 2>&1 &
	pids+=($!)
done

status=0
for i in "${!MODELS[@]}"; do
	wait "${pids[$i]}" || status=1
	model="${MODELS[$i]}"
	echo "### $model  (${ATOMS[$i]} atoms, DP_TRITON_INFER=$TRITON, DP_CUDA_INFER=$LEVEL)"
	grep -E "atoms=|speedup" "$LOG.$model.log" || {
		echo "  FAILED"
		status=1
	}
done
exit "$status"
