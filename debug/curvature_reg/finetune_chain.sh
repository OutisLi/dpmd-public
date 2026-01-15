#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Hydrogen fine-tune chain: run the recipe with its observer (run_recipe.py), while the hydrogen
# evaluation (H2 dimer scan, spike/validation/MD evaluation) runs on every staged EMA checkpoint.
# Usage: finetune_chain.sh <run> <gpu> [steps...]
set -u
run=$1
gpu=$2
shift 2
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(10000 20000 30000 40000)
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
P=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
cd "$D"
(
	for s in "${steps[@]}"; do
		until [ -f "runs/$run/ema_$s.pt" ]; do sleep 120; done
		sleep 60
		for i in 1 2 3; do
			line=$(CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 $P -u dimer_scan.py "runs/$run/ema_$s.pt" --element H --out "runs/$run/dimer_$s.npz" 2>&1 | grep "dimer:")
			[ -n "$line" ] && break
			sleep 60
		done
		echo "== $run @$s dimer: ${line#*dimer: }"
		./eval_retry.sh "$run" "$gpu" "$s"
	done
) >"runs/$run/eval_chain.out" 2>&1 &
evaluator=$!
$P -u run_recipe.py "$run"
wait $evaluator
