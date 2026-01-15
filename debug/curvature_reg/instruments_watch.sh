#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Run the all-element homonuclear scan and the compressed-trimer probe on every new
# EMA checkpoint of a run (and the trimer probe on the matching raw checkpoint when it
# still exists), until the run's observer ends with COMPLETE.
# Usage: instruments_watch.sh <run> <gpu>
set -u
run=$1
gpu=$2
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:$PATH OMP_NUM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0
pairs=$(
	python - <<'PY'
import json; tm = json.load(open('/nas/outisli/Software/deepmd-kit/debug/curvature_reg/runs/neo_seed_r3_gated/input.json'))['model']['type_map'][:94]
print(','.join(f'{e}-{e}' for e in tm))
PY
)
cd "$D" || exit 1
mkdir -p "runs/$run/homo_scan"
log="runs/$run/instruments.log"
echo "instruments watcher started $(date -Is) on $(hostname) GPU $gpu" >>"$log"
while true; do
	for ema in $(ls "runs/$run"/ema_*.pt 2>/dev/null | sed 's/.*ema_\([0-9]*\)\.pt/\1/' | sort -n); do
		if [ ! -f "runs/$run/homo_scan/ema_$ema.json" ]; then
			CUDA_VISIBLE_DEVICES=$gpu python -u dimer_survey.py "runs/$run/ema_$ema.pt" "homo $run ema_$ema" "$pairs" --spacing 0.01 --out "runs/$run/homo_scan/ema_$ema.npz" >"runs/$run/homo_scan/ema_$ema.log" 2>&1
			echo "$(date -Is) homo_scan ema_$ema exit $? $(tail -1 "runs/$run/homo_scan/ema_$ema.log" | cut -c1-120)" >>"$log"
		fi
		if ! grep -q "\"label\": \"$run ema $ema\"" "runs/$run/trimer_probe.jsonl" 2>/dev/null; then
			CUDA_VISIBLE_DEVICES=$gpu python -u diagnose/trimer_probe.py "runs/$run/ema_$ema.pt" "$run ema $ema" --out "runs/$run/trimer_probe.jsonl" >/dev/null 2>&1
			echo "$(date -Is) trimer_probe ema $ema exit $?" >>"$log"
		fi
		raw="runs/$run/ckpt/model.ckpt-$ema.pt"
		if [ -f "$raw" ] && ! grep -q "\"label\": \"$run raw $ema\"" "runs/$run/trimer_probe.jsonl" 2>/dev/null; then
			CUDA_VISIBLE_DEVICES=$gpu python -u diagnose/trimer_probe.py "$raw" "$run raw $ema" --out "runs/$run/trimer_probe.jsonl" >/dev/null 2>&1
			echo "$(date -Is) trimer_probe raw $ema exit $?" >>"$log"
		fi
	done
	# A continued run keeps its screen's COMPLETE line; only the observer's last line ends the watch.
	tail -1 "runs/$run/watch.log" 2>/dev/null | grep -q "COMPLETE verified" && {
		echo "observer complete $(date -Is); watcher exits" >>"$log"
		break
	}
	sleep 120
done
