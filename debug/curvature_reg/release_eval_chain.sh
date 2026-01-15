#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Evaluation chain of one release checkpoint staged as runs/<run>/ema_<step>.pt (and
# runs/<run>/live_<step>.pt for the raw weights) with the run's input.json beside it:
# the 24-pair isolated-pair survey, the all-element homonuclear scan, the compressed-trimer
# probe, the complete sub001 validation scan, the held-out spike/MD evaluation, and the
# 24-pair survey of the raw weights. Usage: release_eval_chain.sh <run> <step> <gpu>
set -u
run=$1
step=$2
gpu=$3
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
P=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0
cd "$D" || exit 1
pairs=$($P - "runs/$run/input.json" <<'PY'
import json, sys
tm = json.load(open(sys.argv[1]))["model"]["type_map"][:94]
print(",".join(f"{e}-{e}" for e in tm))
PY
)
log="runs/$run/chain_$step.log"
echo "$(date -Is) chain start on $(hostname) GPU $gpu" >>"$log"
$P -u dimer_survey.py "runs/$run/ema_$step.pt" "$run ema $step" --out "runs/$run/survey_full_$step.npz" >"runs/$run/survey_full_$step.log" 2>&1
echo "$(date -Is) survey exit $? $(tail -1 "runs/$run/survey_full_$step.log" | cut -c1-160)" >>"$log"
$P -u dimer_survey.py "runs/$run/ema_$step.pt" "homo $run ema_$step" "$pairs" --spacing 0.01 --out "runs/$run/homo_scan/ema_$step.npz" >"runs/$run/homo_scan/ema_$step.log" 2>&1
echo "$(date -Is) homo_scan exit $? $(tail -1 "runs/$run/homo_scan/ema_$step.log" | cut -c1-160)" >>"$log"
$P -u diagnose/trimer_probe.py "runs/$run/ema_$step.pt" "$run ema $step" --out "runs/$run/trimer_probe.jsonl" >"runs/$run/trimer_probe_$step.log" 2>&1
echo "$(date -Is) trimer_probe exit $? $(tail -1 "runs/$run/trimer_probe_$step.log" | cut -c1-160)" >>"$log"
bash matrix_accuracy.sh "$gpu" "$step" "$run" >>"$log" 2>&1
echo "$(date -Is) matrix_accuracy exit $?" >>"$log"
$P -u eval_model.py "runs/$run/ema_$step.pt" --out "runs/$run/eval_$step.json" >"runs/$run/eval_$step.log" 2>&1
echo "$(date -Is) eval_model exit $? $(grep -E '^(accuracy|spikes|MD)' "runs/$run/eval_$step.log" | tr '\n' ' ' | cut -c1-300)" >>"$log"
$P -u dimer_survey.py "runs/$run/live_$step.pt" "$run live $step" --out "runs/$run/survey_live_$step.npz" >"runs/$run/survey_live_$step.log" 2>&1
echo "$(date -Is) survey(live) exit $? $(tail -1 "runs/$run/survey_live_$step.log" | cut -c1-160)" >>"$log"
touch "runs/$run/chain_$step.done"
echo "$(date -Is) chain done" >>"$log"
