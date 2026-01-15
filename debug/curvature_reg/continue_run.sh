#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Continue a finished screen from its last raw checkpoint to a later horizon on one GPU,
# with the same observer (surveys of every saved checkpoint) and full validation at the
# listed steps once the observer completes.
# Usage: continue_run.sh <run> <gpu> <from_step> <horizon> <validation steps...>
set -Eeuo pipefail
run=$1
gpu=$2
from=$3
horizon=$4
shift 4
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
python=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:$PATH
export DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1 DP_TUNE_TRAIN=0
export OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0 NUM_WORKERS=8
export TORCHINDUCTOR_COMPILE_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$D/runs/$run"
[ -f "ckpt/model.ckpt-$from.pt" ]
flags=$($python -c "import json; print(' '.join(json.load(open('experiment.json'))['train_flags']))")
$python - "$horizon" <<'PY'
import json, sys
d = json.load(open("experiment.json")); d["horizon_step"] = int(sys.argv[1]); d["continued_from_step"] = d.get("continued_from_step", [])
json.dump(d, open("experiment.json", "w"), indent=2)
PY
echo "continued at $(date -Is) on $(hostname) GPU $gpu from ckpt/model.ckpt-$from.pt to horizon $horizon: train_refresh.py $flags --restart ckpt/model.ckpt-$from.pt" >>launch.log
CUDA_VISIBLE_DEVICES=$gpu setsid nohup $python -u "$D/train_refresh.py" input.json $flags --restart "ckpt/model.ckpt-$from.pt" >>dp.log 2>&1 </dev/null &
trainer=$!
sleep 3
cd "$D"
CUDA_VISIBLE_DEVICES=$gpu setsid nohup $python -u watch_run.py "$D/runs/$run" --trainer "$trainer" --gpu "$gpu" >>"runs/$run/watch.log" 2>&1 </dev/null &
observer=$!
echo "trainer $trainer observer $observer"
wait $observer
for step in "$@"; do bash matrix_accuracy.sh "$gpu" "$step" "$run"; done
