#!/bin/bash
# Usage: local_h.sh <gpu> <run> <train_refresh.py flags...> -- one hydrogen-testbed run of train_refresh.py
# on one local GPU, with its evaluation chain on the same GPU (the local counterpart of remote_h.sh).
gpu=$1
run=$2
shift 2
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
cd "$D/runs/$run" || exit 1
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python - <<'PY' || exit 1
import json, os
sd = json.load(open("input.json"))["training"].get("save_dir", "ckpt")
if os.path.isabs(sd) or sd.startswith(".."):
    raise SystemExit(f"refusing to launch: training.save_dir={sd!r} is outside the run directory")
PY
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:$PATH
export DP_TRITON_TRAIN=0 DP_CUDA_TRAIN=0 OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8
echo "launched at $(date '+%Y-%m-%d %H:%M') on $(hostname) GPU $gpu (local): train_refresh.py $*" >>launch.log
CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -u "$D/train_refresh.py" input.json "$@" >dp.log 2>&1 </dev/null &
sleep 2
cd "$D"
setsid nohup ./h_eval_chain.sh "$run" "$gpu" 10000 20000 30000 40000 >"runs/$run/eval_chain.out" 2>&1 </dev/null &
echo "started $run on GPU $gpu"
