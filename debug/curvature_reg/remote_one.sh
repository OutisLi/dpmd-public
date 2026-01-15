#!/bin/bash
# Usage: remote_one.sh <host> <gpu> <run> <chain: twin|h|none> <train_refresh.py flags...>
# One single-GPU run of train_refresh.py on another node reached by ssh (the management node p000),
# with its evaluation chain on the same GPU. /nas is shared, so the run directory is the one under
# debug/curvature_reg/runs. The chain selects OMat24, hydrogen, or no evaluation.
host=$1
gpu=$2
run=$3
chain=$4
shift 4
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
case "$chain" in
twin) chain_cmd="./twin_eval_chain.sh $run $gpu 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000" ;;
h) chain_cmd="./h_eval_chain.sh $run $gpu 10000 20000 30000 40000" ;;
*) chain_cmd="" ;;
esac
ssh -o BatchMode=yes "$host" "bash -s" <<REMOTE
set -e
cd $D/runs/$run
/nas/outisli/Software/miniforge3/envs/dpmd/bin/python - <<'PY' || exit 1
import json, os
sd = json.load(open("input.json"))["training"].get("save_dir", "ckpt")
if os.path.isabs(sd) or sd.startswith(".."):
    raise SystemExit(f"refusing to launch: training.save_dir={sd!r} is outside the run directory")
PY
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:\$PATH
export DP_TRITON_TRAIN=\${DP_TRITON_TRAIN:-1} DP_CUDA_TRAIN=\${DP_CUDA_TRAIN:-1} OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8 NUM_WORKERS=8
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "launched at \$(date '+%Y-%m-%d %H:%M') on \$(hostname) GPU $gpu (ssh, one GPU): train_refresh.py $*" >> launch.log
CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -u $D/train_refresh.py input.json $* > dp.log 2>&1 < /dev/null &
sleep 2
cd $D
if [ -n "$chain_cmd" ]; then
  setsid nohup ./copy_refresh_ckpts.sh $run > /dev/null 2>&1 < /dev/null &
  setsid nohup $chain_cmd > runs/$run/eval_chain.out 2>&1 < /dev/null &
fi
echo "started $run on $host GPU $gpu"
REMOTE
