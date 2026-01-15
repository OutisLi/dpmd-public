#!/usr/bin/env bash
# One hydrogen-testbed run of train_refresh.py on one cluster GPU, with its
# evaluation chain on the same GPU.
#   sbatch --job-name=<run> slurm_h.sh <run> <train_refresh.py flags...>
# The run directory runs/<run> must exist with its input.json.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=125000M
#SBATCH --output=slurm-%j.out
set -Eeuo pipefail
run=$1
shift
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
DP_ENV=/nas/outisli/Software/miniforge3/envs/dpmd
export PATH="$DP_ENV/bin:$PATH"
export OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8
cd "$D/runs/$run"
# Refuse an input whose checkpoints would land outside the run directory (an absolute
# save_dir copied from a production input would let this run prune that run's checkpoints).
python - <<'PY' || exit 1
import json, os
sd = json.load(open("input.json"))["training"].get("save_dir", "ckpt")
if os.path.isabs(sd) or sd.startswith(".."):
    raise SystemExit(f"refusing to launch: training.save_dir={sd!r} is outside the run directory")
PY
echo "launched at $(date '+%Y-%m-%d %H:%M') on $(hostname) (Slurm job $SLURM_JOB_ID): train_refresh.py $*" >>launch.log
python -u "$D/train_refresh.py" input.json "$@" >dp.log 2>&1 &
train=$!
cd "$D"
./h_eval_chain.sh "$run" 0 ${EVAL_STEPS:-10000 20000 30000 40000} >"runs/$run/eval_chain.out" 2>&1
wait $train
echo "finished at $(date '+%Y-%m-%d %H:%M')" >>"runs/$run/launch.log"
