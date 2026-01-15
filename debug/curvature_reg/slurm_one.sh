#!/usr/bin/env bash
# One single-GPU run of train_refresh.py on a cluster node, with its checkpoint copier and
# evaluation chain on the same GPU.
#   sbatch --job-name=<run> slurm_one.sh <run> <chain: twin|h|none> <train_refresh.py flags...>
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
chain=$2
shift 2
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
DP_ENV=/nas/outisli/Software/miniforge3/envs/dpmd
export PATH="$DP_ENV/bin:$PATH"
export DP_TRITON_TRAIN=${DP_TRITON_TRAIN:-1} DP_CUDA_TRAIN=${DP_CUDA_TRAIN:-1} DP_TUNE_TRAIN=0 NUM_WORKERS=8
export DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$D/runs/$run"
# Refuse an input whose checkpoints would land outside the run directory.
python - <<'PY' || exit 1
import json, os
sd = json.load(open("input.json"))["training"].get("save_dir", "ckpt")
if os.path.isabs(sd) or sd.startswith(".."):
    raise SystemExit(f"refusing to launch: training.save_dir={sd!r} is outside the run directory")
PY
echo "launched at $(date '+%Y-%m-%d %H:%M') on $(hostname) (Slurm job $SLURM_JOB_ID, one GPU): train_refresh.py $*" >>launch.log
python -u "$D/train_refresh.py" input.json "$@" >dp.log 2>&1 &
train=$!
cd "$D"
case "$chain" in
twin)
	./copy_refresh_ckpts.sh "$run" >/dev/null 2>&1 &
	./twin_eval_chain.sh "$run" 0 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000 >"runs/$run/twin_eval_chain.out" 2>&1 &
	;;
h)
	./h_eval_chain.sh "$run" 0 10000 20000 30000 40000 >"runs/$run/eval_chain.out" 2>&1 &
	;;
esac
wait $train
echo "finished at $(date '+%Y-%m-%d %H:%M')" >>"runs/$run/launch.log"
# The evaluation chain trails the training by one checkpoint, so the job must outlive the trainer or
# the last checkpoint is never read. Wait for the chain's own process before the allocation ends.
wait
