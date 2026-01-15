#!/bin/bash
# Usage: eval_retry.sh <run> <gpu> <step> [broad [frames.npz]]
# Evaluate one copied EMA checkpoint, retrying the launch when the shared library fails to load (a transient NFS
# condition that follows large checkpoint copies). With "broad", run the broad search of eval_model.py instead of
# the hydrogen spike evaluation, around the frames of the given file (default: the hydrogen held-out set).
r=$1
g=$2
s=$3
mode=${4:-spike}
frames=${5:-heldout_h.npz}
P=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
D=$(cd "$(dirname "$0")" && pwd)
if [ "$mode" = broad ]; then
	log=$D/runs/$r/broad_$s.log
	cmd="$P -u $D/eval_model.py $D/runs/$r/ema_$s.pt --frames $D/$frames --n-acc 50 --n-curv 100 --n-search 40 --n-md 0 --out $D/runs/$r/broad_$s.json"
	pat="curvature\[perturbed\]\|spikes"
else
	log=$D/runs/$r/spike_eval_$s.log
	cmd="$P -u $D/h_spike_eval.py $D/runs/$r/ema_$s.pt --n-md 12 --md-steps 2500 --out $D/runs/$r/spike_eval_$s.json"
	pat="pathological\|hill\|validation\|MD "
fi
for attempt in 1 2 3 4; do
	CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 $cmd >$log 2>&1
	grep -q "^saved" $log && break
	sleep 90
done
echo "== $r @$s $mode (attempt $attempt)"
grep "$pat" $log | cut -c1-230
