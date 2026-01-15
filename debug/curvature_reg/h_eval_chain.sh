#!/bin/bash
# Usage: h_eval_chain.sh <run> <gpu> <step>... — the hydrogen-testbed evaluation
# loop for an already-training run: wait for each EMA checkpoint, copy it, run
# the H2 dimer scan and the spike/validation/MD evaluation (eval_retry.sh).
r=$1
g=$2
shift 2
P=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
D=$(cd "$(dirname "$0")" && pwd)
cd $D
for s in "$@"; do
	until [ -f runs/$r/ckpt/model_ema.ckpt-$s.pt ]; do sleep 120; done
	sleep 30
	cp runs/$r/ckpt/model_ema.ckpt-$s.pt runs/$r/ema_$s.pt
	sleep 60
	for i in 1 2 3; do
		line=$(CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 $P -u dimer_scan.py runs/$r/ema_$s.pt --element H --out runs/$r/dimer_$s.npz 2>&1 | grep "dimer:")
		[ -n "$line" ] && break
		sleep 60
	done
	echo "== $r @$s dimer: ${line#*dimer: }"
	./eval_retry.sh $r $g $s
done
