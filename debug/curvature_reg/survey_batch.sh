#!/bin/bash
# Run the isolated-pair survey on a list of checkpoints (run:step) on one GPU of this node; output to survey_<run>_<step>.log in the run dir.
gpu=$1
shift
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
cd $D
for rs in "$@"; do
	run=${rs%%:*}
	s=${rs##*:}
	[ -f runs/$run/ema_$s.pt ] || {
		echo "missing $run $s"
		continue
	}
	CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 /nas/outisli/Software/miniforge3/envs/dpmd/bin/python dimer_survey.py runs/$run/ema_$s.pt "$run $s" 2>&1 | grep -vE "^\s*$|Warning|warn|recommended|patches installed" >runs/$run/survey_$s.log
	echo "done $run $s: $(tail -1 runs/$run/survey_$s.log)"
done
