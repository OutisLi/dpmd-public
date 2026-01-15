#!/bin/bash
# Usage: supervise_all.sh -- keep every force-spike cell alive to its decision horizon.
#
# Every three minutes each run directory is checked. A run counts as alive if a Slurm job carries its
# name or a process on the management node has its directory as working directory. A run that is gone
# and short of its horizon is restarted from its newest checkpoint, on the resource it was launched on,
# with the flags recorded in its own launch.log. A run that has reached the horizon is left alone.
#
# The restart turns the fused CUDA and Triton training kernels on, since they hold far less memory than
# the eager path and the management-node cells were exhausting a 96 GiB card without them. Every cell may
# use them: the kernels take the radial features as an argument and sit downstream of the radial basis,
# which is the only thing the single-envelope patch replaces. A run that carries a STOPPED.txt was ended
# deliberately and is never restarted.
D=/nas/outisli/Software/deepmd-kit/debug/curvature_reg
HOST=p000
cd "$D" || exit 1

horizon() { case "$1" in h_*) echo 40000 ;; *) echo 60000 ;; esac }
chain_of() { case "$1" in h_*) echo h ;; *) echo twin ;; esac }

flags_of() {
	grep -hoE "train_refresh\.py .*" "runs/$1/launch.log" 2>/dev/null | tail -1 |
		sed -e 's/^train_refresh\.py //' -e 's/ --restart [^ ]*//' -e 's/ input\.json//'
}

last_ckpt() {
	ls "runs/$1/ckpt" 2>/dev/null | grep -oE 'model\.ckpt-[0-9]+\.pt' | grep -oE '[0-9]+' | sort -n | tail -1
}

while true; do
	slurm_names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
	remote_dirs=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
		"for p in \$(ls /proc | grep -E '^[0-9]+\$'); do readlink /proc/\$p/cwd 2>/dev/null; done | grep '$D/runs/' | sed 's#.*/runs/##' | sort -u" 2>/dev/null)
	for path in runs/neo_1gpu_*/ runs/h_1gpu_*/ runs/h_ft_1gpu_*/; do
		run=$(basename "$path")
		if [ ! -f "runs/$run/launch.log" ]; then
			echo "$(date '+%H:%M') $run has an input but was never launched"
			continue
		fi
		[ -f "runs/$run/STOPPED.txt" ] && continue
		printf '%s\n' "$slurm_names" | grep -qx "$run" && continue
		printf '%s\n' "$remote_dirs" | grep -qx "$run" && continue
		step=$(last_ckpt "$run")
		step=${step:-0}
		[ "$step" -ge "$(horizon "$run")" ] && continue
		flags=$(flags_of "$run")
		[ -n "$flags" ] || continue
		accel="DP_CUDA_TRAIN=1 DP_TRITON_TRAIN=1"
		restart=""
		[ "$step" -gt 0 ] && restart="--restart ckpt/model.ckpt-$step.pt"
		if grep -q "Slurm job" "runs/$run/launch.log"; then
			echo "$(date '+%H:%M') $run gone at step $step, resubmitting to the cluster"
			env $accel sbatch --export=ALL --job-name="$run" slurm_one.sh "$run" "$(chain_of "$run")" $flags $restart >/dev/null 2>&1
		else
			gpu=$(grep -hoE "GPU [0-7]" "runs/$run/launch.log" | tail -1 | grep -oE "[0-7]")
			[ -n "$gpu" ] || continue
			echo "$(date '+%H:%M') $run gone at step $step, restarting on $HOST GPU $gpu"
			ssh -o BatchMode=yes "$HOST" "bash -s" <<REMOTE
cd $D/runs/$run
export PATH=/nas/outisli/Software/miniforge3/envs/dpmd/bin:\$PATH
export $accel OMP_NUM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=8 NUM_WORKERS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "restarted at \$(date '+%Y-%m-%d %H:%M') from step $step on GPU $gpu: train_refresh.py $flags" >> launch.log
CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -u $D/train_refresh.py input.json $flags $restart >> dp.log 2>&1 < /dev/null &
REMOTE
		fi
	done
	sleep 180
done
