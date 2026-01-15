#!/bin/bash
# Per-step anatomy of every grade and size under Nsight Systems, four runs at
# a time on the devices of the machine profile (BENCH_GPUS of machine.sh).
# Usage: bash anatomy.sh TAG [grades] [sizes]
cd "$(dirname "$0")"
tag=$1
grades=${2:-"nano mini neo air plus"}
models=${MODELS:-models/v20260911_trained}
sizes=${3:-"512 10296 100800"}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_claude_dpa4c
mkdir -p logs/anatomy_$tag
jobs=()
for size in $sizes; do
	for grade in $grades; do
		jobs+=("$grade $size")
	done
done
run_queue() {
	local gpu=$1
	shift
	for job in "$@"; do
		set -- $job
		local grade=$1 size=$2
		local warmup=200 steps=400
		if [ "$size" -ge 50000 ]; then warmup=40 steps=60; elif [ "$size" -ge 5000 ]; then warmup=100 steps=200; fi
		timeout 1800 "${BENCH_PYTHON}" lmp_nsys.py \
			--model $models/$grade.pt2 --atoms $size --gpu $gpu \
			--warmup $warmup --steps $steps --tag ${tag}_${grade}_${size} \
			>logs/anatomy_$tag/${grade}_${size}.log 2>&1
	done
}
for gpu in ${BENCH_GPUS}; do
	queue=()
	for ((i = gpu; i < ${#jobs[@]}; i += 4)); do queue+=("${jobs[$i]}"); done
	run_queue $gpu "${queue[@]}" &
done
wait
for size in $sizes; do
	for grade in $grades; do
		f=logs/anatomy_$tag/${grade}_${size}.log
		printf "%-5s %7s  %s | %s\n" $grade $size "$(grep -m1 'LAMMPS loop' $f)" "$(grep -m1 'steps between rebuilds' $f)"
	done
done
