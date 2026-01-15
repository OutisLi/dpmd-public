#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Sweep the launch-tile constants of the fused SO(2) convolution.
#
# Each configuration is compiled into its own operator library and timed on its
# own GPU, so a sweep costs one build per point and one standalone run. The
# model graph has the final word; this only narrows the candidates.
set -u

ROOT=/nas/outisli/Software/deepmd-kit
PYTHON=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
BENCH="$ROOT/debug/cuda_bench"
MODELS="${MODELS:-mini}"

build_one() {
	local name="$1" flags="$2" dir="/tmp/tilesweep/$name"
	mkdir -p "$dir"
	cmake -S "$ROOT/source" -B "$dir" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DUSE_CUDA_TOOLKIT=TRUE \
		-DENABLE_PYTORCH=ON \
		-DBUILD_PY_IF=TRUE \
		-DBUILD_CPP_IF=FALSE \
		-DENABLE_NATIVE_OPTIMIZATION=TRUE \
		-DCMAKE_PREFIX_PATH=/nas/outisli/Software/miniforge3/envs/dpmd/lib/python3.13/site-packages/torch \
		-DPYTHON_EXECUTABLE="$PYTHON" \
		-DCMAKE_CUDA_FLAGS="$flags" \
		>"$dir/configure.log" 2>&1 || {
		echo "$name: configure failed"
		tail -5 "$dir/configure.log"
		return 1
	}
	ninja -C "$dir" -j 64 deepmd_op_pt >"$dir/build.log" 2>&1 ||
		{
			echo "$name: build failed"
			tail -5 "$dir/build.log"
			return 1
		}
	echo "$dir/op/pt/libdeepmd_op_pt.so"
}

run_one() {
	local name="$1" lib="$2" gpu="$3"
	DP_OP_LIB="$lib" CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 \
		"$PYTHON" "$BENCH/check_conv.py" --models $MODELS --skip-check --nodes 8000 \
		2>&1 | grep -E "^\s+\[" | sed "s/^/[$name] /"
}

main() {
	local gpu=1
	for spec in "$@"; do
		local name="${spec%%:*}" flags="${spec#*:}"
		local lib
		lib=$(build_one "$name" "$flags") || continue
		run_one "$name" "$lib" "$gpu"
		gpu=$(((gpu % 7) + 1))
	done
}

main "$@"
