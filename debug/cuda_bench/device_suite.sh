#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Validation and timing protocol of the compressed DPA4C CUDA inference on a
# device: what the work on the RTX PRO 6000 established, repeated on another
# NVIDIA part before and while it is tuned there (doc/outisli/dpa4c_cuda.md,
# sections 3 and 9).
#
# Stages, run in the order given (default: report freeze test accuracy repro
# lammps spin):
#   report     device properties the launch choices derive from, and versions
#   freeze     the frozen .pt2 models the LAMMPS stages run, compiled for this
#              device: the trained grades (freeze_dpa4c.py --checkpoint) and
#              the randomized native-spin grades (freeze_spin_random.py), in
#              ${OUT}/models
#   test       the pytest suites of the operators, against their CPU and
#              portable references on this device
#   accuracy   compressed against uncompressed fp32 and fp64 models of the
#              trained checkpoints (lowrank_deployed.py)
#   repro      two evaluations of every trained grade at three sizes agree
#              bit for bit (bench_ops.py --trained --dump / --compare)
#   lammps     whole-step LAMMPS time of every grade at three sizes
#              (bench_sizes.py)
#   spin       Newtonian spin step of the native-spin grades (bench_newton.py)
#   lr_freeze  the long-range archives of the trained Air checkpoint
#              (freeze_long_range.py: without a module, and with an LES or an
#              SOG module for clusters and for periodic systems), in
#              ${OUT}/models/long_range, and the long-range fixtures of the
#              test suites (source/tests/infer/gen_dpa4c_long_range.py)
#   lr_test    the pytest suites of the long-range operators and models, and
#              the long-range fixtures in LAMMPS against their Python
#              references on one and two ranks (check_long_range_lammps.py)
#   lr_repro   two LAMMPS trajectories of every long-range archive agree bit
#              for bit in positions and forces (bench_long_range.py --dump)
#   lr_lammps  whole-step LAMMPS time of the long-range archives at three
#              sizes (bench_long_range.py)
#
# Environment:
#   PY                  Python of the DeePMD-kit environment (default BENCH_PYTHON)
#   OPLIB               libdeepmd_op_pt.so of the build under test
#   GPU                 device index (default: the first of BENCH_GPUS)
#   OUT                 results directory (default results/device_<host>_<gpu>)
#   DPA4C_CHECKPOINTS   trained checkpoints, "{}" standing for the capitalized
#                       grade name (Nano, ..., Plus)
#   LMP, DP_LIB, PLUGIN LAMMPS executable, DeePMD C++ library directory and
#                       plugin directory of the lammps stage (default
#                       BENCH_LMP, BENCH_DP_LIB, BENCH_PLUGIN_PATH)
#   SPIN_LMP, SPIN_LIB  fork executable and library directory of the spin
#                       stage (default BENCH_SPIN_LMP, BENCH_DP_LIB), with the
#                       plugin of BENCH_SPIN_PLUGIN_PATH
#   FLAVORS             systems of the lammps stage (default: energy bcn five)
#   SPIN_SYSTEMS        systems of the spin stage (default: carbon bcn)
#   GPUS                devices the lammps stage spreads its runs over
#                       (default: GPU)
#   REPEATS             runs per point of the lammps and spin stages (default 3)
# The defaults of the paths come from the machine profile (machine.sh).
#
# Usage: OPLIB=... bash device_suite.sh [STAGE...]
# The exit status is 1 when a stage failed; every stage runs either way.
set -uo pipefail
# nvidia-smi and the CUDA runtime number the devices in the same order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${HERE}/../.." && pwd)
# shellcheck source=machine.sh
. "${HERE}/machine.sh"
PY=${PY:-${BENCH_PYTHON}}
OPLIB=${OPLIB:?set OPLIB to the libdeepmd_op_pt.so under test}
GPU=${GPU:-${BENCH_GPUS%% *}}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" | tr ' ' '_')
OUT=${OUT:-${HERE}/results/device_$(hostname -s)_${GPU_NAME}}
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(report freeze test accuracy repro lammps spin)
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 DP_CUDA_INFER=2
export DP_INTER_OP_PARALLELISM_THREADS=0 DP_INTRA_OP_PARALLELISM_THREADS=0
export BENCH_OP_LIB=${OPLIB}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_device_suite}
WITH_OPLIB=${HERE}/spin_newton/run_with_oplib.py
SIZES=(512 10296 100800)
read -r -a FLAVORS <<<"${FLAVORS:-energy bcn five}"
read -r -a SPIN_SYSTEMS <<<"${SPIN_SYSTEMS:-carbon bcn}"
GPUS=${GPUS:-${GPU}}
REPEATS=${REPEATS:-3}
GRADES=(nano mini neo air plus)
MODELS=${OUT}/models

report() {
	nvidia-smi -i "${GPU}" --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv
	"${PY}" - <<'EOF'
import torch

p = torch.cuda.get_device_properties(0)
print(f"device {p.name}, compute capability {p.major}.{p.minor}")
print(f"multiprocessors {p.multi_processor_count}, L2 {p.L2_cache_size / 2**20:.0f} MB, "
      f"memory {p.total_memory / 2**30:.0f} GB")
for name in ("max_threads_per_multi_processor", "regs_per_multiprocessor",
             "shared_memory_per_block_optin", "shared_memory_per_multiprocessor"):
    print(f"{name} {getattr(p, name, 'n/a')}")
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
EOF
	cuobjdump --list-elf "${OPLIB}" 2>/dev/null | grep -o "sm_[0-9]*" | sort -u | tr '\n' ' '
	echo "(native images of ${OPLIB})"
}

freeze() {
	cd "${HERE}" || return 1
	mkdir -p "${MODELS}/trained" "${MODELS}/spin"
	local checkpoints=${DPA4C_CHECKPOINTS}
	for grade in "${GRADES[@]}"; do
		local capitalized="$(tr '[:lower:]' '[:upper:]' <<<"${grade:0:1}")${grade:1}"
		"${PY}" "${WITH_OPLIB}" freeze_dpa4c.py --checkpoint "${checkpoints//\{\}/${capitalized}}" \
			--out "${MODELS}/trained/${grade}.pt2"
		"${PY}" "${WITH_OPLIB}" spin_newton/freeze_spin_random.py \
			--preset "dpa4c-${grade}-v20260911" --out "${MODELS}/spin/${grade}.pt2"
	done
}

# Every suite runs in a pytest process of its own, and the stage fails when
# any of them fails. The tests that check which kernels ran read the CUPTI
# activity records of the profiler, which stop arriving late in a long
# process: after the DPA4C suite, on an H20 (driver 535, CUPTI 12.9) and on a
# V100 (driver 580, CUPTI 12.6) alike.
test_suites() {
	cd "${REPO}" || return 1
	local suite status=0
	for suite in \
		source/tests/pt_expt/descriptor/test_dpa4c_cuda.py \
		source/tests/pt_expt/descriptor/test_dpa1_cuda.py \
		source/tests/pt_expt/model/test_dpa4c_graph_lower.py \
		source/tests/pt_expt/model/test_dpa4c_zbl_bridging.py; do
		"${PY}" "${WITH_OPLIB}" -m pytest -q -p no:cacheprovider -rfs -W ignore "${suite}" || status=1
	done
	return "${status}"
}

accuracy() {
	cd "${HERE}" || return 1
	"${PY}" "${WITH_OPLIB}" lowrank_deployed.py --grades nano mini neo air plus
}

repro() {
	cd "${HERE}" || return 1
	for atoms in "${SIZES[@]}"; do
		"${PY}" "${WITH_OPLIB}" bench_ops.py --trained --atoms "${atoms}" --repeats 3 \
			--dump "${OUT}/repro_${atoms}.npz" >/dev/null
		"${PY}" "${WITH_OPLIB}" bench_ops.py --trained --atoms "${atoms}" --repeats 3 \
			--compare "${OUT}/repro_${atoms}.npz" | grep "max rel dev"
	done
}

# Runs a command with the LAMMPS of the lammps stage.
with_lammps() {
	BENCH_LMP=${LMP:-${BENCH_LMP}} BENCH_DP_LIB=${DP_LIB:-${BENCH_DP_LIB}} \
		BENCH_PLUGIN_PATH=${PLUGIN-${BENCH_PLUGIN_PATH}} "$@"
}

lammps() {
	cd "${HERE}" || return 1
	local flavor
	for flavor in "${FLAVORS[@]}"; do
		with_lammps "${PY}" bench_sizes.py --label "device_${GPU_NAME}" --flavor "${flavor}" \
			--gpus "${GPUS// /,}" --repeats "${REPEATS}" --models "${MODELS}/trained"
	done
}

spin() {
	cd "${HERE}/spin_newton" || return 1
	local system
	for system in "${SPIN_SYSTEMS[@]}"; do
		"${PY}" bench_newton.py --label "device_${GPU_NAME}_${system}" --system "${system}" \
			--gpu "${GPU}" --repeats "${REPEATS}" \
			--lmp "${SPIN_LMP:-${BENCH_SPIN_LMP}}" --lib "${SPIN_LIB:-${BENCH_DP_LIB}}" \
			--models "${MODELS}/spin"
	done
}

LR_MODELS=${MODELS}/long_range

lr_freeze() {
	cd "${HERE}" || return 1
	mkdir -p "${LR_MODELS}"
	local checkpoint=${DPA4C_CHECKPOINTS//\{\}/Air}
	"${PY}" "${WITH_OPLIB}" freeze_long_range.py --checkpoint "${checkpoint}" \
		--kernel none --out "${LR_MODELS}/air.pt2"
	local kernel
	for kernel in les sog; do
		"${PY}" "${WITH_OPLIB}" freeze_long_range.py --checkpoint "${checkpoint}" \
			--kernel "${kernel}" --out "${LR_MODELS}/air_${kernel}.pt2"
		"${PY}" "${WITH_OPLIB}" freeze_long_range.py --checkpoint "${checkpoint}" \
			--kernel "${kernel}" --periodic --out "${LR_MODELS}/air_${kernel}_p.pt2"
	done
	cd "${REPO}/source/tests/infer" || return 1
	DP_VARIANT=cuda "${PY}" "${WITH_OPLIB}" gen_dpa4c_long_range.py
}

lr_test() {
	cd "${REPO}" || return 1
	local suite status=0
	for suite in \
		source/tests/pt_expt/kernels/test_long_range_sum.py \
		source/tests/pt_expt/model/test_long_range.py \
		source/tests/common/dpmodel/test_long_range.py; do
		"${PY}" "${WITH_OPLIB}" -m pytest -q -p no:cacheprovider -rfs -W ignore "${suite}" || status=1
	done
	cd "${HERE}" || return 1
	CUDA_VISIBLE_DEVICES=${GPUS// /,} with_lammps "${PY}" check_long_range_lammps.py \
		--work "${OUT}/long_range_check" || status=1
	return "${status}"
}

# Two dump runs of the six cases of bench_long_range.py at its three sizes,
# compared file by file.
lr_repro() {
	cd "${HERE}" || return 1
	local run
	for run in a b; do
		rm -rf "${OUT:?}/long_range_dump_${run}"
		with_lammps "${PY}" bench_long_range.py --label "device_${GPU_NAME}_repro_${run}" \
			--models "${LR_MODELS}" --gpus "${GPUS// /,}" --dump "${OUT}/long_range_dump_${run}"
	done
	local dump count=0 different=0
	for dump in "${OUT}"/long_range_dump_a/*.dump; do
		count=$((count + 1))
		if ! cmp -s "${dump}" "${OUT}/long_range_dump_b/$(basename "${dump}")"; then
			echo "differs: $(basename "${dump}")"
			different=1
		fi
	done
	echo "${count} dumps compared"
	[[ ${count} -eq 18 && ${different} -eq 0 ]]
}

lr_lammps() {
	cd "${HERE}" || return 1
	with_lammps "${PY}" bench_long_range.py --label "device_${GPU_NAME}" --models "${LR_MODELS}" \
		--gpus "${GPUS// /,}" --repeats "${REPEATS}"
}

failed=()
for stage in "${STAGES[@]}"; do
	echo "=== ${stage} -> ${OUT}/${stage}.log"
	case "${stage}" in
	report | freeze | accuracy | repro | lammps | spin | lr_freeze | lr_test | lr_repro | lr_lammps) run=${stage} ;;
	test) run=test_suites ;;
	*)
		echo "unknown stage: ${stage}" >&2
		exit 2
		;;
	esac
	# A stage stops at its first failing command and counts as failed; the
	# stages after it still run.
	(
		set -e
		"${run}"
	) >"${OUT}/${stage}.log" 2>&1
	status=$?
	tail -n 20 "${OUT}/${stage}.log"
	if [[ ${status} -ne 0 ]]; then
		echo "=== ${stage} failed with exit status ${status}"
		failed+=("${stage}")
	fi
done
if [[ ${#failed[@]} -gt 0 ]]; then
	echo "failed stages: ${failed[*]}"
	exit 1
fi
