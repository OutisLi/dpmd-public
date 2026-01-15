#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Exit 0 when the given GPU (default 2) runs no compute process, 1 otherwise.
set -euo pipefail
GPU=${1:-2}
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU}")
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -F "${UUID}" || true)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i "${GPU}"
if [[ -n "${APPS}" ]]; then
	echo "GPU ${GPU} busy: ${APPS}"
	exit 1
fi
echo "GPU ${GPU} idle"
