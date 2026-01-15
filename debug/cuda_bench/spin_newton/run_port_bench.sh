#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Newtonian NVE spin step of all five grades at 512, 10,296 and 100,800 atoms
# with the models of models/spin_random_st and the operator library OPLIB on
# the spin device of the machine profile (bench_newton.py, three repeats each):
#   port_host    the installed build with the host fix nve/spin (hostfix 1)
#   port_device  the installed build with nve/spin/kk, selected by -sf kk
#   port_ref     fixed moments with the device integrator nve/kk (newton 0)
# Usage: bash run_port_bench.sh [REPEATS]   (default 3)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPEATS=${1:-3}
OPLIB=${OPLIB:-/tmp/dpSR/lib}
MODELS=$(cd "${HERE}/../models/spin_random_st" && pwd)
# shellcheck source=../machine.sh
. "${HERE}/../machine.sh"
PYTHON=${BENCH_PYTHON}
cd "${HERE}"
common=(--repeats "${REPEATS}" --lib "${OPLIB}" --models "${MODELS}")
"${PYTHON}" bench_newton.py --label port_device "${common[@]}"
"${PYTHON}" bench_newton.py --label port_host "${common[@]}" --var hostfix=1
"${PYTHON}" bench_newton.py --label port_ref --newton 0 "${common[@]}"
