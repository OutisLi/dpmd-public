#!/bin/bash
# Usage: twin_eval_chain.sh <twin> <gpu> <step>... -- the evaluation loop of a production twin:
# wait for each staged EMA checkpoint (runs/<twin>/ema_<step>.pt, written by copy_refresh_ckpts.sh),
# then run twin_eval.sh on it (contact probes, hardest-contact series, H2 and O2 dimers).
r=$1
g=$2
shift 2
D=$(cd "$(dirname "$0")" && pwd)
cd $D
for s in "$@"; do
	until [ -f runs/$r/ema_$s.pt ]; do sleep 120; done
	sleep 30
	./twin_eval.sh $r $s $g >runs/$r/twin_eval_$s.out 2>&1
done
