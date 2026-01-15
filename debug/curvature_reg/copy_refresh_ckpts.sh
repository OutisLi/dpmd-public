#!/bin/bash
# Usage: copy_refresh_ckpts.sh <run>... — while any production refresh twin
# trains, copy every rotating EMA checkpoint of the named runs to
# runs/<run>/ema_<step>.pt so that the whole series stays available; one
# final pass after the last trainer exits.
D=$(cd "$(dirname "$0")" && pwd)
cd $D
copy_all() {
	for r in "$@"; do
		for f in runs/$r/ckpt/model_ema.ckpt-*.pt; do
			[ -f "$f" ] || continue
			s=${f##*ckpt-}
			s=${s%.pt}
			[ -f runs/$r/ema_$s.pt ] || cp $f runs/$r/ema_$s.pt
		done
	done
}
while pgrep -f "train_refresh.py input.json" >/dev/null; do
	copy_all "$@"
	sleep 120
done
copy_all "$@"
