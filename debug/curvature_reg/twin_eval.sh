#!/bin/bash
# Usage: twin_eval.sh <twin> <step> <gpu> -- the short-range reading of a production twin
# at one staged EMA checkpoint (runs/<twin>/ema_<step>.pt), next to the plain run's
# checkpoint of the same step when it still exists: the contact-anchor probe on the three probed frames
# (Cu-I 16547, Ca-H 6048, Ag-Ag 17311; full frame, dilated 1.3x, 90% deleted, pair
# alone), the hardest-contact series over the preceding 8000 steps, and the H2 and
# O2 dimers.  Writes runs/<twin>/twin_eval_<step>.done when finished.
r=$1
s=$2
g=$3
P=/nas/outisli/Software/miniforge3/envs/dpmd/bin/python
PLAIN=/nas/outisli/Storage/ckpts/DPA4/OMat24/Neo/N_2-s_3-c_32x2
D=$(cd "$(dirname "$0")" && pwd)
cd $D
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4
# The plain run's checkpoint of the same step is the comparison column when it
# still exists (its checkpoints rotate); otherwise the twin is probed alone.
plain=$PLAIN/model_ema.ckpt-$s.pt
[ -f "$plain" ] || plain=""
for f in 16547 6048 17311; do
	$P -u contact_anchor_probe.py $f runs/$r/ema_$s.pt $plain --drop 0.9 --dilate 1.3 >runs/$r/contact_anchor_probe_${f}_$s.log 2>&1
done
steps=$(seq $((s - 8000)) 2000 $s | tr '\n' ' ')
$P -u hard_contact_series.py runs/$r --pattern 'ema_{step}.pt' --steps $steps >>runs/$r/hard_contact_series.log 2>&1
for e in H O; do
	$P -u dimer_scan.py runs/$r/ema_$s.pt --element $e --out runs/$r/dimer_${e}_$s.npz >runs/$r/dimer_${e}_$s.log 2>&1
done
# The isolated-pair survey over 24 element pairs (section 10.13.12): where each pair's radial force turns attractive inside
# 0.6 covalent contacts, the wall at 0.40 and 0.50 contacts, and any hole.
$P -u dimer_survey.py runs/$r/ema_$s.pt "$r $s" 2>&1 | grep -vE "^\s*$|Warning|warn|recommended|patches installed" >runs/$r/survey_$s.log
# The compressed-probe gain (ledger E33): the early gate on the live checkpoint of the same step, on the CPU.
DP_CUDA_TRAIN=0 DP_TRITON_TRAIN=0 CUDA_VISIBLE_DEVICES= $P -u gain_probe.py $r:$s >runs/$r/gain_probe_$s.log 2>&1
touch runs/$r/twin_eval_$s.done
