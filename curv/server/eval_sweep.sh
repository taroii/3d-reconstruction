#!/usr/bin/env bash
# Eval every trained checkpoint (+ base) on Sintel (zero-shot) and PO-val
# (in-dist diagnostic). Run AFTER training, from anywhere:
#   bash curv/server/eval_sweep.sh > eval_all.log 2>&1
# Then read eval_all.log; aggregate seeds (g0_s0/s1/s2 ...) by hand or share it.
set -u
cd "$(dirname "$0")/.."        # -> curv/

PO="400 @ PointOdysseyDUSt3R(dset='val', dataset_location='../data/pointodyssey', S=2, strides=[4], resolution=(512,288))"
SINTEL="300 @ SintelDUSt3R(dataset_location='../data/training', dset='clean', S=2, strides=[7], resolution=(512,224), load_dynamic_mask=False)"
DDR=../DDUSt3R

CKPTS=("$DDR/checkpoints/ddust3r.pth")
for d in "$DDR"/results/*/checkpoint-best.pth; do
  [ -f "$d" ] && CKPTS+=("$d")
done

for ck in "${CKPTS[@]}"; do
  echo "############################################################"
  echo "# $ck"
  echo "############################################################"
  python eval_depth.py --ckpt "$ck" --dataset "$PO"
  python eval_depth.py --ckpt "$ck" --dataset "$SINTEL"
done
echo "EVAL DONE"
