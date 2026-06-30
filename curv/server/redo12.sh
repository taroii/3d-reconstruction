#!/usr/bin/env bash
# Standardize on 12 epochs. Runs exactly the 9 runs needed to make the headline
# (g0/g1) and the gradient control 5-seed/12-epoch, after the 7 stale 10-epoch
# dirs have been deleted (see the delete command in the instructions). The
# gamma sweep (g0p5, g2) is left at its 2 already-12-epoch seeds and is NOT run
# here. Same recipe as run_sweep.sh; EPOCHS defaults to 12.
#
#   nohup bash curv/server/redo12.sh > redo12.log 2>&1 &
#   tail -f redo12.log
set -u
cd "$(dirname "$0")/../../DDUSt3R" || exit 1

EPOCHS=${EPOCHS:-12}
LR=${LR:-3e-5}
PRE=checkpoints/ddust3r.pth
TRAIN="6000 @ PointOdysseyDUSt3R(dset='train', dataset_location='../data/pointodyssey', S=2, strides=[2,4,6,8], resolution=(512,288), aug_crop=16) + 2000 @ TarTanAirDUSt3R(dset='Easy', dataset_location='../data/tartanair', S=2, strides=[8], resolution=(512,288), aug_crop=16)"
TEST="500 @ PointOdysseyDUSt3R(dset='val', dataset_location='../data/pointodyssey', S=2, strides=[4], resolution=(512,288))"
BASECONF="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"

run () {  # $1=train_criterion  $2=seed  $3=name
  local out="results/$3"
  if [ -f "$out/checkpoint-best.pth" ]; then echo "SKIP $3 (already done)"; return; fi
  echo "=== RUN $3  seed=$2  epochs=$EPOCHS  lr=$LR ==="
  python launch.py --pretrained "$PRE" \
    --train_criterion "$1" --test_criterion "$BASECONF" \
    --train_dataset "$TRAIN" --test_dataset "$TEST" \
    --seed "$2" --batch_size 2 --accum_iter 8 --epochs "$EPOCHS" \
    --lr "$LR" --min_lr 1e-6 --warmup_epochs 1 \
    --amp 1 --num_workers 6 --output_dir "$out" --save_freq "$EPOCHS" --keep_freq "$EPOCHS" \
    > "results/$3.log" 2>&1
}
C () { echo "CurvWeightedConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, gamma=$1)"; }
G () { echo "GradWeightedConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, gamma=$1)"; }

# headline reruns: g0/g1 seeds 0-2 (the deleted 10-epoch ones), now at 12
for s in 0 1 2; do
  run "$(C 0.0)" "$s" "g0_s$s"
  run "$(C 1.0)" "$s" "g1_s$s"
done
# gradient control to 5 seeds at 12: s0 reran (deleted), s3/s4 are new
run "$(G 1.0)" 0 "grad_s0"
run "$(G 1.0)" 3 "grad_s3"
run "$(G 1.0)" 4 "grad_s4"

echo "REDO DONE"
