#!/usr/bin/env bash
# Curvature ACCV sweep: matched A/B (3 seeds) + gradient control + gamma sweep.
# Sequential, unattended, RESUMABLE (skips runs whose checkpoint-best.pth exists).
#
#   nohup bash curv/server/run_sweep.sh > sweep.log 2>&1 &
#   tail -f sweep.log
#
# Tune EPOCHS to fit your time budget: measure one run first; if a single run
# exceeds ~6h, set EPOCHS=8 (or trim the run list at the bottom -- the 3-seed
# g0/g1 core is the priority; control next; gamma-sweep is optional).
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

# Even slate: every condition at five seeds. run() skips any run that already has
# a checkpoint-best.pth, so a rerun only fills the gaps (currently grad s3-4,
# g0p5 s1-4, g2 s1-4 = 10 runs, ~3-4 days). Seed-outer ordering finishes the
# missing seeds of each condition together.
for s in 0 1 2 3 4; do
  run "$(C 0.0)" "$s" "g0_s$s"      # baseline (gamma=0 == ConfLoss)
  run "$(C 1.0)" "$s" "g1_s$s"      # curvature (gamma=1)
  run "$(G 1.0)" "$s" "grad_s$s"    # gradient control (1st-order)
  run "$(C 0.5)" "$s" "g0p5_s$s"    # gamma=0.5
  run "$(C 2.0)" "$s" "g2_s$s"      # gamma=2
done

echo "ALL DONE"
