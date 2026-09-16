#!/bin/bash
# The properly matched Westhausen S0N+-60 condition: target in FRONT, two
# competing talkers at +-60 deg.
#
# Why frontal matters: with two competing talkers and no steering input, the
# task is ill-posed unless the target has a consistent spatial prior -- the same
# -30 deg voice must be kept when it is the target and suppressed when it is an
# interferer.  Westhausen fix the target at 0 deg (S0), which makes "enhance the
# frontal talker" learnable.  Our dirspeech2x60 lets the target roam over +-30
# deg and is kept as the ill-posed contrast.
#
# |az| <= 5 deg keeps 815 test targets (>= the 750 the eval uses, so the pair
# count matches every other condition) and 3784 train targets.
#
#   bash run_make_s0.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG"

echo "=== S0N+-60 test set ==="
"$PY" scripts/make_directional_testset.py --mode speech --n_interferers 2 \
    --separation 60 --max_target_az 5 --limit 0 --interferer_pool 500 --tag_suffix _s0
echo "GEN_S0_test_EXIT=$?"
for split in val train; do
  echo "=== S0N+-60 ${split} set ==="
  "$PY" scripts/make_directional_testset.py --mode speech --n_interferers 2 \
      --separation 60 --max_target_az 5 --split "$split" --limit 0 \
      --interferer_pool 500 --tag_suffix _s0
  echo "GEN_S0_${split}_EXIT=$?"
done
echo "S0_DONE"
