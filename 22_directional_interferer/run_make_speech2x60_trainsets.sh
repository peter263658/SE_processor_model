#!/bin/bash
# Training data matched to the dirspeech2x60 test condition, so the retrain
# actually sees what it will be scored on.  A bounded interferer pool mirrors
# Westhausen et al.'s single fixed IFFM masker.
#
#   bash run_make_speech2x60_trainsets.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG"
for split in val train; do
  echo "=== dirspeech2x60 ${split} set ==="
  "$PY" scripts/make_directional_testset.py --mode speech --n_interferers 2 \
      --separation 60 --max_target_az 30 --split "$split" --limit 0 \
      --interferer_pool 500
  echo "GEN_S2X60_${split}_EXIT=$?"
done
echo "SPEECH2X60_TRAINSETS_DONE"
