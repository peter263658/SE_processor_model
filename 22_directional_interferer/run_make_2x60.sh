#!/bin/bash
# The decisive conditions: two interferers flanking the target at +-60 deg, the
# Westhausen et al. (TASLP 2024) S0N+-60 configuration.  Measured better-ear
# advantage drops to 2.16 dB (vs 8.8-10.9 dB with a single lateral interferer),
# so better-ear listening can no longer carry the score.
#
#   bash run_make_2x60.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG"

for mode in noise speech; do
  echo "=== generating ${mode}, 2 interferers at +-60 deg ==="
  "$PY" scripts/make_directional_testset.py --mode "$mode" --n_interferers 2 \
      --separation 60 --max_target_az 30 --limit 800
  echo "GEN_2x60_${mode}_EXIT=$?"
done
echo "2X60_DONE"
