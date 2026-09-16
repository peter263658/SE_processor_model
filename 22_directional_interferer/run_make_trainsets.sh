#!/bin/bash
# Build directional train/val sets so the models can be retrained for the
# condition (only needed if the zero-shot directional eval comes back a tie).
# One mix per clean file, reusing that file's own SNR from the isotropic set,
# so only the interferer's spatial distribution changes.
#
#   bash run_make_trainsets.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG"

for split in val train; do
  echo "=== generating directional noise ${split} set ==="
  "$PY" scripts/make_directional_testset.py --mode noise --separation 90 \
      --split "$split" --limit 0
  echo "GEN_${split}_EXIT=$?"
done
echo "TRAINSETS_DONE"
