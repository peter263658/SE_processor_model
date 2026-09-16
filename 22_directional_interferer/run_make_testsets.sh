#!/bin/bash
# Build both directional-interferer test sets (CPU only).
#   bash run_make_testsets.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG"

for mode in noise speech; do
  echo "=== generating directional ${mode}, 90 deg separation ==="
  "$PY" scripts/make_directional_testset.py --mode "$mode" --separation 90 --limit 800
  echo "GEN_${mode}_EXIT=$?"
done
echo "TESTSETS_DONE"
