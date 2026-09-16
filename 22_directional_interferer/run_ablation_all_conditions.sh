#!/bin/bash
# Westhausen-style ablation applied to OUR model: same weights, same architecture,
# the only variable is whether the model sees the contralateral ear.
#   L_enh = model(stack([L, L]))[:, 0]   <- ipsilateral only
# This avoids the confound that wrecked the cross-architecture comparison
# (BCCTN vs NTU CNN differ in params, FFT size, causality and loss, which swamps
# the binaural effect).
#
#   GPU_ID=3 bash run_ablation_all_conditions.sh
set -uo pipefail
DEP=/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
GPU="${GPU_ID:-3}"
for cond in dirspeech2x60_s0 dirnoise2x60 dirspeech2x60 dirnoise90 dirspeech90; do
  echo "=== ablation: $cond ==="
  COND="$cond" TAG="ablation_$cond" GPU_ID="$GPU" LIMIT=600 WORKERS=32 \
    bash "$DEP/run_bilateral_ablation.sh"
  echo "ABLATION_${cond}_EXIT=$?"
done
echo "ABLATION_ALL_DONE"
