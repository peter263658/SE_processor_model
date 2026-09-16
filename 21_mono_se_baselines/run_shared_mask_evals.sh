#!/bin/bash
# Experiment (A): bilateral eval with one mask shared by both ears.
# Answers the reviewer counter-proposal "a mono net can keep ILD by sharing a
# single mask across the ears" -- run on the same考卷 as the independent-mask
# rows so the two are directly comparable.
#
#   GPU_ID=3 bash run_shared_mask_evals.sh
set -euo pipefail
PKG=/raid/R12K41024/21_mono_se_baselines
GPU_ID="${GPU_ID:-3}"

for tag in ntu snu; do
  echo "=== shared-mask bilateral eval: ${tag} ==="
  CKPT="$PKG/results/${tag}/best_${tag}.pt" TAG="${tag}_shared" \
    GPU_ID="$GPU_ID" MASK_MODE=shared bash "$PKG/run_bilateral_eval.sh"
done
