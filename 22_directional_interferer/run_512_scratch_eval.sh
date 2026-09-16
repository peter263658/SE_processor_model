#!/bin/bash
# Evaluate the from-scratch causal-512 model on its training condition
# (dirnoise90, against NTU as-published 0.8144) and on isotropic (generalisation
# / regression check -- confirm it is not only useful on directional data).
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
DATA=/raid/R12K41024/BCCTN/Dataset
CKPT="$PKG/outputs/scratch512_dirnoise90_2026-08-27_13-49-48/logs/lightning_logs/version_0/checkpoints/epoch=58-step=57112.ckpt"
GPU="${GPU_ID:-3}"
cd "$PKG/code512"
for cond in dirnoise90 isotropic; do
  if [ "$cond" = isotropic ]; then TEST="$DATA/noisy_testset"; else TEST="$DATA/noisy_testset_$cond"; fi
  echo "=== eval scratch-512 on $cond ==="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" eval.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir "$TEST" --vctk_clean_dir "$DATA/clean_testset" \
    --output_dir "$PKG/results/scratch512_eval_$cond" \
    --paper_style_eval --batch_size 8 --limit_pairs 750
  echo "EVAL_SCRATCH512_${cond}_EXIT=$?"
done
echo "SCRATCH512_EVAL_DONE"
