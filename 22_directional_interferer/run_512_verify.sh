#!/bin/bash
# Pre-training check 3: does the warm-started model, before any training,
# reproduce the recorded MBSTOI 0.8535? Checks 1 and 2 (tensor count, shapes)
# already passed. Both earlier retrain attempts skipped this and burned a run.
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
cd "$PKG/code512"
CUDA_VISIBLE_DEVICES="${GPU_ID:-3}" PYTHONUNBUFFERED=1 "$PY" eval.py \
  --model_checkpoint /raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt \
  --vctk_test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
  --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir "$PKG/results/verify512_identity" \
  --paper_style_eval --batch_size 8 --limit_pairs 750
echo "VERIFY512_EXIT=$?"
echo "VERIFY512_DONE"
