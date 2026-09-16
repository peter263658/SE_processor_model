#!/bin/bash
set -uo pipefail
S=/raid/R12K41024/22_directional_interferer/code512
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
CKPT=/raid/R12K41024/22_directional_interferer/outputs/scratch512_dirnoise90_2026-08-27_13-49-48/logs/lightning_logs/version_0/checkpoints/epoch=58-step=57112.ckpt
DATA=/raid/R12K41024/BCCTN/Dataset
cd "$S"
for cond in dirnoise90 isotropic; do
  if [ "$cond" = isotropic ]; then TEST="$DATA/noisy_testset"; else TEST="$DATA/noisy_testset_$cond"; fi
  echo "===== eval_min on $cond ====="
  CUDA_VISIBLE_DEVICES="${GPU_ID:-3}" "$PY" eval_min.py --ckpt "$CKPT" \
    --test_dir "$TEST" --clean_dir "$DATA/clean_testset" \
    --out /raid/R12K41024/22_directional_interferer/results/scratch512_min_$cond --limit 750 --workers 32
  echo "MINEVAL_${cond}_EXIT=$?"
done
echo "MIN_EVAL_ALL_DONE"
