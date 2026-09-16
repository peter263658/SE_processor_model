#!/bin/bash
# Strictly sequential single-GPU pipeline: SNU train -> SNU eval -> NTU eval.
# One GPU process at a time (concurrent train+eval got SIGTERMed on this box).
set -uo pipefail

PKG=/raid/R12K41024/21_mono_se_baselines
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
GPU="${GPU_ID:-1}"

echo "=== [1/3] SNU training ==="
cd "$PKG/code"
CUDA_VISIBLE_DEVICES=$GPU PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  "$PY" train_mono.py --arch snu --output_dir ../results/snu
rc=$?
echo "SNU_TRAIN_EXIT=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=== [2/3] SNU bilateral eval ==="
CKPT="$PKG/results/snu/best_snu.pt" TAG=snu GPU_ID=$GPU bash "$PKG/run_bilateral_eval.sh"
echo "SNU_EVAL_EXIT=$?"

echo "=== [3/3] NTU bilateral eval ==="
CKPT="$PKG/results/ntu/best_ntu.pt" TAG=ntu GPU_ID=$GPU bash "$PKG/run_bilateral_eval.sh"
echo "NTU_EVAL_EXIT=$?"

echo "PIPELINE_DONE"
