#!/bin/bash
# FP32 binaural control on the directional conditions.
#
# The deploy model is W4A8 with *statically* calibrated per-tensor pow2 shared
# exponents, and that calibration was measured on the isotropic set.  On an
# out-of-distribution input those fixed scales can saturate, which would look
# exactly like "the architecture cannot do spatial filtering".  The FP32
# ancestor has no static scales, so running it separates the two causes.
#
#   GPU_ID=3 bash run_fp32_control.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
MONO=/raid/R12K41024/21_mono_se_baselines
DATA=/raid/R12K41024/BCCTN/Dataset
CKPT="${CKPT:-/raid/R12K41024/09_weight_tied_mha/outputs/finetune/2026-05-09_22-29-20/logs/lightning_logs/version_0/checkpoints/epoch=57-step=56144.ckpt}"
TAG="${TAG:-fp32_binaural}"
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
GPU="${GPU_ID:-3}"
CONDS="${COND:-dirspeech2x60 dirnoise2x60}"
# "isotropic" maps to the original考卷 so a retrained model can be checked for
# regression on the condition the deploy gates were measured on.

cd "$MONO/code"
for cond in $CONDS; do
  if [ "$cond" = isotropic ]; then TEST_DIR="$DATA/noisy_testset";
  else TEST_DIR="$DATA/noisy_testset_$cond"; fi
  [ -d "$TEST_DIR" ] || { echo "SKIP $cond"; continue; }
  OUT="$MONO/results/bilateral_eval/${cond}_${TAG}"
  mkdir -p "$OUT"
  echo "=== [$cond] FP32 binaural control ==="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" eval_fp32.py \
    --model_checkpoint "$CKPT" \
    --vctk_test_dir "$TEST_DIR" \
    --vctk_clean_dir "$DATA/clean_testset" \
    --output_dir "$OUT" \
    --paper_style_eval --batch_size 8 --limit_pairs 750 --metric_workers 32 \
    2>&1 | tee "$PKG/logs/${TAG}_${cond}.log" | grep -aE "MBSTOI Enhanced|ILD Error Enhanced" || true
  echo "FP32_${TAG}_${cond}_EXIT=$?"
done
echo "FP32_CONTROL_DONE"
