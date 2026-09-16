#!/bin/bash
# Experiment (B): step the NTU JSSC'20 baseline down from the paradigm ceiling
# to the training regime the paper actually reports (Sec. VI-B).
#
# Ladder (each row adds one constraint; existing results/ntu is the top row):
#   ntu        SI-SDR loss, 30,969 mixes, -7..16 dB   <- favorable upper bound
#   ntu_mse    MSE loss,    30,969 mixes, -7..16 dB   <- isolates the loss
#   ntu_aspub  MSE loss,       694 mixes, ~0 dB only  <- as-published regime
#
# The paper trains on 620 utterances mixed at a single 0 dB SNR; |SNR| <= 0.2 dB
# selects 694 of our mixes, so the data scale and the single-SNR constraint both
# land close to the published numbers.
#
# Strictly sequential, one GPU process at a time -- concurrent train+eval gets
# SIGTERMed on this box (see run_full_pipeline.sh).
#
#   GPU_ID=3 bash run_aspub_regime.sh
set -uo pipefail

PKG=/raid/R12K41024/21_mono_se_baselines
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
GPU="${GPU_ID:-3}"
cd "$PKG/code"

echo "=== [1/4] train ntu_mse: MSE loss, full data ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  "$PY" train_mono.py --arch ntu --loss mse --output_dir "$PKG/results/ntu_mse"
rc=$?; echo "NTU_MSE_TRAIN_EXIT=$rc"; [ $rc -ne 0 ] && exit $rc

# 694 mixes = 10 batches/epoch, so the epoch budget is raised to make sure the
# small-data run converges rather than merely being cut short.
echo "=== [2/4] train ntu_aspub: MSE loss, 694 mixes at ~0 dB ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  "$PY" train_mono.py --arch ntu --loss mse --snr_band 0.2 \
    --epochs 400 --patience 40 --output_dir "$PKG/results/ntu_aspub"
rc=$?; echo "NTU_ASPUB_TRAIN_EXIT=$rc"; [ $rc -ne 0 ] && exit $rc

echo "=== [3/4] bilateral eval ntu_mse ==="
CKPT="$PKG/results/ntu_mse/best_ntu.pt" TAG=ntu_mse GPU_ID=$GPU \
  bash "$PKG/run_bilateral_eval.sh"
echo "NTU_MSE_EVAL_EXIT=$?"

echo "=== [4/4] bilateral eval ntu_aspub ==="
CKPT="$PKG/results/ntu_aspub/best_ntu.pt" TAG=ntu_aspub GPU_ID=$GPU \
  bash "$PKG/run_bilateral_eval.sh"
echo "NTU_ASPUB_EVAL_EXIT=$?"

echo "ASPUB_REGIME_DONE"
