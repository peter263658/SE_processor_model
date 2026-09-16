#!/bin/bash
# Retrain on the directional dataset and see whether the gap widens.
#
# Vehicle is the ORIGINAL full-size BCCTN (win 400 / fft 512 / untied input_proj
# MHA, MBSTOI 0.8535 on the isotropic考卷) rather than the deploy lineage. The
# deploy FP32 ancestor already carries tied MHA, grouped pointwise, 128-pt FFT and
# a causal sliding window; fine-tuning it onto a new condition destroyed it
# (0.2765 on isotropic, see RESULTS_DIRECTIONAL_INTERFERER §2.4). The original
# model has none of those constraints.
#
# Baseline arm is NTU at its **as-published** regime (magnitude MSE, the 694
# mixes at |SNR| <= 0.2 dB standing in for the paper's 620 utterances at 0 dB),
# trained on the same directional data. The data-size asymmetry is deliberate:
# it is what that paper reports, so this is chip-versus-chip, not a controlled
# ablation -- label it that way.
#
#   GPU_ID=3 [EPOCHS=15] bash run_512_directional_retrain.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
S512=/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/code
MONO=/raid/R12K41024/21_mono_se_baselines
DATA=/raid/R12K41024/BCCTN/Dataset
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
CKPT512=/raid/R12K41024/BCCTN/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt
GPU="${GPU_ID:-3}"; EPOCHS="${EPOCHS:-15}"
COND="${COND:-dirnoise90}"
TRAIN="$DATA/noisy_trainset_$COND"; VAL="$DATA/noisy_valset_$COND"
TEST="$DATA/noisy_testset_$COND"
OUTD="$PKG/outputs/retrain512_${COND}_$(date +%Y-%m-%d_%H-%M-%S)"

echo "=== [1/4] FP32-512 finetune on $COND (full $(find "$TRAIN" -name "*.wav" | wc -l) mixes) ==="
cd "$S512"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="${PYTHONPATH:-}:$S512" HYDRA_FULL_ERROR=1 \
  "$PY" -u train.py \
    training.train_checkpoint_path="'$CKPT512'" \
    training.learning_rate=5e-5 \
    training.n_epochs="$EPOCHS" \
    dataset.noisy_training_dataset_dir="$TRAIN" \
    dataset.noisy_validation_dataset_dir="$VAL" \
    hydra.run.dir="$OUTD"
echo "TRAIN512_EXIT=$?"

echo "=== [2/4] NTU as-published regime on the same directional data ==="
cd "$MONO/code"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" train_mono.py --arch ntu \
    --loss mse --snr_band 0.2 \
    --noisy_train_dir "$TRAIN" --noisy_val_dir "$VAL" \
    --epochs 400 --patience 40 --output_dir "$MONO/results/ntu_aspub_$COND"
echo "TRAIN_NTU_EXIT=$?"

echo "=== [3/4] eval NTU as-published on $COND ==="
CKPT="$MONO/results/ntu_aspub_$COND/best_ntu.pt" TAG="${COND}_ntu_aspub" \
VCTK_ONLY=1 MASK_MODE=independent VCTK_TEST_DIR="$TEST" GPU_ID="$GPU" \
  bash "$MONO/run_bilateral_eval.sh"
echo "EVAL_NTU_EXIT=$?"

echo "=== [4/4] retrained FP32-512 checkpoint ==="
find "$OUTD" -name "*.ckpt" | head -5
echo "RETRAIN512_DONE"
