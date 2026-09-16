#!/bin/bash
# Retrain on the S0N+-60 condition and re-score, to answer: can this binaural
# architecture learn spatial filtering when the training data actually contains
# a spatially concentrated interferer?
#
# The isotropic-trained models learn diffuse-noise statistics ("energy that is
# decorrelated between the ears is noise"), a prior that inverts when the
# interferer is spatially compact and therefore highly inter-ear correlated.
#
# Fairness: the NTU baseline is retrained on the SAME data for the SAME budget.
# Vehicle is the FP32 binaural ancestor, not the deploy checkpoint, so the frozen
# W4A8 gates (VCTK 0.8357 / TIMIT 0.7898) are untouched.
#
#   GPU_ID=3 bash run_s0_retrain.sh
set -uo pipefail
PKG=/raid/R12K41024/22_directional_interferer
MONO=/raid/R12K41024/21_mono_se_baselines
DATA=/raid/R12K41024/BCCTN/Dataset
FP32=/raid/R12K41024/09_weight_tied_mha
CKPT="$FP32/outputs/finetune/2026-05-09_22-29-20/logs/lightning_logs/version_0/checkpoints/epoch=57-step=56144.ckpt"
PY=/home/R12K41024/miniforge3/envs/bcctn/bin/python
GPU="${GPU_ID:-3}"
EPOCHS="${EPOCHS:-40}"
TRAIN="$DATA/noisy_trainset_dirspeech2x60_s0"
VAL="$DATA/noisy_valset_dirspeech2x60_s0"

echo "=== [1/4] FP32 binaural finetune on S0N+-60 ==="
cd "$FP32"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="${PYTHONPATH:-}:$FP32" HYDRA_FULL_ERROR=1 \
  "$PY" -u train.py \
    training.train_checkpoint_path="'$CKPT'" \
    training.learning_rate=1e-4 \
    training.n_epochs="$EPOCHS" \
    dataset.noisy_training_dataset_dir="$TRAIN" \
    dataset.noisy_validation_dataset_dir="$VAL" \
    hydra.run.dir="$PKG/outputs/s0_finetune_$(date +%Y-%m-%d_%H-%M-%S)"
echo "FP32_S0_TRAIN_EXIT=$?"

echo "=== [2/4] NTU baseline retrained on the same data, same budget ==="
cd "$MONO/code"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" train_mono.py --arch ntu \
    --noisy_train_dir "$TRAIN" --noisy_val_dir "$VAL" \
    --init_from "$MONO/results/ntu/best_ntu.pt" \
    --epochs "$EPOCHS" --output_dir "$MONO/results/ntu_s0"
echo "NTU_S0_TRAIN_EXIT=$?"

echo "=== [3/4] retrained NTU, both mask modes, on S0N+-60 ==="
for variant in shared independent; do
  CKPT="$MONO/results/ntu_s0/best_ntu.pt" TAG="dirspeech2x60_s0_ntuS0_${variant}" \
  VCTK_ONLY=1 MASK_MODE="$variant" \
  VCTK_TEST_DIR="$DATA/noisy_testset_dirspeech2x60_s0" GPU_ID="$GPU" \
    bash "$MONO/run_bilateral_eval.sh"
  echo "NTU_S0_EVAL_${variant}_EXIT=$?"
done

echo "=== [4/4] retrained FP32 binaural on S0N+-60 -- fill in the new ckpt path ==="
echo "see $PKG/outputs/s0_finetune_*/ for the checkpoint, then:"
echo "  COND=dirspeech2x60_s0 CKPT=<new.ckpt> bash run_fp32_control.sh"
echo "S0_RETRAIN_DONE"
