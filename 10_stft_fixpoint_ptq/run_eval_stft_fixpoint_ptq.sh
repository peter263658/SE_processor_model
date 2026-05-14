#!/bin/bash
# Eval the supplied strict W4A8 checkpoint with fixed-point STFT/iSTFT PTQ.
#
# Usage:
#   ./run_eval_stft_fixpoint_ptq.sh [checkpoint_path] [output_dir] [gpu_id] [hydra overrides...]
#
# Examples:
#   ./run_eval_stft_fixpoint_ptq.sh
#   ./run_eval_stft_fixpoint_ptq.sh "$CKPT" results/q16_10 5 model.stft_fixed_point.spectrum_frac_bits=12
set -e
cd "$(dirname "$0")"

DEFAULT_CHECKPOINT="/raid/R12K41024/09_weight_tied_mha/outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt"
CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
OUTPUT_DIR="${2:-results/stft_fixpoint_ptq}"
GPU_ID="${3:-5}"
EXTRA_OVERRIDES=("${@:4}")

VCTK_NOISY_DIR="/raid/R12K41024/BCCTN/Dataset/noisy_testset"
VCTK_CLEAN_DIR="/raid/R12K41024/BCCTN/Dataset/clean_testset"

mkdir -p "$OUTPUT_DIR"
mkdir -p /tmp/numba_cache /tmp/mpl_cache

echo "STFT/iSTFT fixed-point PTQ eval"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPU         : $GPU_ID"
echo "  Datasets    : VCTK only"
echo "  Precision   : config/model.stft_fixed_point plus optional overrides"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n bcctn python eval.py \
    --model_checkpoint "$CHECKPOINT" \
    --vctk_test_dir "$VCTK_NOISY_DIR" \
    --vctk_clean_dir "$VCTK_CLEAN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --paper_style_eval \
    --batch_size 8 \
    --limit_pairs 750 \
    --hydra_overrides \
        training.qat_scope=full \
        training.top_k_attention=4 \
        model.stft_fixed_point.enabled=true \
        "${EXTRA_OVERRIDES[@]}"
